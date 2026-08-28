import copy
import json
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
import uuid   
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from transformers import ViTModel, ViTImageProcessor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score, accuracy_score


# 1. DATASET WITH AUGMENTATION 
class PersonalEmotionDataset(Dataset):
    def __init__(self, samples, emotion_names, processor, augment=False):
        self.samples = samples
        self.emotion_names = emotion_names
        self.processor = processor
        self.augment = augment

        # Define Augmentations (Train only)
        # We use PIL transforms because they are standard for ViT
        self.transform_ops = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomResizedCrop(size=(224, 224), scale=(0.85, 1.0), ratio=(0.9, 1.1))
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        frame_bgr = sample["frame"]
        label = sample["true_emotion"]

        # Convert BGR (OpenCV) to RGB (PIL)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        # Apply augmentation if enabled (Training set)
        if self.augment:
            pil_image = self.transform_ops(pil_image)

        # Processor handles normalization and resizing
        inputs = self.processor(images=pil_image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)

        emotion_idx = self.emotion_names.index(label)

        return {
            "pixel_values": pixel_values,
            "emotion_labels": torch.tensor(emotion_idx, dtype=torch.long),
        }


#   2. THE MODEL CLASS  
class PersonalizedEmotionViT(nn.Module):
    def __init__(self, num_emotions=7, freeze_base=True, dropout_p=0.25):
        super().__init__()
        # Load Processor and Base Model
        self.processor = ViTImageProcessor.from_pretrained("WinKawaks/vit-tiny-patch16-224")
        self.base_vit = ViTModel.from_pretrained(
            "WinKawaks/vit-tiny-patch16-224",
            add_pooling_layer=False
        )
        hidden_size = 192

        # Freeze the heavy ViT backbone?
        if freeze_base:
            for p in self.base_vit.parameters():
                p.requires_grad = False

        # Adapter Layers
        self.adapter = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout_p),
        )
        
        # Heads
        self.emotion_head = nn.Linear(128, num_emotions)
        self.intensity_head = nn.Linear(128, 1)
        self.valence_arousal_head = nn.Linear(128, 2)

        self.emotion_names = ["happy", "sad", "angry", "fear", "surprise", "disgust", "neutral"]
        
        # NEW: Temperature Parameter for Calibration (Learnable but separate from main training)
        self.temperature = nn.Parameter(torch.ones(1) * 1.0, requires_grad=False)

    def forward(self, pixel_values):
        out = self.base_vit(pixel_values=pixel_values)
        cls = out.last_hidden_state[:, 0] # Take CLS token
        z = self.adapter(cls)

        return {
            "emotion_logits": self.emotion_head(z),
            "intensity": torch.sigmoid(self.intensity_head(z)),
            "valence_arousal": torch.tanh(self.valence_arousal_head(z)),
        }

    def predict_frame(self, frame_bgr):
        """
        Inference helper for ROS. 
        Takes BGR frame -> Returns calibrated dictionary.
        """
        self.eval()
        device = next(self.parameters()).device
        
        with torch.no_grad():
            # Preprocess
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            inputs = self.processor(images=frame_rgb, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
 
            # Forward Pass
            outputs = self.forward(pixel_values)
            logits = outputs["emotion_logits"]
            
            # Calibration: Apply Temperature Scaling
            T = torch.clamp(self.temperature, 0.05, 10.0)
            probs = F.softmax(logits / T, dim=-1)
            
            # Get Result
            idx = probs.argmax(dim=-1).item()
            
            return {
                "primary_emotion": self.emotion_names[idx],
                "confidence": float(probs[0, idx].item()),
                "raw_logits": logits.cpu().numpy(),
                "temperature_used": float(T.item())
            }


#   3. LEARNING SYSTEM  
class InteractiveLearningSystem:
    def __init__(self, model: PersonalizedEmotionViT, storage_path="./memory/cara_personal_data"):
        self.model = model
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.processor = self.model.processor

  
    def save_labeled_sample(self, frame_bgr, label):
        ts = datetime.now()
        # Add a short random ID (first 4 chars of a UUID) to ensure uniqueness
        unique_id = str(uuid.uuid4())[:4]
        
        # New format: YYYYMMDD_HHMMSS_microseconds_randomID
        filename = f"{ts.strftime('%Y%m%d_%H%M%S_%f')}_{unique_id}"

        img_path = self.storage_path / f"{filename}.jpg"
        meta_path = self.storage_path / f"{filename}.json"

        # The rest is perfect...
        cv2.imwrite(str(img_path), frame_bgr)
        meta = {"true_emotion": label, "timestamp": ts.isoformat()}
        with open(meta_path, "w") as f:
            json.dump(meta, f)
            
            
    def load_samples_with_groups(self):
        samples = []
        for meta_path in self.storage_path.glob("*.json"):
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                img_path = meta_path.with_suffix(".jpg")
                if not img_path.exists():
                    continue
                frame = cv2.imread(str(img_path))
                if frame is None:
                    continue

                # group by day to avoid leakage
                day = meta["timestamp"][:10]  # YYYY-MM-DD
                samples.append({
                    "frame": frame,
                    "true_emotion": meta["true_emotion"],
                    "group": day,
                })
            except Exception:
                continue
        return samples

    def fit_temperature(self, val_loader):
        """Fits temperature T on validation set by minimizing NLL."""
        print("Fitting temperature scaling...")
        device = next(self.model.parameters()).device
        self.model.eval()

        all_logits = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["emotion_labels"].to(device)
                out = self.model(pixel_values)
                all_logits.append(out["emotion_logits"])
                all_labels.append(labels)

        if not all_logits: return 1.0

        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)

        # Optimize T (scalar)
        T = torch.ones(1, device=device, requires_grad=True)
        optimizer = torch.optim.LBFGS([T], lr=0.01, max_iter=50)

        def closure():
            optimizer.zero_grad()
            loss = F.cross_entropy(logits / torch.clamp(T, 0.05, 10.0), labels)
            loss.backward()
            return loss

        optimizer.step(closure)

        final_T = torch.clamp(T.detach(), 0.05, 10.0)
        self.model.temperature.copy_(final_T)
        print(f"Temperature fitted to: {final_T.item():.4f}")
        return float(final_T.item())

    @staticmethod
    def _eval_epoch(model, dataloader, device):
        model.eval()
        all_y = []
        all_hat = []

        with torch.no_grad():
            for batch in dataloader:
                x = batch["pixel_values"].to(device)
                y = batch["emotion_labels"].to(device)

                logits = model(x)["emotion_logits"]
                hat = logits.argmax(dim=-1)

                all_y.append(y.cpu().numpy())
                all_hat.append(hat.cpu().numpy())

        if not all_y: return {"acc": 0.0, "f1": 0.0}

        y_true = np.concatenate(all_y)
        y_pred = np.concatenate(all_hat)

        return {
            "acc": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, average="macro"))
        }

    def update_model(
        self,
        epochs=20,
        batch_size=8,
        lr=1e-4,
        weight_decay=1e-2,
        label_smoothing=0.1,
        val_ratio=0.2,
        patience=5,
        min_delta=0.001,
    ):
        samples = self.load_samples_with_groups()
        if len(samples) < 8:
            print(f"Not enough data to train (Found {len(samples)} samples). Needs ~8.")
            return False, None

#   1. Chronological Split (Train on Past, Test on Today)  
        groups = [s["group"] for s in samples]
        unique_groups = sorted(list(set(groups))) # Sort dates: ['2025-10-01', '2025-10-02', ...]

        if len(unique_groups) > 1:
            # The latest day (Today) becomes Validation
            # All previous days become Training
            latest_day = unique_groups[-1]
            print(f" splitting: Training on Past Data | Validating on Latest Session ({latest_day})")
            
            train_idx = [i for i, g in enumerate(groups) if g != latest_day]
            val_idx = [i for i, g in enumerate(groups) if g == latest_day]
        else:
            # Fallback: If this is very first day, just split randomly
            print("Only one session found. Using standard random split.")
            gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=42)
            train_idx, val_idx = next(gss.split(samples, groups=groups))
        
        train_s = [samples[i] for i in train_idx]
        val_s = [samples[i] for i in val_idx]

        print(f"Training on {len(train_s)} samples, Validating on {len(val_s)} samples.")
        
        # 2. Datasets (Augment Train only) 
        train_ds = PersonalEmotionDataset(train_s, self.model.emotion_names, self.processor, augment=True)
        val_ds = PersonalEmotionDataset(val_s, self.model.emotion_names, self.processor, augment=False)

        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False) if val_s else None

        #  3. Optimizer setup 
        # Note: We used self.adapter in the class above, so we optimize that
        params = list(self.model.adapter.parameters()) + \
                 list(self.model.emotion_head.parameters()) + \
                 list(self.model.intensity_head.parameters()) + \
                 list(self.model.valence_arousal_head.parameters())

        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        device = next(self.model.parameters()).device

        # 4. Loop  
        best_state = None
        best_acc = -1.0
        bad_epochs = 0

        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0.0

            for batch in train_dl:
                x = batch["pixel_values"].to(device)
                y = batch["emotion_labels"].to(device)

                optimizer.zero_grad()
                logits = self.model(x)["emotion_logits"]

                loss = F.cross_entropy(logits, y, label_smoothing=label_smoothing)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / max(1, len(train_dl))

            # Validation
            if val_dl:
                metrics = self._eval_epoch(self.model, val_dl, device)
                val_acc = metrics['acc']
                print(f"Epoch {epoch:02d}: Loss={avg_loss:.4f} | Val Acc={val_acc:.3f} | F1={metrics['f1']:.3f}")

                if val_acc > best_acc + min_delta:
                    best_acc = val_acc
                    best_state = copy.deepcopy(self.model.state_dict())
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                    if bad_epochs >= patience:
                        print("Early stopping triggered.")
                        break
            else:
                print(f"Epoch {epoch:02d}: Loss={avg_loss:.4f} (No val set)")

        #  5. Finish 
        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"Restored best model (Val Acc: {best_acc:.3f})")
            if val_dl:
                self.fit_temperature(val_dl)

        self.model.eval()
        return True, best_acc
