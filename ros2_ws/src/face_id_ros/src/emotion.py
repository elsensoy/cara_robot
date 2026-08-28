"""
Personal Emotion Recognition System for AI Teddy Bear
Optimized for Jetson Orin with interactive learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import ViTModel, ViTImageProcessor
import cv2
import numpy as np
from collections import deque
from datetime import datetime
import json
import os
from pathlib import Path
import threading
import queue

# ============================================================================
# PERSONALIZED EMOTION VIT MODEL
# ============================================================================

class PersonalizedEmotionViT(nn.Module):
    """ViT model that learns YOUR specific emotional expressions"""
    
    def __init__(self, num_emotions=7, freeze_base=True):
        super().__init__()
        
        # Use lightweight ViT for Jetson Orin
        self.base_vit = ViTModel.from_pretrained(
            'WinKawaks/vit-tiny-patch16-224',
            add_pooling_layer=False
        )
        
        hidden_size = 192  # ViT-Tiny hidden size
        
        # Freeze base model initially (only train on your data)
        if freeze_base:
            for param in self.base_vit.parameters():
                param.requires_grad = False
        
        # Personal adapter layers (these learn YOUR patterns)
        self.personal_adapter = nn.ModuleDict({
            'adapter': nn.Sequential(
                nn.Linear(hidden_size, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(0.1),
            ),
            
            # Multiple heads for rich emotion understanding
            'emotion_head': nn.Linear(128, num_emotions),
            'intensity_head': nn.Linear(128, 1),
            'authenticity_head': nn.Linear(128, 1),
            'valence_arousal_head': nn.Linear(128, 2),
        })
        
        # Temporal LSTM for tracking emotion changes
        self.temporal_lstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        
        # Learnable emotion memory (your personal emotion signatures)
        self.emotion_memory = nn.Parameter(
            torch.randn(num_emotions, 128) * 0.01,
            requires_grad=True
        )
        
        self.emotion_names = [
            'happy', 'sad', 'angry', 'fear', 
            'surprise', 'disgust', 'neutral'
        ]
        
    def forward(self, pixel_values, temporal_features=None, return_features=False):
        # Extract base features
        outputs = self.base_vit(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0]
        
        # Personal adaptation
        adapted = self.personal_adapter['adapter'](cls_token)
        
        # Add temporal context if available
        if temporal_features is not None:
            temporal_out, _ = self.temporal_lstm(temporal_features)
            adapted = adapted + temporal_out[:, -1]
        
        # Multi-head outputs
        emotion_logits = self.personal_adapter['emotion_head'](adapted)
        intensity = torch.sigmoid(self.personal_adapter['intensity_head'](adapted))
        authenticity = torch.sigmoid(self.personal_adapter['authenticity_head'](adapted))
        valence_arousal = torch.tanh(self.personal_adapter['valence_arousal_head'](adapted))
        
        result = {
            'emotion_logits': emotion_logits,
            'intensity': intensity,
            'authenticity': authenticity,
            'valence_arousal': valence_arousal,
        }
        
        if return_features:
            result['features'] = adapted
            
        return result
    
    def predict(self, image, temporal_context=None):
        """Single image prediction with rich output"""
        self.eval()
        with torch.no_grad():
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image).float()
            
            if image.dim() == 3:
                image = image.unsqueeze(0)
            
            outputs = self.forward(image, temporal_context, return_features=True)
            
            probs = F.softmax(outputs['emotion_logits'], dim=-1)
            primary_emotion_idx = probs.argmax(dim=-1).item()
            
            return {
                'primary_emotion': self.emotion_names[primary_emotion_idx],
                'confidence': probs[0, primary_emotion_idx].item(),
                'all_probabilities': {
                    name: probs[0, i].item() 
                    for i, name in enumerate(self.emotion_names)
                },
                'intensity': outputs['intensity'][0, 0].item(),
                'authenticity': outputs['authenticity'][0, 0].item(),
                'valence': outputs['valence_arousal'][0, 0].item(),
                'arousal': outputs['valence_arousal'][0, 1].item(),
                'features': outputs['features'][0].cpu().numpy()
            }


# ============================================================================
# MOTION AND GESTURE ANALYSIS
# ============================================================================

class MotionEmotionAnalyzer:
    """Analyze body language and motion patterns"""
    
    def __init__(self):
        self.motion_history = deque(maxlen=30)  # 1 second at 30fps
        
    def analyze_motion(self, face_bbox_history):
        """Extract motion features from face bounding box movements"""
        if len(face_bbox_history) < 10:
            return self._default_features()
        
        # Convert to numpy for easier computation
        boxes = np.array(face_bbox_history)
        
        # Calculate center points
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        
        # Motion features
        features = {
            'head_movement_speed': self._calculate_speed(centers),
            'head_stability': self._calculate_stability(centers),
            'vertical_movement': self._detect_nodding(centers),
            'horizontal_movement': self._detect_shaking(centers),
            'movement_energy': self._calculate_energy(centers),
            'face_size_change': self._detect_lean(boxes),
        }
        
        return features
    
    def _calculate_speed(self, centers):
        """Average movement speed"""
        if len(centers) < 2:
            return 0.0
        diffs = np.diff(centers, axis=0)
        speeds = np.sqrt((diffs ** 2).sum(axis=1))
        return float(np.mean(speeds))
    
    def _calculate_stability(self, centers):
        """How stable is head position (0=unstable, 1=stable)"""
        if len(centers) < 2:
            return 1.0
        variance = np.var(centers, axis=0).sum()
        return float(1.0 / (1.0 + variance))
    
    def _detect_nodding(self, centers):
        """Detect vertical head movement (nodding)"""
        if len(centers) < 5:
            return 0.0
        vertical = centers[:, 1]
        # Look for oscillation
        diff = np.diff(vertical)
        sign_changes = np.sum(np.diff(np.sign(diff)) != 0)
        return float(sign_changes / len(diff))
    
    def _detect_shaking(self, centers):
        """Detect horizontal head movement (shaking)"""
        if len(centers) < 5:
            return 0.0
        horizontal = centers[:, 0]
        diff = np.diff(horizontal)
        sign_changes = np.sum(np.diff(np.sign(diff)) != 0)
        return float(sign_changes / len(diff))
    
    def _calculate_energy(self, centers):
        """Overall movement energy level"""
        if len(centers) < 2:
            return 0.0
        diffs = np.diff(centers, axis=0)
        energy = np.sqrt((diffs ** 2).sum(axis=1)).sum()
        return float(min(energy / 100.0, 1.0))  # Normalize
    
    def _detect_lean(self, boxes):
        """Detect if leaning forward/back (face getting larger/smaller)"""
        if len(boxes) < 10:
            return 0.0
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        trend = np.polyfit(range(len(areas)), areas, 1)[0]
        return float(np.clip(trend / 100.0, -1.0, 1.0))
    
    def _default_features(self):
        return {
            'head_movement_speed': 0.0,
            'head_stability': 1.0,
            'vertical_movement': 0.0,
            'horizontal_movement': 0.0,
            'movement_energy': 0.0,
            'face_size_change': 0.0,
        }
    
    def fuse_with_facial_emotion(self, facial_emotion, motion_features):
        """Combine facial expression with motion cues"""
        adjustments = {
            'confidence_boost': 0.0,
            'intensity_multiplier': 1.0,
            'emotion_override': None
        }
        
        # High energy + neutral face = likely happy
        if (facial_emotion['primary_emotion'] == 'neutral' and 
            motion_features['movement_energy'] > 0.6):
            adjustments['emotion_override'] = 'happy'
            adjustments['confidence_boost'] = 0.2
        
        # Low energy + sad face = definitely sad
        if (facial_emotion['primary_emotion'] == 'sad' and 
            motion_features['movement_energy'] < 0.2):
            adjustments['confidence_boost'] = 0.3
            adjustments['intensity_multiplier'] = 1.3
        
        # Nodding + positive emotion = strong agreement
        if (motion_features['vertical_movement'] > 0.3 and 
            facial_emotion['primary_emotion'] in ['happy', 'surprise']):
            adjustments['intensity_multiplier'] = 1.4
        
        # Apply adjustments
        result = facial_emotion.copy()
        result['confidence'] = min(
            result['confidence'] + adjustments['confidence_boost'], 
            1.0
        )
        result['intensity'] *= adjustments['intensity_multiplier']
        
        if adjustments['emotion_override']:
            result['primary_emotion'] = adjustments['emotion_override']
        
        result['motion_features'] = motion_features
        
        return result


# ============================================================================
# INTERACTIVE LEARNING SYSTEM
# ============================================================================

class InteractiveLearningSystem:
    """Collect labeled samples and continuously improve the model"""
    
    def __init__(self, model, storage_path="./teddy_personal_data"):
        self.model = model
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.feedback_buffer = []
        self.training_samples = []
        
        # Load existing samples
        self._load_existing_samples()
        
    def collect_labeled_sample(self, frame, emotion_label, context=None):
        """You tell the teddy how you're feeling"""
        timestamp = datetime.now()
        
        sample = {
            'frame': frame,  # Will be saved as image
            'true_emotion': emotion_label,
            'timestamp': timestamp.isoformat(),
            'context': context or {},
            'model_prediction': self.model.predict(frame) if frame is not None else None,
        }
        
        self.feedback_buffer.append(sample)
        self.training_samples.append(sample)
        
        # Save to disk
        self._save_sample(sample)
        
        print(f" Collected sample: {emotion_label} (Total: {len(self.training_samples)})")
        
        return sample
    
    def passive_learning_mode(self, frame, conversation_context):
        """Learn from natural conversation without explicit labels"""
        
        # Get model prediction
        prediction = self.model.predict(frame)
        
        # Infer emotion from conversation
        inferred_emotion = self._infer_from_conversation(conversation_context)
        
        # If high confidence match, use as training sample
        if (prediction['confidence'] > 0.8 and 
            inferred_emotion and 
            prediction['primary_emotion'] == inferred_emotion):
            
            self.collect_labeled_sample(
                frame,
                prediction['primary_emotion'],
                context={
                    'source': 'passive_learning',
                    'conversation': conversation_context,
                    'confidence': prediction['confidence']
                }
            )
    
    def _infer_from_conversation(self, conversation):
        """Simple keyword-based emotion inference"""
        if not conversation:
            return None
        
        text = conversation.lower()
        
        # Keyword matching (you can make this more sophisticated)
        emotion_keywords = {
            'happy': ['happy', 'great', 'awesome', 'love', 'excited', 'haha', 'lol'],
            'sad': ['sad', 'upset', 'down', 'rough', 'bad day', 'disappointed'],
            'angry': ['angry', 'mad', 'frustrated', 'annoyed', 'irritated'],
            'fear': ['scared', 'worried', 'anxious', 'nervous', 'afraid'],
            'neutral': ['okay', 'fine', 'alright', 'normal'],
        }
        
        for emotion, keywords in emotion_keywords.items():
            if any(keyword in text for keyword in keywords):
                return emotion
        
        return None
    
    def update_model(self, num_epochs=10, batch_size=8, learning_rate=1e-4):
        """Fine-tune model on collected samples"""
        
        if len(self.training_samples) < batch_size:
            print(f"Not enough samples yet ({len(self.training_samples)}/{batch_size})")
            return
        
        print(f"\n🎓 Training on {len(self.training_samples)} personal samples...")
        
        # Create dataset
        dataset = PersonalEmotionDataset(self.training_samples, self.model.emotion_names)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Optimizer (only train personal adapters)
        optimizer = torch.optim.AdamW([
            {'params': self.model.personal_adapter.parameters()},
            {'params': self.model.temporal_lstm.parameters()},
            {'params': [self.model.emotion_memory]},
        ], lr=learning_rate)
        
        # Training loop
        self.model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in dataloader:
                optimizer.zero_grad()
                
                outputs = self.model(batch['pixel_values'])
                
                # Multi-task loss
                emotion_loss = F.cross_entropy(
                    outputs['emotion_logits'], 
                    batch['emotion_labels']
                )
                
                loss = emotion_loss
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        print("✓ Training complete!")
        self._save_checkpoint()
    
    def _save_sample(self, sample):
        """Save sample to disk"""
        timestamp = datetime.fromisoformat(sample['timestamp'])
        filename = timestamp.strftime("%Y%m%d_%H%M%S")
        
        # Save image
        img_path = self.storage_path / f"{filename}.jpg"
        if sample['frame'] is not None:
            cv2.imwrite(str(img_path), sample['frame'])
        
        # Save metadata
        meta_path = self.storage_path / f"{filename}.json"
        meta = {
            'true_emotion': sample['true_emotion'],
            'timestamp': sample['timestamp'],
            'context': sample['context'],
            'model_prediction': sample['model_prediction'],
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
    
    def _load_existing_samples(self):
        """Load previously saved samples"""
        json_files = list(self.storage_path.glob("*.json"))
        print(f"Loading {len(json_files)} existing samples...")
        
        for json_path in json_files:
            with open(json_path, 'r') as f:
                meta = json.load(f)
            
            # Load corresponding image
            img_path = json_path.with_suffix('.jpg')
            if img_path.exists():
                frame = cv2.imread(str(img_path))
                
                sample = {
                    'frame': frame,
                    'true_emotion': meta['true_emotion'],
                    'timestamp': meta['timestamp'],
                    'context': meta.get('context', {}),
                    'model_prediction': meta.get('model_prediction'),
                }
                self.training_samples.append(sample)
    
    def _save_checkpoint(self):
        """Save model checkpoint"""
        checkpoint_path = self.storage_path / "model_checkpoint.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'num_samples': len(self.training_samples),
            'timestamp': datetime.now().isoformat(),
        }, checkpoint_path)
        print(f"✓ Checkpoint saved: {checkpoint_path}")


# ============================================================================
# DATASET FOR PERSONAL SAMPLES
# ============================================================================

class PersonalEmotionDataset(Dataset):
    def __init__(self, samples, emotion_names):
        self.samples = samples
        self.emotion_names = emotion_names
        self.processor = ViTImageProcessor.from_pretrained(
            'WinKawaks/vit-tiny-patch16-224'
        )
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Process image
        image = sample['frame']
        if image.shape[2] == 3:  # BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0)
        
        # Emotion label
        emotion_idx = self.emotion_names.index(sample['true_emotion'])
        
        return {
            'pixel_values': pixel_values,
            'emotion_labels': torch.tensor(emotion_idx, dtype=torch.long)
        }


# ============================================================================
# REAL-TIME EMOTION RECOGNITION PIPELINE
# ============================================================================

class JetsonEmotionPipeline:
    """Real-time emotion recognition optimized for Jetson Orin"""
    
    def __init__(self, model, camera_id=0, use_tensorrt=False):
        self.model = model
        self.motion_analyzer = MotionEmotionAnalyzer()
        
        # Camera setup
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Buffers
        self.frame_queue = queue.Queue(maxsize=2)
        self.emotion_buffer = deque(maxlen=90)  # 3 seconds
        self.face_bbox_history = deque(maxlen=30)
        
        # Threading
        self.running = False
        self.current_emotion = None
        
        # Face detection (using OpenCV's pre-trained model)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Image processor
        self.processor = ViTImageProcessor.from_pretrained(
            'WinKawaks/vit-tiny-patch16-224'
        )
    
    def start(self):
        """Start the emotion recognition pipeline"""
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        
        self.capture_thread.start()
        self.inference_thread.start()
        
        print("✓ Emotion recognition pipeline started")
    
    def stop(self):
        """Stop the pipeline"""
        self.running = False
        self.capture_thread.join()
        self.inference_thread.join()
        self.cap.release()
        print("✓ Pipeline stopped")
    
    def _capture_loop(self):
        """Capture frames from camera"""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
    
    def _inference_loop(self):
        """Run inference on frames"""
        frame_skip = 2  # Process every 2nd frame (15 FPS)
        count = 0
        
        while self.running:
            if not self.frame_queue.empty() and count % frame_skip == 0:
                frame = self.frame_queue.get()
                
                # Detect face
                face_frame, bbox = self._detect_and_crop_face(frame)
                
                if face_frame is not None:
                    # Track face bbox for motion analysis
                    self.face_bbox_history.append(bbox)
                    
                    # Predict emotion
                    facial_emotion = self.model.predict(
                        self._preprocess_face(face_frame)
                    )
                    
                    # Analyze motion
                    motion_features = self.motion_analyzer.analyze_motion(
                        list(self.face_bbox_history)
                    )
                    
                    # Fuse facial + motion
                    final_emotion = self.motion_analyzer.fuse_with_facial_emotion(
                        facial_emotion, motion_features
                    )
                    
                    # Add timestamp
                    final_emotion['timestamp'] = datetime.now().isoformat()
                    
                    # Update buffers
                    self.emotion_buffer.append(final_emotion)
                    self.current_emotion = final_emotion
            
            count += 1
    
    def _detect_and_crop_face(self, frame):
        """Detect and crop face from frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
        )
        
        if len(faces) > 0:
            # Take largest face
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            
            # Add margin
            margin = int(0.2 * w)
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(frame.shape[1], x + w + margin)
            y2 = min(frame.shape[0], y + h + margin)
            
            face = frame[y1:y2, x1:x2]
            return face, (x1, y1, x2, y2)
        
        return None, None
    
    def _preprocess_face(self, face):
        """Preprocess face for ViT"""
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=face_rgb, return_tensors="pt")
        return inputs['pixel_values']
    
    def get_current_emotion(self, smoothed=True):
        """Get current emotional state"""
        if not self.current_emotion:
            return None
        
        if smoothed and len(self.emotion_buffer) > 5:
            return self._smooth_emotions(list(self.emotion_buffer)[-15:])
        
        return self.current_emotion
    
    def _smooth_emotions(self, recent_emotions):
        """Average emotions over recent frames"""
        if not recent_emotions:
            return None
        
        # Average probabilities
        emotion_counts = {}
        for e in recent_emotions:
            emotion = e['primary_emotion']
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # Most common emotion
        primary = max(emotion_counts.items(), key=lambda x: x[1])[0]
        
        # Average other metrics
        avg_confidence = np.mean([e['confidence'] for e in recent_emotions])
        avg_intensity = np.mean([e['intensity'] for e in recent_emotions])
        avg_valence = np.mean([e['valence'] for e in recent_emotions])
        avg_arousal = np.mean([e['arousal'] for e in recent_emotions])
        
        return {
            'primary_emotion': primary,
            'confidence': float(avg_confidence),
            'intensity': float(avg_intensity),
            'valence': float(avg_valence),
            'arousal': float(avg_arousal),
            'stability': self._calculate_stability(recent_emotions),
            'timestamp': recent_emotions[-1]['timestamp']
        }
    
    def _calculate_stability(self, emotions):
        """Calculate emotional stability"""
        if len(emotions) < 2:
            return 1.0
        
        changes = sum(
            1 for i in range(1, len(emotions))
            if emotions[i]['primary_emotion'] != emotions[i-1]['primary_emotion']
        )
        return 1.0 - (changes / len(emotions))
    
    def get_emotion_history(self, seconds=3):
        """Get emotion history for temporal analysis"""
        frames = int(seconds * 15)  # 15 FPS effective
        return list(self.emotion_buffer)[-frames:]


# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================

def main_training_loop():
    """Interactive training session"""
    
    print("=" * 60)
    print("Personal Emotion Recognition Training")
    print("For Your AI Teddy Bear 🧸")
    print("=" * 60)
    
    # Initialize model
    print("\n Loading model...")
    model = PersonalizedEmotionViT(num_emotions=7, freeze_base=True)
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"✓ Model loaded on {device}")
    
    # Initialize learning system
    learning_system = InteractiveLearningSystem(model)
    
    # Initialize camera pipeline
    print("\n📹 Starting camera...")
    pipeline = JetsonEmotionPipeline(model, camera_id=0)
    pipeline.start()
    
    print("\n" + "=" * 60)
    print("INTERACTIVE LEARNING MODE")
    print("=" * 60)
    print("Commands:")
    print("  [emotion] - Label current frame (e.g., 'happy', 'sad', 'neutral')")
    print("  'show' - Show current emotion prediction")
    print("  'train' - Train model on collected samples")
    print("  'stats' - Show collection statistics")
    print("  'quit' - Exit")
    print("=" * 60)
    
    emotions = model.emotion_names
    
    try:
        while True:
            command = input("\n> ").strip().lower()
            
            if command == 'quit':
                break
            
            elif command == 'show':
                emotion = pipeline.get_current_emotion()
                if emotion:
                    print(f"\n Current Emotion Analysis:")
                    print(f"   Primary: {emotion['primary_emotion']}")
                    print(f"   Confidence: {emotion['confidence']:.2%}")
                    print(f"   Intensity: {emotion['intensity']:.2f}")
                    print(f"   Valence: {emotion['valence']:.2f} (negative ← → positive)")
                    print(f"   Arousal: {emotion['arousal']:.2f} (calm ← → excited)")
                    if 'stability' in emotion:
                        print(f"   Stability: {emotion['stability']:.2%}")
                else:
                    print(" No face detected")
            
            elif command == 'train':
                learning_system.update_model(num_epochs=10, batch_size=4)
            
            elif command == 'stats':
                print(f"\n Collection Statistics:")
                print(f"   Total samples: {len(learning_system.training_samples)}")
                
                # Count per emotion
                emotion_counts = {}
                for sample in learning_system.training_samples:
                    emotion = sample['true_emotion']
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                
                print(f"   Breakdown:")
                for emotion, count in sorted(emotion_counts.items()):
                    print(f"      {emotion}: {count}")
            
            elif command in emotions:
                # Label current frame
                if not pipeline.frame_queue.empty():
                    frame = pipeline.frame_queue.get()
                    face, _ = pipeline._detect_and_crop_face(frame)
                    
                    if face is not None:
                        learning_system.collect_labeled_sample(
                            face, 
                            command,
                            context={'time_of_day': datetime.now().strftime('%H:%M')}
                        )
                    else:
                        print("  No face detected in frame")
                else:
                    print("  No frame available")
            
            else:
                print(f" Unknown command: {command}")
                print(f"   Valid emotions: {', '.join(emotions)}")
    
    finally:
        pipeline.stop()
        print("\n Goodbye!")


if __name__ == "__main__":
    main_training_loop()
