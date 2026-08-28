from __future__ import annotations
import numpy as np

class PersonalizedEmotionViT:
    def __init__(self, num_emotions: int = 7, freeze_base: bool = True):
        self.emotion_names = ['happy','sad','angry','fear','surprise','disgust','neutral']

    def predict(self, image, temporal_context=None):
        return {
            'primary_emotion': 'neutral',
            'confidence': 0.5,
            'all_probabilities': {k: (1.0 if k=='neutral' else 0.0) for k in self.emotion_names},
            'intensity': 0.3,
            'authenticity': 0.5,
            'valence': 0.0,
            'arousal': 0.0,
            'features': np.zeros(8, dtype=np.float32),
        }

class EmotionCore:
    def __init__(self):
        self.model = PersonalizedEmotionViT()
    def predict(self, frame_rgb: np.ndarray):
        return self.model.predict(frame_rgb)

class JetsonEmotionPipeline:
    """Stub pipeline used by cara_emotion_node; replace with your real camera pipeline later."""
    def __init__(self, model, camera_id=0, use_tensorrt=False):
        self.core = EmotionCore()
        self._last = None
    def get_current_emotion(self, smoothed=True):
        # Return a neutral placeholder
        if self._last is None:
            self._last = self.core.predict(np.zeros((224,224,3), dtype=np.uint8))
            self._last.update({'stability': 1.0, 'timestamp': 'now'})
        return self._last
