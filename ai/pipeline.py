# pipeline.py
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class Pipeline:
    def __init__(self):
        """
        Standalone pipeline class that works with files in its directory.
        """
        self.base_path = Path(__file__).parent
        self._load_components()

    def _load_components(self):
        """Loads all components from the current directory."""
        # Checking for required files
        required_files = [
            'model_files',
            'question_embeddings.npy',
            'answers.json',
            'meta.json'
        ]
        
        for file in required_files:
            if not (self.base_path / file).exists():
                raise FileNotFoundError(f"Required file missing: {file}")

        # Load the model
        self.model = SentenceTransformer(str(self.base_path / 'model_files'))
        
        # Load embeddings
        self.embeddings = np.load(self.base_path / 'question_embeddings.npy')
        
        # Load answers
        with open(self.base_path / 'answers.json', 'r', encoding='utf-8') as f:
            self.answers = json.load(f)
        
        # Load metadata
        with open(self.base_path / 'meta.json', 'r', encoding='utf-8') as f:
            self.meta = json.load(f)

    def query(self, question: str, threshold: float = 0.7) -> dict:
        """
        Main method to process a query.
        """
        # Encode the question
        question_embedding = self.model.encode([question])
        
        # Find the closest match
        sim_scores = cosine_similarity(question_embedding, self.embeddings)[0]
        best_idx = np.argmax(sim_scores)
        best_score = float(sim_scores[best_idx])
        
        return {
            'answer': self.answers[best_idx] if best_score > threshold else None,
            'score': best_score,
            'is_match': best_score > threshold,
            'strategy': self.meta['training_params']['answer_strategy']
        }
