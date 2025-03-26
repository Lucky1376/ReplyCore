import numpy as np
import json
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

class PipelineTester:
    def __init__(self, model_name, models_path="build"):
        """
        :param model_name: Name of the trained model (e.g. 'faq_model')
        :param models_path: Path to the folder with trained models (default is 'build')
        """
        self.models_path = Path(models_path)
        self.model_path = self.models_path / model_name
        self.model = None
        self.embeddings = None
        self.answers = None
        self.meta = None
        self.stats = {
            'total_queries': 0,
            'matches': 0,
            'threshold': 0.7,
            'queries': []
        }
        self._load_model()

    def _load_model(self):
        """Load the model and data from the folder of the trained model"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_path}")

        # Load the model from saved files
        model_files_path = self.model_path / 'model_files'
        if not model_files_path.exists():
            raise FileNotFoundError(f"Model files not found in {model_files_path}")
        
        self.model = SentenceTransformer(str(model_files_path))
        
        # Load the other components
        self.embeddings = np.load(self.model_path / 'question_embeddings.npy')
        
        with open(self.model_path / 'answers.json', 'r', encoding='utf-8') as f:
            self.answers = json.load(f)
            
        with open(self.model_path / 'meta.json', 'r', encoding='utf-8') as f:
            self.meta = json.load(f)

    def get_trained_models(self):
        """Returns a list of trained models (similar to Education.get_trained_models)"""
        models = []
        for model_dir in self.models_path.iterdir():
            if model_dir.is_dir():
                meta_path = model_dir / 'meta.json'
                if meta_path.exists():
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        try:
                            meta = json.load(f)
                            models.append({
                                'name': model_dir.name,
                                'source': meta['source_data'],
                                'questions': meta['questions_count'],
                                'created_at': meta['training_params']['created_at'],
                                'path': str(model_dir),
                                'strategy': meta['training_params']['answer_strategy'],
                                'model_info': meta.get('model_info', {})
                            })
                        except (json.JSONDecodeError, KeyError) as e:
                            print(f"Error reading metadata from {meta_path}: {str(e)}")
                            continue
        return models

    def query(self, question, threshold=None):
        """
        Query the model
        :param question: The question text
        :param threshold: Similarity threshold (None for the default value)
        :return: {
            'answer': str|None, 
            'score': float,
            'is_match': bool
        }
        """
        threshold = threshold or self.stats['threshold']
        self.stats['total_queries'] += 1

        # Encode the question
        question_embedding = self.model.encode([question])
        
        # Find the closest match
        sim_scores = cosine_similarity(question_embedding, self.embeddings)[0]
        best_idx = np.argmax(sim_scores)
        best_score = float(sim_scores[best_idx])
        is_match = best_score > threshold
        
        # Record the statistics
        result = {
            'question': question,
            'answer': self.answers[best_idx] if is_match else None,
            'score': best_score,
            'is_match': is_match,
            'timestamp': datetime.now().isoformat()
        }
        
        if is_match:
            self.stats['matches'] += 1
            
        self.stats['queries'].append(result)
        return result

    def get_stats(self, reset=False):
        """
        Get statistics
        :param reset: Reset statistics after fetching
        :return: {
            'total_queries': int,
            'matches': int,
            'match_rate': float,
            'threshold': float,
            'last_query': dict|None
        }
        """
        stats = {
            'total_queries': self.stats['total_queries'],
            'matches': self.stats['matches'],
            'match_rate': self.stats['matches'] / self.stats['total_queries'] if self.stats['total_queries'] > 0 else 0,
            'threshold': self.stats['threshold'],
            'last_query': self.stats['queries'][-1] if self.stats['queries'] else None
        }
        
        if reset:
            self.reset_stats()
            
        return stats

    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            'total_queries': 0,
            'matches': 0,
            'threshold': self.stats['threshold'],
            'queries': []
        }

    def set_threshold(self, threshold):
        """Set the similarity threshold"""
        self.stats['threshold'] = float(threshold)
