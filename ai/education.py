import json
import random
import shutil
import os
from datetime import datetime
import time
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from typing import Literal, Optional, Dict, List
from pathlib import Path
from tqdm import tqdm

class Education:
    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        """
        Initialization of model training
        :param model_name: model name (with or without the prefix)
        :param hub_dir: folder with saved models (optional)
        """
        self.model_name = model_name
        self.hub_dir = 'hub'
        self.data_dir = 'data'
        self.pipeline_dir = 'build'
        self._answer_embeddings_cache = {}
        self._current_answers_hash = None
        self._ensure_dirs_exist()
        self.model = self._init_model()
    
    def _init_model(self):
        """Initializes the model, first trying the local hub, then creating directly"""
        # Trying to load from hub if the folder is specified
        if self.hub_dir:
            local_path = os.path.join(self.hub_dir, self.model_name)
            if os.path.exists(local_path):
                try:
                    _a = SentenceTransformer(local_path)
                    print("Model loaded from hub")
                    return _a
                except Exception as e:
                    print(f"Failed to load the model from hub: {str(e)}")
        
        # If not found in the hub, load directly
        full_model_name = self.model_name
        _a = SentenceTransformer(full_model_name)
        print("Model loaded directly")
        return _a

    def _load_model_from_hub(self, model_name):
        """Loads the model from the local hub folder"""
        model_path = os.path.join(self.hub_dir, model_name)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model {model_name} not found in folder {self.hub_dir}. "
                f"Available models: {os.listdir(self.hub_dir)}"
            )
            
        return SentenceTransformer(model_path)

    def _ensure_dirs_exist(self):
        """Creates necessary directories if they do not exist"""
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.pipeline_dir).mkdir(parents=True, exist_ok=True)
        if self.hub_dir:
            Path(self.hub_dir).mkdir(parents=True, exist_ok=True)

    def _copy_pipeline_files(self, model_dir: str):
        """Copies the necessary files for the pipeline to work"""
        dest_path = Path(model_dir)
        
        # Copying the self-contained pipeline.py
        current_dir = Path(__file__).parent
        shutil.copy(current_dir / 'pipeline.py', dest_path)
        
        # Creating the __init__.py file
        with open(dest_path / '__init__.py', 'w') as f:
            f.write('# Auto-generated pipeline package\n')
        
        # Creating requirements.txt
        possible_req_paths = [
            current_dir.parent / 'requirements.txt',  # In the root of the project
            Path.cwd() / 'requirements.txt'           # In the working directory
        ]
        
        for req_path in possible_req_paths:
            if req_path.exists():
                shutil.copy(req_path, dest_path)
                break
    
    def _get_embeddings(self, answers: List[str], force_update: bool = False) -> torch.Tensor:
        """Smart caching of answer embeddings"""
        answers_tuple = tuple(answers)
        current_hash = hash(answers_tuple)
        
        if force_update or current_hash != self._current_answers_hash:
            self._clear_embeddings_cache()
            self._current_answers_hash = current_hash
            
        if current_hash not in self._answer_embeddings_cache:
            with torch.no_grad():
                self._answer_embeddings_cache[current_hash] = {
                    'embeddings': self.model.encode(answers, convert_to_tensor=True),
                    'timestamp': time.time()
                }
        
        return self._answer_embeddings_cache[current_hash]['embeddings']

    def _clear_embeddings_cache(self, max_items: int = 3, max_age_hours: int = 24):
        """Clearing old caches"""
        now = time.time()
        to_delete = []
        
        if len(self._answer_embeddings_cache) > max_items:
            oldest = sorted(self._answer_embeddings_cache.items(), 
                          key=lambda x: x[1]['timestamp'])[0][0]
            to_delete.append(oldest)
        
        for h, data in self._answer_embeddings_cache.items():
            if (now - data['timestamp']) > max_age_hours * 3600:
                to_delete.append(h)
        
        for h in set(to_delete):
            del self._answer_embeddings_cache[h]
            if h == self._current_answers_hash:
                self._current_answers_hash = None

    def train_on_file(self, data_file: str, model_name: str, 
                 answer_strategy: Literal['last', 'cycle', 'random', 'most_similar'] = 'last',
                 show_progress: bool = True, chunk_size: int = 100):
        """
        Train the model on the specified data file
        :param data_file: name of the data file (e.g. 'faq.json')
        :param model_name: name for saving the model
        :param answer_strategy: answer selection strategy 
            ('last' - last, 'cycle' - cyclic, 'random' - random, 'most_similar' - most similar)
        :param show_progress: whether to show progress bars
        :param chunk_size: batch size for question encoding
        :return: dictionary with training results
        """
        # Data validation
        if not data_file.endswith('.json'):
            data_file += '.json'
        
        data_path = os.path.join(self.data_dir, data_file)
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file {data_path} not found")

        # Load data
        with open(data_path, 'r', encoding='utf-8') as f:
            try:
                faq = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON format error: {str(e)}")

        # Validate data structure
        if not isinstance(faq, list):
            raise ValueError("Data should be an array of objects")

        # Prepare data with progress bars
        all_questions = []
        all_answers = []
        
        # Main progress bar for FAQ items
        faq_iter = tqdm(faq, desc="Processing FAQ items", disable=not show_progress)
        for item in faq_iter:
            if not all(k in item for k in ['questions', 'answers']):
                raise ValueError("Each item must contain 'questions' and 'answers'")
            
            answers = item['answers']
            questions = item['questions']
            
            if not answers:
                raise ValueError("Answer list cannot be empty")
            
            # Nested progress bar for questions
            questions_iter = tqdm(questions, desc="   Processing questions", 
                                leave=False, disable=not show_progress)
            for i, question in enumerate(questions_iter):
                all_questions.append(question)
                
                # Select answer by strategy
                if answer_strategy == 'last':
                    answer = answers[i] if i < len(answers) else answers[-1]
                elif answer_strategy == 'cycle':
                    answer = answers[i % len(answers)]
                elif answer_strategy == 'random':
                    answer = random.choice(answers)
                elif answer_strategy == 'most_similar':
                    answer_embeddings = self._get_embeddings(answers)
                    question_embedding = self.model.encode(question, convert_to_tensor=True)
                    similarities = util.cos_sim(question_embedding, answer_embeddings)[0]
                    answer = answers[similarities.argmax().item()]
                else:
                    raise ValueError(f"Invalid strategy: {answer_strategy}")
                
                all_answers.append(answer)

        if not all_questions:
            raise ValueError("No questions found for training")

        # Encode questions with chunked progress bar
        question_embeddings = []
        chunks = [all_questions[i:i + chunk_size] for i in range(0, len(all_questions), chunk_size)]
        
        encoding_iter = tqdm(chunks, desc="Encoding questions", disable=not show_progress)
        for chunk in encoding_iter:
            question_embeddings.extend(self.model.encode(chunk))
        
        question_embeddings = np.array(question_embeddings)

        # Create model folder
        model_dir = os.path.join(self.pipeline_dir, model_name)
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        
        # Save results
        print(f"Saving pipeline...")
        np.save(os.path.join(model_dir, 'question_embeddings.npy'), question_embeddings)
        with open(os.path.join(model_dir, 'answers.json'), 'w', encoding='utf-8') as f:
            json.dump(all_answers, f, ensure_ascii=False, indent=2)
        
        # Save the model
        model_files_path = os.path.join(model_dir, 'model_files')
        self.model.save(model_files_path)
        
        # Get model name safely
        try:
            model_name_attr = getattr(self.model, 'model_name', None)
            model_name_str = model_name_attr if model_name_attr else str(self.model[0].auto_model.config._name_or_path)
            base_model_name = os.path.basename(model_name_str)
        except Exception:
            base_model_name = "unknown_model"

        # Model metadata
        meta = {
            'source_data': data_file,
            'questions_count': len(all_questions),
            'answers_count': len(all_answers),
            'model_info': {
                'name': base_model_name,
                'source': 'local_hub',
                'embedding_dim': question_embeddings.shape[1],
                'max_seq_length': self.model.max_seq_length,
                'model_files_path': 'model_files'
            },
            'training_params': {
                'answer_strategy': answer_strategy,
                'created_at': datetime.now().isoformat(),
                'chunk_size': chunk_size
            }
        }
        
        with open(os.path.join(model_dir, 'meta.json'), 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        
        self._copy_pipeline_files(model_dir)

        return {
            'status': 'success',
            'model_name': model_name,
            'model_dir': model_dir,
            'model_files_path': model_files_path,
            'questions_processed': len(all_questions),
            'answers_processed': len(all_answers),
            'embedding_shape': question_embeddings.shape
        }

    def update_answers(self, new_answers: List[str]):
        """Принудительное обновление кэша эмбеддингов"""
        self._clear_embeddings_cache(max_items=0)
        self._get_embeddings(new_answers, force_update=True)
    
    def get_trained_models(self):
        """Returns a list of trained models"""
        models = []
        for model_dir in Path(self.pipeline_dir).iterdir():
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
                            print(f"Error reading metadata {meta_path}: {str(e)}")
                            continue
        return models
