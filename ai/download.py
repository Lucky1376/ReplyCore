from huggingface_hub import snapshot_download
from pathlib import Path
from datetime import datetime
import json
import shutil

def delete_model(model_name: str, target_dir: str = "hub"):
    """
    Deletes the model (folder with files).
    
    :param model_name: Name of the model
    :param target_dir: Directory with models
    """
    
    model_dir = Path(target_dir) / model_name
    if model_dir.exists():
        shutil.rmtree(model_dir)
    else:
        raise FileNotFoundError(f"Model {model_name} not found")

def download_model(model_name: str, custom_save_name="", target_dir: str = "hub", 
                  ignore_patterns: list = None):
    """
    Downloads a model from Hugging Face Hub with selective files
    
    :param model_name: Name of the model (with or without the prefix)
    :param custom_save_name: Custom name for the saved model
    :param target_dir: Directory to save the model
    :param ignore_patterns: List of file patterns to ignore
    :return: Path to the saved model
    """
    # if not model_name.startswith('sentence-transformers/'):
    #     model_name = f'sentence-transformers/{model_name}'

    if custom_save_name and ('/' in custom_save_name or '\\' in custom_save_name):
        raise ValueError("Custom save name cannot contain path separators (/, \\)")
    
    if custom_save_name:
        model_dir = Path(target_dir) / custom_save_name
    else:
        model_dir = Path(target_dir) / model_name.split('/')[-1]

    model_dir.mkdir(parents=True, exist_ok=True)

    # Default ignore patterns
    default_ignore = [
        "*.h5",         # TensorFlow
        "*.msgpack",    # Flax/JAX
        "*.onnx",       # ONNX
        "*.ot",         # Other
        "*.tflite",     # TensorFlow Lite
        "*.mlmodel",    # Core ML
        "*.bin",        # PyTorch
    ]
    
    final_ignore = ignore_patterns if ignore_patterns is not None else default_ignore

    try:
        snapshot_download(
            repo_id=model_name,
            local_dir=model_dir,
            local_dir_use_symlinks=False,
            ignore_patterns=final_ignore,
            allow_patterns=["*.json", "*.txt", "*.safetensors", "tokenizer.model"]  # Only the ones we need
        )
        
        # Deleting possible empty directories
        for subdir in ["tf_model.h5", "flax_model.msgpack", "onnx"]:
            dir_path = model_dir / subdir
            if dir_path.exists():
                shutil.rmtree(dir_path)
        
        # Saving metadata
        with open(model_dir / "meta.json", "w") as f:
            json.dump({
                "source": model_name,
                "downloaded_at": datetime.now().isoformat(),
                "downloaded_files": [f.name for f in model_dir.glob("*") if f.is_file()]
            }, f, indent=2)
            
        return str(model_dir)
        
    except Exception as e:
        raise RuntimeError(f"Error downloading model: {str(e)}")