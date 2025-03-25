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

def download_model(model_name: str, custom_save_name="", target_dir: str = "hub"):
    """
    Downloads a model from the Hugging Face Hub.
    
    :param model_name: Name of the model (with or without the prefix)
    :param target_dir: Directory to save the model
    :return: Path to the saved model
    """
    # Add the prefix if it's missing
    if not model_name.startswith('sentence-transformers/'):
        model_name = f'sentence-transformers/{model_name}'
    
    if custom_save_name:
        model_dir = Path(target_dir) / custom_save_name
    else:
        model_dir = Path(target_dir) / model_name.split('/')[-1]

    model_dir.mkdir(parents=True, exist_ok=True)

    try:
        snapshot_download(
            repo_id=model_name,
            local_dir=model_dir,
            local_dir_use_symlinks=False
        )
        
        # Save metadata
        with open(model_dir / "meta.json", "w") as f:
            json.dump({
                "source": model_name,
                "downloaded_at": datetime.now().isoformat()
            }, f, indent=2)
            
        return str(model_dir)
        
    except Exception as e:
        raise RuntimeError(f"Error downloading model: {str(e)}")
