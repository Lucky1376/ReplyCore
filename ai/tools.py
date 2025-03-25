import os
import shutil
from pathlib import Path

def get_download_models(target_dir: str = "hub"):
    """
    Returns a list of downloaded models.
    
    :return: List of dictionaries with model information.
    """
    models = []
    path = Path(target_dir)

    for item in path.iterdir():
        if item.is_dir():
            data = {
                "name": item.name,
                "source": target_dir + "/" + item.name,
            }
            models.append(data)
    
    return models

def delete_downloaded_sentence_transformers_models():
    """
    Deletes all downloaded models from known cache directories.
    """
    deleted = False
    cache_paths = [
        Path.home() / ".cache" / "huggingface" / "hub",
        Path.home() / ".cache" / "torch" / "sentence_transformers",
        Path.home() / ".cache" / "huggingface" / "transformers"
    ]

    for cache_path in cache_paths:
        if cache_path.exists():
            print(f"Found model directory: {cache_path}")
            try:
                shutil.rmtree(cache_path)
                print(f"✅ Successfully deleted: {cache_path}")
                deleted = True
            except Exception as e:
                print(f"⚠️ Error deleting {cache_path}: {str(e)}")

    if not deleted:
        print("❌ No cache directories found")
