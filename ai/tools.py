import os, json
import shutil
from pathlib import Path

def get_built_pipelines(target_dir: str = "build"):
    """
    Returns a list of built pipelines.

    :param target_dir: Directory with built pipelines.
    :return: List of dictionaries with pipeline information.
    """
    pipelines = []
    path = Path(target_dir)

    for item in path.iterdir():
        if item.is_dir():
            with open(item / "meta.json", "r", encoding="utf-8") as f:
                meta = json.load(f)
                data = {
                    "name": item.name,
                    "questions": meta["questions_count"],
                    "created_at": meta["training_params"]["created_at"]
                }
                pipelines.append(data)
    
    return pipelines

def delete_built_pipeline(pipeline_name: str, target_dir: str = "build"):
    """
    Deletes a built pipeline.

    :param pipeline_name: Name of the pipeline.
    :param target_dir: Directory with built pipelines.
    """
    pipeline_dir = Path(target_dir) / pipeline_name
    if pipeline_dir.exists():
        shutil.rmtree(pipeline_dir)
    else:
        raise FileNotFoundError(f"Pipeline {pipeline_name} not found")

def get_download_models(target_dir: str = "hub"):
    """
    Returns a list of downloaded models.
    
    :param target_dir: Directory with downloaded models.
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
