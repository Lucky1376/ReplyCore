from ai.download import delete_model

def Delete(get_download_models):
    print()
    print("[0] - Exit\n")
    print()
    print("Downloaded models:")
    models: list = get_download_models()
    for i, model in enumerate(models):
        print(f"[{i+1}] {model['name']}")
    print()
    model_name = input("Select a model to delete (You can use a space): ")

    if model_name in ["0", ""]:
        return
    
    for model_name in model_name.split():
        model_name = models[int(model_name)-1]["name"]
        delete_model(model_name)
        print(f"Model {model_name} deleted")
    print("\nPress Enter to continue...")
    input()
