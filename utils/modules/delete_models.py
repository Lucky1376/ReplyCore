from ai.download import delete_model

def Delete(get_download_models):
    print()
    print("[0] - Exit\n")
    print()
    print("Downloaded models:")
    models = get_download_models()
    for i, model in enumerate(models):
        print(f"[{i+1}] {model['name']}")
    print()
    model_name = input("Select a model to delete: ")

    if model_name in ["0", ""]:
        return
    
    model_name = models[int(model_name)-1]["name"]
    delete_model(model_name)
    print(f"Model {model_name} deleted")
    print("\nPress Enter to continue...")
    input()
