from ai.download import download_model
from utils.const import models

def Download():
    print("\n[0] - Exit\n")
    for i, model in models.items():
        name = model["name"].split("/")[-1]
        desc = model["desc"]
        print(f"[{i}] {name} ({desc})")

    choice = int(input("\nSelect a model: "))

    local_name = input("Enter a name to save the model (press Enter to skip): ")

    if choice not in models:
        return

    download_model(models[choice]["name"], custom_save_name=local_name)

    print(f"\nModel {models[choice]} successfully downloaded")
    print("\nPress Enter to continue...")
    input()
    return
