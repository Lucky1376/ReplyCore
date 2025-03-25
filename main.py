from utils.modules import build_pipeline, exit, test_model, download_models, delete_models, delete_storage_models
from utils.utils import clear
from ai.tools import get_download_models

__version__ = "0.1.0"

def main():
    while True:
        clear()

        print()
        print("[1] - Build pipeline")
        print("[2] - Test pipeline") 
        print("[3] - Download model")
        print("[4] - Remove downloaded model")
        print("[5] - Clear model cache")
        print("[0] - Exit")

        print()
        choice = input("Select action: ")

        if choice in ["0", ""]:
            exit.exit()
        if choice == "1":
            build_pipeline.Build(get_download_models)
        elif choice == "2":
            test_model.Test()
        elif choice == "3":
            download_models.Download()
        elif choice == "4":
            delete_models.Delete(get_download_models)
        elif choice == "5":
            delete_storage_models.DelStorage()


if __name__ == '__main__':
    main()