from ai.tools import delete_downloaded_sentence_transformers_models

def DelStorage():
    print("\nConfirm deletion\n")
    print("[1] Delete all cached models")
    print("[0] Cancel")
    print()
    choice = input("Select an action: ")

    if choice == "1":
        delete_downloaded_sentence_transformers_models()
        print("\nPress Enter to continue...")
        input()
    
    return
