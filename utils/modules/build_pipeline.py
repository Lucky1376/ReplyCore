from ai.education import Education
from utils.const import strategies, models

def Build(get_downloaded_models):
    print("\n[0] - Exit\n")
    data_file = input("Enter the name of the data file: ")
    pipeline_name = input("Enter a name for the pipeline: ")
    print()
    print("[1] - Cyclic strategy")
    print("[2] - Random strategy")
    print("[3] - Last strategy")
    print("[4] - Most similar strategy")
    print()
    answer_strategy = input("Select a strategy: ")

    # Return if any values are empty
    if any(x in ["0", ""] for x in [data_file, pipeline_name, answer_strategy]):
        return

    print()
    i = 0
    print("Models from sentence-transformers:")
    for i, model in models.items():
        name = model["name"].split("/")[-1]
        desc = model["desc"]
        print(f"[{i}] {name} ({desc})")
        i += 1
    print()
    print("Models from hub:")
    hub_models: list = get_downloaded_models()
    for model in hub_models:
        print(f'[{i}] {model["name"]}')
        i += 1
    if len(hub_models) == 0:
        print("None")
    print()
    model_name = int(input("Select a model: "))
    
    # Training

    print("Training the pipeline...")
    model_name = models.get(model_name, model_name)
    if type(model_name) == int:
        # print(model_name, model_name-len(models) - 1)
        # print(hub_models)
        model_name = hub_models[model_name - len(models) - 1]
    edu = Education(model_name=model_name["name"])
    result = edu.train_on_file(data_file, pipeline_name, answer_strategy=strategies.get(answer_strategy, "cycle"))
    print("Pipeline saved at", result['model_dir'])
    print("\nPress Enter to continue...")
    input()
    return
