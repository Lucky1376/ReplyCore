from ai.tools import get_built_pipelines, delete_built_pipeline


def DeleteBuiltPipeline():
    pipelines = get_built_pipelines()
    print("\n[0] - Exit\n")
    for i, pipeline in enumerate(pipelines):
        print(f"[{i + 1}] {pipeline['name']}")
    print()
    answer = input("Select a pipeline to delete (You can use a space): ")
    if answer in ["0", ""]:
        return
    
    for pipe in answer.split():
        try:
            delete_built_pipeline(pipelines[int(pipe) - 1]["name"])
            print(f"Pipeline {pipelines[int(pipe) - 1]['name']} deleted")
        except FileNotFoundError as e:
            print(e)
    print("\nPress Enter to continue...")
    input()
    return