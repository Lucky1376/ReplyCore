from ai.pipeline_tester import PipelineTester
from ai.tools import get_built_pipelines
import json

def Test():
    print("\n\n[0] - Exit\n")
    print("Available pipelines:\n")
    pipelines = get_built_pipelines()
    for i, pipeline in enumerate(pipelines):
        # print(pipeline)
        print(f"[{i + 1}] {pipeline['name']} ({pipeline['questions']} questions, created on {pipeline['created_at']})")
    if not pipelines:
        print("None")
    # for model in edu.get_trained_models():
    #     print(f" - {model['name']} ({model['questions']} вопросов, создана {model['created_at']})")
    model_name = input("\nEnter the pipeline name: ")

    if model_name in ["0", ""]:
        return

    try:
        model_name = pipelines[int(model_name) - 1]["name"]
    except:
        pass

    tester = PipelineTester(model_name)

    print("\n[stats] - Show statistics")
    print("[0] - Exit\n")
    while True:
        question = input("\nEnter your question: ")
        if question in ["0", ""]:
            break
        elif question == "stats":
            print(json.dumps(tester.get_stats(), indent=4, ensure_ascii=False))
            continue

        result = tester.query(question)
        print(f"Answer: {result['answer']} (similarity: {result['score']:.2f})")


    return