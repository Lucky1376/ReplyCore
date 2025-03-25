from ai.pipeline_tester import PipelineTester
import json

def Test():
    print("\n\n[0] - Exit\n")
    # print("Доступные модели:\n")
    # for model in edu.get_trained_models():
    #     print(f" - {model['name']} ({model['questions']} вопросов, создана {model['created_at']})")
    model_name = input("\nEnter the pipeline name: ")

    if model_name in ["0", ""]:
        return

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