# ReplyCore

### _Fast QA pipeline creation using your data with [sentence-transformers](https://pypi.org/project/sentence-transformers/): model training and production-ready integration_

## ❓Why is this needed?

_I personally use it to automate responses to frequent repetitive questions in tech support, but there are many possible use cases._

## ⚙️How does it work?

Your questions and answers are converted into numerical vectors using a neural network model.  
`"How do I reset my password?"` → `[0.24, -0.12, 0.76, ...]`

The model does not look for exact word matches but calculates **semantic similarity** based on the angle between vectors.

The system understands **rephrased questions** thanks to:

- Considering word order
- Recognizing synonyms (`"reset password" ≈ "recover access"`)
- Multi-task model training

## 🤖📊Available Models in the Interactive Program

> You can select additional models for `utils/const.py`  
> from [this list](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html)

| Model ID | Name                                    | Dimensions | Speed | Languages | Best For                  | Size  | Benchmark (MTEB) |
| -------- | --------------------------------------- | ---------- | ----- | --------- | ------------------------- | ----- | ---------------- |
| 1        | `paraphrase-multilingual-mpnet-base-v2` | 768        | 🐢    | 50+       | Highest accuracy tasks    | 1.2GB | 65.3             |
| 2        | `paraphrase-multilingual-MiniLM-L12-v2` | 384        | 🚗    | 50+       | Balanced speed/quality    | 470MB | 63.7             |
| 3        | `distiluse-base-multilingual-cased-v2`  | 512        | 🚄    | 50+       | Low-resource environments | 480MB | 61.2             |
| 4        | `LaBSE`                                 | 768        | 🐢    | 109       | Multilingual applications | 1.8GB | 58.2             |
| 5        | `multilingual-e5-large`                 | 1024       | 🚗    | 100+      | Large-scale production    | 2.1GB | 72.1             |

## 💡✨Why is the Interactive Program Beneficial?

1. _Easily train a pipeline without writing custom code_
2. _Assemble a ready-to-use pipeline with your model and a built-in module for operation_
3. _Download any models directly in the program for offline training_
4. _Test your pipelines immediately after training—no need to constantly move folders into your project. Validate on the spot and check statistics_

## 🧠🔄Training Strategies

### `last` (_Default_)

**How it works:**

- Takes the answer with the same index as the question (`answers[i]`).
- If there are fewer answers than questions, it uses the last answer (`answers[-1]`).

**Example:**

    questions = ["Q1", "Q2", "Q3"]
    answers = ["A1", "A2"]

    Result:
    Q1 → A1, Q2 → A2, Q3 → A2 (last answer)

**When to use:**

- For "one question → one answer" pairs.
- When answers are ordered correctly for the questions.

##

### `cycle` (_Cyclic_)

**How it works:**

- Reuses answers cyclically: `answers[i % len(answers)]`.

**Example:**

    questions = ["Q1", "Q2", "Q3", "Q4"]
    answers = ["A1", "A2"]

    Result:
    Q1 → A1, Q2 → A2, Q3 → A1, Q4 → A2

**When to use:**

- When there are more questions than answers.
- When answers are general-purpose (e.g., common hints).

  ##

### `random` (_Random_)

**How it works:**

- Selects a random answer from the list using `random.choice(answers)`.

**Example:**

    questions = ["Q1", "Q2", "Q3"]
    answers = ["A1", "A2", "A3"]

    Possible result:
    Q1 → A3, Q2 → A1, Q3 → A3

**When to use:**

- To add variety to responses.

##

### `most-similar`

**How it works:**

1. For each question, its **embedding** (vector representation) is calculated.
2. The **embeddings** of all answers are **pre-cached** (for speed).
3. The answer **most semantically similar** to the question is selected (via cosine similarity).

**Example**

    questions = ["How to reset password?", "Payment failed", "Contact support"]
    answers = ["Click 'Forgot password'", "Check balance", "Email us at help@site.com"]

    # Embeddings:
    q_embeddings = model.encode(questions)  # Vector for each question
    a_embeddings = model.encode(answers)   # Vector for each answer

    # For the question "Payment failed":
    question_idx = 1
    question_embedding = q_embeddings[1]

    # Compare with answer embeddings:
    similarities = cosine_similarity([question_embedding], a_embeddings)[0]
    best_answer_idx = similarities.argmax()  # Index of the most similar answer

    Result:
    "Payment failed" → "Check balance" (as their embeddings are the closest)

**When to use:**

- When **answers are not tied** to specific questions (e.g., a general knowledge base).
- For complex questions, where **direct matching** (`last`, `cycle`) produces poor results.
- In **RAG systems**, where finding semantic matches is important.

## ⬇️🚀Installation and Launch

**Requirements: Python 3.9+**

**Install dependencies:**

    pip install -r requirements.txt

**Add your training data to the `data/` directory**

> An example is provided in the `data/example.json` file.

**Launch the interactive program:**

    python main.py

## 🔗🧩Integration with the Project

_The assembled pipelines with models are saved in the `build/your_pipeline` directory. This folder contains the `pipeline.py` module for working with the pipeline._

**Working with the assembled pipeline**

    from your_pipeline.pipeline import Pipeline

    pipe  = Pipeline()
    result  =  pipe.query("Shall we have a cup of coffee?")

    print(result)

**Result:**

    {
        "answer": "I suggest having a freshly squeezed juice",
        "score": 0.8474252223968506,
        "is_match": True,
        "strategy": "cycle"
    }

**Where:**

- `answer` - _The answer_
- `score` - _Confidence level of the answer_
- `is_match` - _Has the pre-defined similarity threshold been
  exceeded?_
- `strategy` - _Training strategy of the pipeline_

## 🌟In conclusion

_This program **will not create a real artificial intelligence**. It will only train a pipeline on existing data. It is not self-learning, it doesn't think, and it can't come up with answers. It simply helps to automate responses._
