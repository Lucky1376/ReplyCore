models = {
    1: {
        "name": "paraphrase-multilingual-mpnet-base-v2",
        "desc": "high quality, 50+ languages"
    },
    2: {
        "name": "paraphrase-multilingual-MiniLM-L12-v2",
        "desc": "optimal, high speed"
    },
    3: {
        "name": "distiluse-base-multilingual-cased-v2",
        "desc": "lite version"
    },
    4: {
        "name": "LaBSE",
        "desc": "Language-agnostic BERT Sentence Embedding (google)"
    }
}

strategies = {
    "1": "cycle",
    "2": "random",
    "3": "last"
}