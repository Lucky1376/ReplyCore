models = {
    1: {
        "name": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "desc": "High quality multilingual embeddings",
        "details": {
            "languages": "50+",
            "embedding_size": 768,
            "speed": "medium",
            "best_for": "Semantic search, clustering",
            "pros": "Excellent quality for multilingual tasks",
            "cons": "Larger memory footprint",
            "release_year": 2020
        }
    },
    2: {
        "name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "desc": "Optimal balance of speed and quality",
        "details": {
            "languages": "50+",
            "embedding_size": 384,
            "speed": "fast",
            "best_for": "Real-time applications, production use",
            "pros": "4x faster than mpnet with good accuracy",
            "cons": "Lower dimensionality than mpnet",
            "release_year": 2021
        }
    },
    3: {
        "name": "sentence-transformers/distiluse-base-multilingual-cased-v2",
        "desc": "Lightweight multilingual model",
        "details": {
            "languages": "50+",
            "embedding_size": 512,
            "speed": "very fast",
            "best_for": "Mobile/edge devices, low-resource environments",
            "pros": "Small size, decent performance",
            "cons": "Lower accuracy than full-size models",
            "release_year": 2020
        }
    },
    4: {
        "name": "sentence-transformers/LaBSE",
        "desc": "Google's universal language encoder",
        "details": {
            "languages": 109,
            "embedding_size": 768,
            "speed": "medium",
            "best_for": "Cross-lingual tasks, language detection",
            "pros": "Widest language coverage",
            "cons": "Outdated architecture",
            "release_year": 2019
        }
    },
    6: {
        "name": "intfloat/multilingual-e5-large",
        "desc": "Microsoft's efficient multilingual encoder",
        "details": {
            "languages": "100+",
            "embedding_size": 1024,
            "speed": "medium-fast",
            "best_for": "Large-scale production systems",
            "pros": "Excellent speed/accuracy balance",
            "cons": "Slightly less precise than BGE-M3",
            "release_year": 2023,
            "benchmarks": {
                "MTEB": 72.1,
                "RAG": 89.7
            }
        }
    }
}

strategies = {
    "1": "cycle",
    "2": "random",
    "3": "last",
    "4": "most_similar"
}