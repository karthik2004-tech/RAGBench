from src.ragbench.benchmark import ExperimentStorage


def create_result(name, recall):
    return {
        "experiment": {
            "name": name,
            "chunk_size": 400,
            "chunk_overlap": 50,
            "top_k": 3,
            "embedding_model": "all-MiniLM-L6-v2",
            "llm_model": "qwen2.5:1.5b",
        },
        "results": [
            {
                "id": "q001",
                "retrieval": {
                    "recall_at_k": recall,
                    "precision_at_k": 1.0,
                    "reciprocal_rank": 1.0,
                },
                "generation": {
                    "answer_relevancy": 0.8,
                    "faithfulness": {
                        "score": 0.5
                    },
                },
            }
        ],
    }


def test_save_and_load_experiment(tmp_path):

    storage = ExperimentStorage(
        output_dir=str(tmp_path)
    )

    result = create_result(
        "test_experiment",
        1.0,
    )

    storage.save(result)

    loaded = storage.load(
        "test_experiment"
    )

    assert loaded == result


def test_load_all_experiments(tmp_path):

    storage = ExperimentStorage(
        output_dir=str(tmp_path)
    )

    result1 = create_result(
        "experiment_1",
        0.5,
    )

    result2 = create_result(
        "experiment_2",
        1.0,
    )

    storage.save(result1)
    storage.save(result2)

    experiments = storage.load_all()

    assert len(experiments) == 2

    names = {
        experiment["experiment"]["name"]
        for experiment in experiments
    }

    assert names == {
        "experiment_1",
        "experiment_2",
    }


def test_load_all_empty_directory(tmp_path):

    storage = ExperimentStorage(
        output_dir=str(tmp_path)
    )

    experiments = storage.load_all()

    assert experiments == []