import pytest

from src.ragbench.benchmark import ExperimentConfig


def test_default_experiment_config():
    config = ExperimentConfig(
        name="baseline"
    )

    assert config.name == "baseline"
    assert config.chunk_size == 400
    assert config.chunk_overlap == 50
    assert config.top_k == 3
    assert config.embedding_model == "all-MiniLM-L6-v2"
    assert config.llm_model == "qwen2.5:1.5b"


def test_custom_experiment_config():
    config = ExperimentConfig(
        name="small_chunks",
        chunk_size=200,
        chunk_overlap=20,
        top_k=5,
        embedding_model="all-MiniLM-L6-v2",
        llm_model="qwen2.5:1.5b",
    )

    assert config.name == "small_chunks"
    assert config.chunk_size == 200
    assert config.chunk_overlap == 20
    assert config.top_k == 5
    assert config.embedding_model == "all-MiniLM-L6-v2"
    assert config.llm_model == "qwen2.5:1.5b"


def test_empty_experiment_name():
    with pytest.raises(ValueError):
        ExperimentConfig(name="")


def test_whitespace_experiment_name():
    with pytest.raises(ValueError):
        ExperimentConfig(name="   ")


def test_invalid_chunk_size():
    with pytest.raises(ValueError):
        ExperimentConfig(
            name="invalid_chunk",
            chunk_size=0,
        )


def test_negative_chunk_size():
    with pytest.raises(ValueError):
        ExperimentConfig(
            name="invalid_chunk",
            chunk_size=-100,
        )


def test_negative_chunk_overlap():
    with pytest.raises(ValueError):
        ExperimentConfig(
            name="invalid_overlap",
            chunk_overlap=-1,
        )


def test_overlap_equal_to_chunk_size():
    with pytest.raises(ValueError):
        ExperimentConfig(
            name="invalid_overlap",
            chunk_size=100,
            chunk_overlap=100,
        )


def test_overlap_greater_than_chunk_size():
    with pytest.raises(ValueError):
        ExperimentConfig(
            name="invalid_overlap",
            chunk_size=100,
            chunk_overlap=150,
        )


def test_invalid_top_k():
    with pytest.raises(ValueError):
        ExperimentConfig(
            name="invalid_top_k",
            top_k=0,
        )


def test_negative_top_k():
    with pytest.raises(ValueError):
        ExperimentConfig(
            name="invalid_top_k",
            top_k=-1,
        )


def test_empty_embedding_model():
    with pytest.raises(ValueError):
        ExperimentConfig(
            name="invalid_embedding",
            embedding_model="",
        )


def test_empty_llm_model():
    with pytest.raises(ValueError):
        ExperimentConfig(
            name="invalid_llm",
            llm_model="",
        )