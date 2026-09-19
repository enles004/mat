"""Unit tests for explicit Vietnamese segmentation and its pipeline variant."""

from scripts import train_baseline
from src.nlp.modeling.baseline import BaselineTrainer
from src.nlp.preprocessing.vietnamese_tokenizer import VietnameseSegmenter


def test_segmenter_joins_compound_words() -> None:
    segmenter = VietnameseSegmenter()
    assert segmenter.segment("Xe ô tô chạy ổn") == "Xe ô_tô chạy ổn"


def test_segmenter_batch_preserves_order_and_length() -> None:
    segmenter = VietnameseSegmenter()
    texts = ["máy chạy êm", "", "thiết kế cân đối và hiện đại"]
    segmented = segmenter.segment_batch(texts)
    assert len(segmented) == len(texts)
    assert segmented[0] == "máy chạy êm"
    assert segmented[1] == ""
    assert segmented == VietnameseSegmenter().segment_batch(texts)


def test_default_pipeline_has_no_segment_step() -> None:
    pipeline = BaselineTrainer.build_pipeline(seed=42)
    assert list(pipeline.named_steps) == ["features", "classifier"]


def test_segmented_pipeline_inserts_segment_step_first() -> None:
    pipeline = BaselineTrainer.build_pipeline(seed=42, segmenter=VietnameseSegmenter())
    assert list(pipeline.named_steps) == ["segment", "features", "classifier"]
    pipeline.fit(
        ["Xe ô tô chạy ổn", "Xe ô tô rất ổn", "máy rung và phản hồi chậm", "máy rung quá"],
        ["positive", "positive", "negative", "negative"],
    )
    assert pipeline.predict(["Xe ô tô chạy ổn"]).tolist() == ["positive"]


def test_train_cli_defaults_to_no_tokenizer() -> None:
    args = train_baseline.build_parser().parse_args(
        ["--data", "d.csv", "--splits", "s.json", "--config", "c.yaml", "--output", "o"]
    )
    assert args.tokenizer == "none"
