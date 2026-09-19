from pathlib import Path

from src.nlp.preprocessing.catalog_loader import CatalogLoader
from src.nlp.preprocessing.matcher import RegexMatcher
from src.nlp.preprocessing.normalizer import TextNormalizer

RULES = Path("configs/normalization.yaml")


def test_normalizer_is_idempotent() -> None:
    normalizer = TextNormalizer(RegexMatcher.from_catalog(CatalogLoader().load(RULES).catalog))
    once = normalizer.normalize("Xe nhà H. chạy cx ổn :))")
    twice = normalizer.normalize(once.normalized_text)
    assert twice.normalized_text == once.normalized_text


def test_contextual_brand_alias_is_expanded_for_car_context() -> None:
    normalizer = TextNormalizer(RegexMatcher.from_catalog(CatalogLoader().load(RULES).catalog))
    result = normalizer.normalize("Xe nhà H. chạy ổn")
    assert result.normalized_text == "Xe nhà Hyundai chạy ổn"
    assert "hyundai_h_dot" in result.applied_rules


def test_h_dot_is_not_expanded_without_automotive_context() -> None:
    normalizer = TextNormalizer(RegexMatcher.from_catalog(CatalogLoader().load(RULES).catalog))
    result = normalizer.normalize("Anh H. đang họp")
    assert result.normalized_text == "Anh H. đang họp"


def test_negation_and_emoji_are_preserved() -> None:
    normalizer = TextNormalizer(RegexMatcher.from_catalog(CatalogLoader().load(RULES).catalog))
    result = normalizer.normalize("xe ko ổn 😡")
    assert result.normalized_text == "xe không ổn 😡"
