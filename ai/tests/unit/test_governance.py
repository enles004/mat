from pathlib import Path


def test_model_governance_requires_pinned_reviewed_artifacts() -> None:
    governance = (Path(__file__).parents[2] / "CLAUDE.md").read_text(encoding="utf-8")

    required_controls = (
        "Pin the exact model revision at run start.",
        "Record the model checksum and license for every model.",
        "Do not publish an artifact until its terms have been reviewed.",
    )

    missing_controls = [control for control in required_controls if control not in governance]

    assert not missing_controls, f"Missing model governance controls: {missing_controls}"
