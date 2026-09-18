import pytest

from mcsirius.live_backend import (
    LiveModel,
    SOURCE_COMMAND_TOPICS,
    _parse_payload,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1", True),
        ("0", False),
        ("14.25", 14.25),
        ("14,25", 14.25),
        ("hello", "hello"),
    ],
)
def test_live_payload_parser(
    text,
    expected,
):
    assert _parse_payload(text) == expected


def test_live_model_updates_timestamp_and_value():
    model = LiveModel()

    model.update(
        "test",
        1.0,
    )

    first = model.get("test")

    assert first.value == 1.0
    assert first.quality == "good"

    model.update(
        "test",
        2.0,
    )

    second = model.get("test")

    assert second.value == 2.0
    assert second.timestamp >= first.timestamp


def test_exact_source_command_topics_match_flavia():
    assert SOURCE_COMMAND_TOPICS == {
        "cs/sputter/set_u_v":
            "cs/sputter/cmd/set_u_v",
        "cs/extraction/set_u_v":
            "cs/extraction/cmd/set_u_v",
        "cs/einzellens/set_u_v":
            "cs/einzellens/cmd/set_u_v",
    }