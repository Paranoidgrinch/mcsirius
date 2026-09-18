import pytest

from mcsirius.machine import (
    DEFAULT_LIMITS,
    DEFAULT_OPERATING_POINT,
    MachineLimits,
    OperatingPoint,
)


def test_default_operating_point_is_valid():
    DEFAULT_LIMITS.validate(DEFAULT_OPERATING_POINT)


@pytest.mark.parametrize(
    ("sputter_kv", "extraction_kv", "einzel_kv"),
    [
        (4.0, 14.0, 14.0),
        (9.0, 14.0, 16.0),
        (6.5, 20.0, 18.0),
        (6.5, 20.0, 22.0),
        (9.0, 25.0, 23.0),
        (9.0, 25.0, 25.0),
    ],
)
def test_valid_operating_points(sputter_kv, extraction_kv, einzel_kv):
    DEFAULT_LIMITS.validate(
        OperatingPoint(
            sputter_kv=sputter_kv,
            extraction_kv=extraction_kv,
            einzel_kv=einzel_kv,
        )
    )


@pytest.mark.parametrize(
    "point",
    [
        OperatingPoint(3.9, 14.0, 14.0),
        OperatingPoint(9.1, 14.0, 14.0),
        OperatingPoint(4.0, 13.9, 14.0),
        OperatingPoint(4.0, 25.1, 25.0),
        OperatingPoint(4.0, 14.0, 13.9),
        OperatingPoint(4.0, 25.0, 25.1),
        OperatingPoint(4.0, 18.0, 15.9),
        OperatingPoint(4.0, 18.0, 20.1),
    ],
)
def test_invalid_operating_points_are_rejected(point):
    with pytest.raises(ValueError):
        DEFAULT_LIMITS.validate(point)


def test_einzel_window_at_low_extraction():
    limits = MachineLimits()

    assert limits.einzel_window(14.0) == (14.0, 16.0)


def test_einzel_window_in_middle():
    limits = MachineLimits()

    assert limits.einzel_window(20.0) == (18.0, 22.0)


def test_einzel_window_at_high_extraction():
    limits = MachineLimits()

    assert limits.einzel_window(25.0) == (23.0, 25.0)
