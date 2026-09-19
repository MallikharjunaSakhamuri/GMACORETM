"""Tests for configuration loading and override handling."""

import pytest

from gmacore.utils.config import apply_overrides, merge


def test_nested_override_is_applied():
    config = {"pretrain": {"epochs": 150, "batch_size": 64}}
    updated = apply_overrides(config, ["pretrain.epochs=10"])
    assert updated["pretrain"]["epochs"] == 10
    assert updated["pretrain"]["batch_size"] == 64


def test_override_does_not_mutate_the_input():
    config = {"pretrain": {"epochs": 150}}
    apply_overrides(config, ["pretrain.epochs=10"])
    assert config["pretrain"]["epochs"] == 150


@pytest.mark.parametrize(
    "raw,expected",
    [("a=1", 1), ("a=1.5", 1.5), ("a=true", True), ("a=false", False),
     ("a=none", None), ("a=cuda", "cuda")],
)
def test_value_coercion(raw, expected):
    assert apply_overrides({}, [raw])["a"] == expected


def test_missing_key_is_created():
    assert apply_overrides({}, ["framework.decay=0.95"])["framework"]["decay"] == 0.95


def test_malformed_override_is_rejected():
    with pytest.raises(ValueError):
        apply_overrides({}, ["no_equals_sign"])


def test_merge_is_recursive():
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    result = merge(base, {"a": {"y": 20}})
    assert result == {"a": {"x": 1, "y": 20}, "b": 3}
