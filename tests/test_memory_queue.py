"""Tests for the age-tracking memory queue."""

import pytest
import torch

from gmacore.models.memory_queue import MemoryQueue


def test_keys_are_normalized_on_enqueue():
    queue = MemoryQueue(size=16, dim=8)
    queue.enqueue(torch.randn(4, 8) * 10.0)
    norms = queue.keys[queue.valid].norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_age_increments_and_resets():
    queue = MemoryQueue(size=8, dim=4)

    queue.enqueue(torch.randn(2, 4))
    assert queue.age[:2].max().item() == 0.0

    queue.enqueue(torch.randn(2, 4))
    # The first pair aged by one step; the new pair starts at zero.
    assert queue.age[:2].min().item() == 1.0
    assert queue.age[2:4].max().item() == 0.0


def test_fifo_wraparound():
    queue = MemoryQueue(size=4, dim=4)
    for _ in range(3):
        queue.enqueue(torch.randn(3, 4))
    assert queue.num_valid == 4
    assert int(queue.pointer.item()) == 9 % 4


def test_only_written_slots_are_valid():
    queue = MemoryQueue(size=10, dim=4)
    assert queue.num_valid == 0
    queue.enqueue(torch.randn(3, 4))
    assert queue.num_valid == 3
    assert queue.negatives().shape == (3, 4)


def test_decay_weights_decrease_with_age():
    queue = MemoryQueue(size=8, dim=4, decay=0.9)
    queue.enqueue(torch.randn(2, 4))
    queue.enqueue(torch.randn(2, 4))
    queue.enqueue(torch.randn(2, 4))

    weights = queue.decay_weights(enabled=True)
    assert weights[0].item() < weights[4].item()
    assert weights[4].item() == pytest.approx(1.0)


def test_disabled_decay_returns_unit_weights():
    queue = MemoryQueue(size=8, dim=4, decay=0.5)
    queue.enqueue(torch.randn(4, 4))
    queue.enqueue(torch.randn(4, 4))
    weights = queue.decay_weights(enabled=False)
    assert torch.allclose(weights, torch.ones_like(weights))


def test_oversized_batch_is_rejected():
    queue = MemoryQueue(size=4, dim=4)
    with pytest.raises(ValueError):
        queue.enqueue(torch.randn(5, 4))


def test_buffers_are_registered_for_device_movement():
    queue = MemoryQueue(size=4, dim=4)
    names = {name for name, _ in queue.named_buffers()}
    assert {"keys", "age", "valid", "pointer"} <= names
