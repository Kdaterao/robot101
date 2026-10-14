"""VRAM-aware caption batch sizing with bounded growth and OOM backoff."""
from __future__ import annotations

import math


class AutoBatch:
    def __init__(self, initial, maximum=64, reserve_fraction=0.10):
        self.maximum = maximum
        self.size = min(initial, maximum)
        self.reserve_fraction = reserve_fraction
        self.ceiling = maximum
        self.peak_allocated = 0
        self.events = []

    def begin(self, torch, device):
        torch.cuda.synchronize(device)
        self.peak_allocated = max(self.peak_allocated, torch.cuda.max_memory_allocated(device))
        torch.cuda.reset_peak_memory_stats(device)
        free, total = torch.cuda.mem_get_info(device)
        self.baseline = torch.cuda.memory_allocated(device)
        # Add the allocator's own reserved memory back to free device memory.
        # Other jobs' allocations remain excluded from the usable budget.
        reserve = max(2 * 2**30, total * self.reserve_fraction)
        self.budget = max(0, free + torch.cuda.memory_reserved(device) - reserve)

    def complete(self, torch, device, count, seconds):
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device)
        self.peak_allocated = max(self.peak_allocated, peak)
        per_item = max(1, (peak - self.baseline) / count)
        estimate = max(1, math.floor((self.budget - self.baseline) / per_item))
        self.size = max(1, min(self.ceiling, estimate, max(count + 1, count * 2)))
        if peak > self.budget:
            self.size = max(1, min(self.size, count // 2))
        event = dict(status="ok", batch_size=count, seconds=seconds,
                     peak_allocated_mib=peak / 2**20, budget_mib=self.budget / 2**20,
                     next_batch_size=self.size)
        self.events.append(event)
        return event

    def oom(self, count, seconds):
        self.ceiling = min(self.ceiling, max(1, count - 1))
        self.size = max(1, min(self.ceiling, count // 2))
        self.events.append(dict(status="oom_retry" if count > 1 else "oom_failed_single",
                                batch_size=count, seconds=seconds, next_batch_size=self.size))


def is_cuda_oom(exc, torch):
    return isinstance(exc, torch.cuda.OutOfMemoryError) or (
        isinstance(exc, RuntimeError) and "cuda" in str(exc).lower()
        and "out of memory" in str(exc).lower())
