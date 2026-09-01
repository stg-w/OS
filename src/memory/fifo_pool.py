"""
FIFO Memory Pool Module
Baseline access-history-only FIFO replacement queue.
"""


class FifoMemoryPool:
    """Standard First-In-First-Out (FIFO) queue for memory page replacement baseline."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.queue = []

    def admit(self, page: dict):
        if len(self.queue) >= self.capacity:
            self.queue.pop(0)
        self.queue.append(page)

    def pages(self) -> list:
        return list(self.queue)

    def size(self) -> int:
        return len(self.queue)
