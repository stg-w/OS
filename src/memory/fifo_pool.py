"""
FIFO Memory Pool Module
=======================

Purpose
-------
Implements a standard First-In-First-Out (FIFO) memory replacement policy
used as a baseline benchmark within the LivingMemoryOS framework.

Overview
--------
The FIFO pool maintains a fixed-capacity queue of memory pages.
When the pool reaches its capacity, the oldest page (the first page
admitted into memory) is evicted to make room for a new page.

This implementation does not consider page importance, access frequency,
clinical significance, or patient risk scores. It serves as a traditional
access-history-only replacement strategy against which advanced memory
architectures can be compared.

Role in LivingMemoryOS
----------------------
- Provides a classical memory management baseline.
- Enables performance comparison against the adaptive
  LivingMemoryOS memory retention system.
- Demonstrates the limitations of non-priority-aware
  replacement policies in clinical environments.

Replacement Strategy
--------------------
1. New pages are admitted to the end of the queue.
2. When capacity is reached, the oldest page is removed.
3. No prioritization or protection mechanisms are applied.

Complexity
----------
Admission: O(n) due to front removal operation.
Size Query: O(1)
Page Retrieval: O(n)

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