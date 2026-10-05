"""
LivingMemoryOS Tier Pool

This module implements the core memory management structure used by 
LivingMemoryOS. It maintains a bounded memory pool using a min-heap,
allowing efficient O(log n) insertion and eviction operations based on
Clinical Memory Score (CMS). When capacity is reached, lower-priority
records can be replaced by higher-value records, ensuring that the most
clinically significant patient data remains in memory. The pool also
supports non-evictable tiers, such as EMERGENCY memory, where critical
patient records are protected from replacement once admitted.
"""

import heapq
import itertools

_counter = itertools.count()


class TierPool:
    """
    Min-heap bounded memory tier pool.
    - O(log n) eviction based on lowest Clinical Memory Score (CMS).
    - Supports non-evictable policy (e.g. EMERGENCY tier) to prevent displacing critical patients.
    """

    def __init__(self, capacity: int, evictable: bool = True):
        self.capacity = capacity
        self.evictable = evictable
        self.heap = []  # entries of (cms, tie_breaker_id, page_dict)
        self.by_id = {}

    def admit(self, page: dict) -> bool:
        """
        Attempts to admit a page into this tier pool.
        Returns True if admitted, False if rejected.
        """
        if len(self.heap) < self.capacity:
            entry = (page["cms"], next(_counter), page)
            heapq.heappush(self.heap, entry)
            self.by_id[page["page_id"]] = page
            return True

        if not self.evictable:
            # Capacity reached in non-evictable emergency tier -> reject / alert
            return False

        weakest_cms, _, weakest_page = self.heap[0]
        if page["cms"] > weakest_cms:
            heapq.heapreplace(self.heap, (page["cms"], next(_counter), page))
            del self.by_id[weakest_page["page_id"]]
            self.by_id[page["page_id"]] = page
            return True

        return False

    def pages(self) -> list:
        """Returns list of admitted pages sorted descending by CMS."""
        return sorted([p for _, _, p in self.heap], key=lambda x: x["cms"], reverse=True)

    def size(self) -> int:
        return len(self.heap)