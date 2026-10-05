"""
FIFO Memory Pool Compatibility Module
=====================================

Purpose
-------
Provides a compatibility alias for the FIFO memory pool implementation
used throughout the LivingMemoryOS framework.

Overview
--------
This module re-exports the FifoMemoryPool class from the primary
implementation file (`fifo_pool.py`). It exists to preserve backward
compatibility with older imports and simplify module organization.

Role in LivingMemoryOS
----------------------
- Acts as a lightweight wrapper around the FIFO baseline memory pool.
- Maintains stable import paths across different project versions.
- Prevents breaking changes when internal module structures evolve.

Notes
-----
No additional functionality is implemented in this file.
All queue management and replacement logic are handled by
the underlying FifoMemoryPool class.

"""

from src.memory.fifo_pool import FifoMemoryPool

__all__ = ["FifoMemoryPool"]

