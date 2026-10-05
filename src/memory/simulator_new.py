"""
LivingMemoryOS Simulator Compatibility Layer
============================================

Purpose
-------
Provides a stable public interface for accessing the primary
LivingMemoryOS simulation engine.

Overview
--------
This module acts as a lightweight compatibility wrapper around the
MIMIC-IV simulation implementation. It exposes the
MimicLivingMemorySimulator under the generic name
LivingMemorySimulator so that other components of the system can
import a simulator without depending on a dataset-specific class name.

Why This Exists
---------------
As LivingMemoryOS evolves to support multiple datasets and
experimental environments, application code can continue using:

    LivingMemorySimulator

instead of referencing a specific implementation directly.

Current Mapping
---------------
LivingMemorySimulator
    -> MimicLivingMemorySimulator

Benefits
--------
- Simplifies imports throughout the project.
- Provides backward compatibility with earlier versions.
- Allows future simulator implementations to be swapped without
  changing dependent modules.
- Creates a clean abstraction layer between the application and
  dataset-specific simulation engines.

Architecture Role
-----------------
This file serves as an alias and interface bridge within the
LivingMemoryOS simulation framework. It contains no simulation
logic itself; all execution is delegated to the
MimicLivingMemorySimulator implementation.

"""

from src.memory.mimic_simulator import MimicLivingMemorySimulator

LivingMemorySimulator = MimicLivingMemorySimulator

__all__ = ["LivingMemorySimulator", "MimicLivingMemorySimulator"]

