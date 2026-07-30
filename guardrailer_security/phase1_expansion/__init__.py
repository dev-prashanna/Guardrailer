"""
phase1_expansion
Dataset expansion module for Guardrailer Phase 1.
Provides research dataset collection, synthetic attack generation,
edge case injection, and benchmarking framework.
"""

from .schema import (
    AttackCategory,
    AttackTechnique,
    RiskLevel,
    DatasetSource,
    SampleRecord,
    BenchmarkEntry,
    MultiTurnSequence,
)
from .research_collectors import collect_all_research_datasets
from .synthetic_generator import SyntheticAttackGenerator
from .edge_cases import EdgeCaseGenerator
from .benchmark_framework import GuardrailerBenchmark
from .orchestrator import Phase1Orchestrator

__all__ = [
    "AttackCategory",
    "AttackTechnique",
    "RiskLevel",
    "DatasetSource",
    "SampleRecord",
    "BenchmarkEntry",
    "MultiTurnSequence",
    "collect_all_research_datasets",
    "SyntheticAttackGenerator",
    "EdgeCaseGenerator",
    "GuardrailerBenchmark",
    "Phase1Orchestrator",
]
