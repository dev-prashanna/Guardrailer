"""
research/__init__.py
Research evaluation framework for Guardrailer LLM guardrail systems.

Provides publication-ready experimental methodology:
- Stratified K-Fold Cross-Validation
- Bootstrap Confidence Intervals
- Power Analysis
- Dataset Engineering Pipeline
- Ablation Studies
- Robustness Testing
- Statistical Significance Testing
"""

from guardrailer_security.research.statistics import (
    StratifiedKFoldEvaluator,
    BootstrapConfidenceInterval,
    PowerAnalysis,
)
from guardrailer_security.research.dataset_engineering import (
    DatasetAcquisitionPipeline,
    BenignSampleGenerator,
    HardnessScorer,
    DatasetValidator,
)
from guardrailer_security.research.experimental_design import (
    ResearchMetrics,
    AblationStudyRunner,
    SignificanceTester,
    EvaluationLifecycle,
)
from guardrailer_security.research.robustness import (
    ObfuscationTester,
    MultilingualTester,
    ReproducibilityManager,
)

__all__ = [
    "StratifiedKFoldEvaluator",
    "BootstrapConfidenceInterval",
    "PowerAnalysis",
    "DatasetAcquisitionPipeline",
    "BenignSampleGenerator",
    "HardnessScorer",
    "DatasetValidator",
    "ResearchMetrics",
    "AblationStudyRunner",
    "SignificanceTester",
    "EvaluationLifecycle",
    "ObfuscationTester",
    "MultilingualTester",
    "ReproducibilityManager",
]
