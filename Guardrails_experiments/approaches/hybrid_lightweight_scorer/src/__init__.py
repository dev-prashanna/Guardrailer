try:
    from .scorer import HybridScorer
    from .features import extract_features
    from .patterns import pattern_classify
    from .similarity import TFIDFSimilarity
except ImportError:
    from scorer import HybridScorer
    from features import extract_features
    from patterns import pattern_classify
    from similarity import TFIDFSimilarity
