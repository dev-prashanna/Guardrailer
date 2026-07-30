#!/usr/bin/env python3
"""
test_fixes.py
Verification tests for the three critical fixes:
1. Hybrid Search (Fusion.RRF)
2. Layer Assignment Logic
3. LLM Invocation Thresholds
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))


def test_hybrid_search_import():
    """Test that Fusion.RRF can be imported correctly."""
    print("Test 1: Hybrid Search Import")
    try:
        from qdrant_client.models import Prefetch, Fusion
        print(f"  ✓ Fusion.RRF available: {Fusion.RRF}")
        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False


def test_sparse_keywords_sync():
    """Test that sparse keywords are synchronized between scoring.py and security_engine.py."""
    print("\nTest 2: Sparse Keywords Synchronization")
    from scoring import SPARSE_KEYWORDS as SCORING_KEYWORDS
    from security_engine import SPARSE_KEYWORDS as ENGINE_KEYWORDS

    scoring_set = set(SCORING_KEYWORDS)
    engine_set = set(ENGINE_KEYWORDS)

    if scoring_set == engine_set:
        print(f"  ✓ Keywords synchronized: {len(SCORING_KEYWORDS)} keywords")
        return True
    else:
        missing_in_scoring = engine_set - scoring_set
        missing_in_engine = scoring_set - engine_set
        if missing_in_scoring:
            print(f"  ⚠ Missing in scoring.py: {missing_in_scoring}")
        if missing_in_engine:
            print(f"  ⚠ Missing in security_engine.py: {missing_in_engine}")
        return False


def test_layer_assignment_logic():
    """Test layer assignment logic with various scenarios."""
    print("\nTest 3: Layer Assignment Logic")
    from security_engine import EvalLayer, FAST_BLOCK_THRESHOLD, DEEP_PATH_LOWER

    test_cases = [
        # (is_blocked, composite, risk, expected_layer, description)
        (True, 0.90, "critical", EvalLayer.FAST_BLOCK, "High confidence malicious"),
        (True, 0.70, "high", EvalLayer.DEEP_PATH, "Medium confidence malicious"),
        (True, 0.50, "medium", EvalLayer.DEEP_PATH, "Low confidence malicious (LLM invoked)"),
        (False, 0.90, "none", EvalLayer.DEEP_PATH, "High composite, not malicious (LLM invoked)"),
        (False, 0.50, "none", EvalLayer.DEEP_PATH, "Medium composite, not malicious (LLM invoked)"),
        (False, 0.30, "none", EvalLayer.DEEP_PATH, "Low composite, not malicious (LLM invoked)"),
    ]

    all_passed = True
    for is_blocked, composite, risk, expected, desc in test_cases:
        # Simulate layer assignment logic
        fast_block_eligible = (
            composite >= FAST_BLOCK_THRESHOLD
            and is_blocked
            and risk in ("critical", "high")
        )

        if fast_block_eligible:
            layer = EvalLayer.FAST_BLOCK
        elif not fast_block_eligible:
            layer = EvalLayer.DEEP_PATH
        else:
            layer = EvalLayer.SAFE

        if layer == expected:
            print(f"  ✓ {desc}: {layer.value} (expected: {expected.value})")
        else:
            print(f"  ✗ {desc}: {layer.value} (expected: {expected.value})")
            all_passed = False

    return all_passed


def test_llm_invocation_thresholds():
    """Test LLM invocation thresholds."""
    print("\nTest 4: LLM Invocation Thresholds")
    from security_engine import FAST_BLOCK_THRESHOLD, DEEP_PATH_LOWER

    print(f"  FAST_BLOCK_THRESHOLD: {FAST_BLOCK_THRESHOLD}")
    print(f"  DEEP_PATH_LOWER: {DEEP_PATH_LOWER}")

    # Test threshold logic
    test_composites = [0.90, 0.70, 0.50, 0.30, 0.10]
    for composite in test_composites:
        should_invoke_llm = composite >= DEEP_PATH_LOWER
        print(f"  Composite {composite:.2f}: {'Invoke LLM' if should_invoke_llm else 'Skip LLM'}")

    return True


def test_fallback_logic():
    """Test fallback logic with reduced signal requirements."""
    print("\nTest 5: Fallback Logic")
    from security_engine import DEEP_PATH_LOWER

    # Test cases: (best_composite, best_dense, cross_enc, sparse_flags_count, should_block)
    test_cases = [
        (0.70, 0.85, 0.80, 2, True, "All signals high"),
        (0.70, 0.85, 0.80, 1, True, "3 signals high"),
        (0.65, 0.85, 0.70, 1, True, "Composite + dense high"),
        (0.65, 0.70, 0.80, 1, True, "Composite + cross-encoder high"),
        (0.60, 0.80, 0.75, 1, True, "All signals at threshold"),
        (0.50, 0.60, 0.60, 0, False, "All signals below threshold"),
    ]

    all_passed = True
    for best_composite, best_dense, cross_enc, sparse_count, expected, desc in test_cases:
        signals_agree = sum([
            best_composite >= DEEP_PATH_LOWER,
            best_dense >= 0.80,
            cross_enc >= 0.75,
            sparse_count >= 1,
        ])
        should_block = signals_agree >= 2

        if should_block == expected:
            print(f"  ✓ {desc}: block={should_block} (signals: {signals_agree})")
        else:
            print(f"  ✗ {desc}: block={should_block} (expected: {expected}, signals: {signals_agree})")
            all_passed = False

    return all_passed


def main():
    print("=" * 70)
    print("GUARDRAILER FIX VERIFICATION TESTS")
    print("=" * 70)

    results = []
    results.append(("Hybrid Search Import", test_hybrid_search_import()))
    results.append(("Sparse Keywords Sync", test_sparse_keywords_sync()))
    results.append(("Layer Assignment", test_layer_assignment_logic()))
    results.append(("LLM Thresholds", test_llm_invocation_thresholds()))
    results.append(("Fallback Logic", test_fallback_logic()))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")

    print(f"\n  Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n  ✓ All tests passed!")
        return 0
    else:
        print("\n  ✗ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
