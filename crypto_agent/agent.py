try:
    from ..decoder.crypto_decoder import multi_layer_decode, AnalysisResult
    from ..classifier.safety_classifier import (
        analyze_encoded_payload,
        ClassificationResult,
        ThreatLevel,
        ThreatCategory,
    )
    from ..classifier.llm_classifier import classify_with_llm, LLMClassification
except ImportError:
    from decoder.crypto_decoder import multi_layer_decode, AnalysisResult
    from classifier.safety_classifier import (
        analyze_encoded_payload,
        ClassificationResult,
        ThreatLevel,
        ThreatCategory,
    )
    from classifier.llm_classifier import classify_with_llm, LLMClassification


class CryptoAgent:
    def __init__(self, use_llm: bool = True, use_rag: bool = True):
        self.use_llm = use_llm
        self.use_rag = use_rag

    def analyze(self, input_text: str) -> dict:
        analysis = multi_layer_decode(input_text)
        classification = analyze_encoded_payload(analysis)

        llm_result = None
        if self.use_llm:
            llm_result = classify_with_llm(
                decoded_text=analysis.fully_decoded,
                original_length=len(input_text),
                encoding_layers=[r.encoding_type.value for r in analysis.decoding_chain],
                entropy=analysis.entropy,
                pattern_matches=[ind["pattern"] for ind in analysis.malicious_indicators],
                use_rag=self.use_rag,
            )

        final_threat_level = classification.threat_level
        final_category = classification.threat_category
        final_confidence = classification.confidence
        final_reasoning = classification.reasoning
        source = "rule_based"

        if llm_result:
            source = "llm_rag_enhanced" if self.use_rag and llm_result.retrieved_examples else "llm_enhanced"
            if llm_result.is_malicious:
                if final_threat_level in (ThreatLevel.SAFE, ThreatLevel.LOW):
                    final_threat_level = ThreatLevel.MEDIUM
                    final_confidence = max(final_confidence, llm_result.confidence)
                final_reasoning += f"\n\n[LLM Analysis] {llm_result.reasoning}"
            else:
                if final_threat_level in (ThreatLevel.CRITICAL, ThreatLevel.HIGH):
                    final_confidence = min(final_confidence, llm_result.confidence)
                    final_reasoning += f"\n\n[LLM Analysis] {llm_result.reasoning}"
                elif final_threat_level == ThreatLevel.MEDIUM and llm_result.confidence < 0.3:
                    final_threat_level = ThreatLevel.LOW
                    final_reasoning += f"\n\n[LLM Analysis] {llm_result.reasoning}"

        return {
            "original_input": input_text,
            "is_encoded": analysis.is_encoded,
            "encoding_layers": [r.encoding_type.value for r in analysis.decoding_chain],
            "fully_decoded": analysis.fully_decoded,
            "hash_identifications": [
                {
                    "type": h.hash_type.value,
                    "length": h.length,
                    "confidence": round(h.confidence, 3),
                    "metadata": h.metadata,
                }
                for h in analysis.hash_identifications
            ],
            "malicious_indicators": analysis.malicious_indicators,
            "entropy": round(analysis.entropy, 4),
            "classification": {
                "threat_level": final_threat_level.value,
                "threat_category": final_category.value,
                "confidence": round(final_confidence, 4),
                "reasoning": final_reasoning,
                "recommendations": classification.recommendations,
                "encoding_layers": classification.encoding_layers,
                "analysis_source": source,
            },
            "llm_analysis": {
                "available": llm_result is not None,
                "is_malicious": llm_result.is_malicious if llm_result else None,
                "threat_category": llm_result.threat_category if llm_result else None,
                "confidence": round(llm_result.confidence, 4) if llm_result else None,
                "reasoning": llm_result.reasoning if llm_result else None,
                "decoded_intent": llm_result.decoded_intent if llm_result else None,
                "recommended_action": llm_result.recommended_action if llm_result else None,
            } if llm_result else None,
            "retrieved_examples": [
                {
                    "text": ex.text,
                    "category": ex.category,
                    "technique": ex.technique,
                    "severity": ex.severity,
                    "similarity": ex.similarity,
                }
                for ex in (llm_result.retrieved_examples if llm_result else [])
            ],
            "decoding_chain": [
                {
                    "layer": i + 1,
                    "encoding": r.encoding_type.value,
                    "decoded_preview": r.decoded_text[:200] + ("..." if len(r.decoded_text) > 200 else ""),
                    "confidence": round(r.confidence, 3),
                }
                for i, r in enumerate(analysis.decoding_chain)
            ],
        }


agent = CryptoAgent()


def analyze_input(text: str, use_llm: bool = True, use_rag: bool = True) -> dict:
    a = CryptoAgent(use_llm=use_llm, use_rag=use_rag)
    return a.analyze(text)
