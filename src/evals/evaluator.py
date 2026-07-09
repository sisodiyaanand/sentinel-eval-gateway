from typing import Dict, Optional
from src.evals.metrics import (
    compute_toxicity_score,
    compute_semantic_similarity,
    compute_hallucination_proxy,
)
from src.core.config import settings


class SentinelEvaluator:
    """Runs real, computed evaluations on prompt/response pairs.
    No mock or hardcoded values - every score is derived from the
    actual input text."""

    async def evaluate_interaction(
        self, prompt: str, output: str, context: Optional[str] = None
    ) -> Dict[str, float]:
        toxicity = compute_toxicity_score(output)
        relevance = compute_semantic_similarity(prompt, output)
        hallucination = compute_hallucination_proxy(output, context)

        return {
            "toxicity": toxicity,
            "relevance_to_prompt": relevance,
            "hallucination_risk": hallucination,
        }

    def passes_safety_gate(self, metrics: Dict[str, float]) -> bool:
        return (
            metrics["toxicity"] < settings.TOXICITY_THRESHOLD
            and metrics["relevance_to_prompt"] > settings.SIMILARITY_THRESHOLD
        )