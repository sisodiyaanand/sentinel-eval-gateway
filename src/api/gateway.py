from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, Dict
from src.attacks.jailbreak_gcg import PromptVariationAttack
from src.evals.evaluator import SentinelEvaluator

router = APIRouter()
attack_engine = PromptVariationAttack()
eval_engine = SentinelEvaluator()


class InterceptRequest(BaseModel):
    prompt: str
    response: str
    context: Optional[str] = None


class InterceptResponse(BaseModel):
    original_prompt: str
    adversarial_variants_tested: int
    passed_safety_gate: bool
    metrics: Dict[str, float]


@router.post("/evaluate", response_model=InterceptResponse)
async def intercept_and_evaluate(payload: InterceptRequest):
    adv_prompts = await attack_engine.generate_adversarial_prompts(payload.prompt)
    metrics = await eval_engine.evaluate_interaction(
        prompt=payload.prompt,
        output=payload.response,
        context=payload.context,
    )
    passed = eval_engine.passes_safety_gate(metrics)

    return InterceptResponse(
        original_prompt=payload.prompt,
        adversarial_variants_tested=len(adv_prompts),
        passed_safety_gate=passed,
        metrics=metrics,
    )