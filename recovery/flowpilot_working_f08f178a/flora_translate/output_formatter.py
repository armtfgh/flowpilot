"""FLORA-Translate — Output formatter: structured JSON + human-readable."""

import json
import logging

import anthropic

from flora_translate.config import SUMMARY_MODEL
from flora_translate.schemas import DesignCandidate

logger = logging.getLogger("flora.output")

def _get_client():
    return anthropic.Anthropic()

EXPLANATION_SYSTEM = (
    "You are a flow chemistry expert. Convert this validated flow chemistry design "
    "into a clear, concise explanation for a synthetic chemist. Structure it as:\n"
    "(1) Proposed flow setup\n"
    "(2) Why these conditions were chosen (cite analogies by DOI or title)\n"
    "(3) Key differences from the batch protocol\n"
    "(4) Any warnings or flags from the engineering validation\n"
    "Use plain technical language. No marketing language."
)


class OutputFormatter:
    """Format DesignCandidate into structured JSON + human explanation."""

    def format(
        self,
        candidate: DesignCandidate,
        analogies: list[dict],
    ) -> dict:
        """Generate the final output dict with both structured and readable forms."""
        # Compute confidence from two independent signals:
        #   1. Analogy quality — max final_score from retrieval.
        #      Scores now use cosine similarity (corrected from L2 in retriever.py),
        #      so calibrated ranges are:
        #        > 0.75  → strong match (same class, similar conditions)
        #        > 0.50  → decent match (related class)
        #        ≤ 0.50  → weak/no match
        #   2. Council convergence — how many revision rounds were needed.
        #      MIN_COUNCIL_ROUNDS = 2, so 2 rounds = cleanest possible run.
        #        ≤ 2 rounds → no significant issues found
        #        3 rounds   → required corrections (acceptable)
        max_score = max(
            (a.get("final_score", 0) for a in analogies), default=0
        )
        rounds = candidate.council_rounds
        if max_score > 0.75 and rounds <= 2:
            confidence = "HIGH"
        elif max_score > 0.50 and rounds <= 3:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        candidate.proposal.confidence = confidence

        # Generate human-readable explanation
        explanation = self._generate_explanation(candidate)
        candidate.human_explanation = explanation

        # Add confidence-appropriate note
        if confidence == "LOW":
            candidate.human_explanation += (
                "\n\n**Confidence: LOW** — The closest literature analogy has a "
                f"similarity score of {max_score:.2f} (threshold for MEDIUM: 0.50). "
                "No close precedent found in the corpus for this exact reaction class "
                "and conditions. Treat this proposal as a starting hypothesis "
                "requiring careful experimental validation before scale-up."
            )
        elif confidence == "MEDIUM":
            candidate.human_explanation += (
                f"\n\n**Confidence: MEDIUM** — Best analogy score: {max_score:.2f}. "
                "A related literature precedent was found but conditions differ. "
                "Validate key parameters (residence time, concentration) experimentally."
            )

        return {
            "proposal": candidate.proposal.model_dump(),
            "unit_operations": candidate.unit_operations,
            "pid_description": candidate.pid_description,
            "council_rounds": candidate.council_rounds,
            "safety_report": candidate.safety_report,
            "council_messages": [
                m.model_dump() for m in candidate.council_messages
            ],
            "confidence": confidence,
            "explanation": candidate.human_explanation,
        }

    def _generate_explanation(self, candidate: DesignCandidate) -> str:
        """Use Claude to generate a human-readable explanation."""
        try:
            resp = _get_client().messages.create(
                model=SUMMARY_MODEL,
                max_tokens=1500,
                system=EXPLANATION_SYSTEM,
                messages=[
                    {
                        "role": "user",
                        "content": json.dumps(
                            candidate.model_dump(), indent=2, default=str
                        ),
                    }
                ],
            )
            return resp.content[0].text.strip()
        except Exception as e:
            logger.warning(f"Failed to generate explanation: {e}")
            # Fallback: basic text summary
            p = candidate.proposal
            return (
                f"Flow Proposal: {p.reactor_type} reactor, "
                f"{p.tubing_material} {p.tubing_ID_mm}mm ID, "
                f"{p.reactor_volume_mL}mL volume, "
                f"{p.residence_time_min} min residence time, "
                f"{p.flow_rate_mL_min} mL/min flow rate, "
                f"{p.temperature_C}°C, {p.concentration_M}M, "
                f"BPR {p.BPR_bar} bar. "
                f"Confidence: {p.confidence}."
            )
