"""ENGINE — Kinetics Agent: validates residence time against first-principles calculations.

Reads the DesignCalculator results and compares with the proposal.
Issues quantitative REJECT when the proposal deviates from physics.
"""

import logging
import statistics

from flora_translate.config import (
    DEFAULT_INTENSIFICATION,
    INTENSIFICATION_HIGH_BOUND,
    INTENSIFICATION_LOW_BOUND,
)
from flora_translate.schemas import BatchRecord, CouncilMessage, FlowProposal

logger = logging.getLogger("flora.engine.kinetics")


class KineticsAgent:
    """Validate that the proposed residence time is consistent with kinetics."""

    def run(
        self,
        batch_record: BatchRecord,
        proposal: FlowProposal,
        analogies: list[dict],
        calculations=None,
    ) -> CouncilMessage:
        rt_min = proposal.residence_time_min
        batch_time_h = batch_record.reaction_time_h

        # ── Zero or missing RT → REJECT with calculated value ───────────
        if not rt_min or rt_min <= 0:
            suggested = (
                f"{calculations.residence_time_min:.1f}"
                if calculations and calculations.residence_time_min
                else "10"
            )
            return CouncilMessage(
                agent="KineticsAgent",
                status="REJECT",
                field="residence_time_min",
                value="0",
                concern="Residence time is zero or not specified.",
                revision_required=True,
                suggested_revision=suggested,
            )

        # ── Compare with calculator ─────────────────────────────────────
        if calculations and calculations.residence_time_min > 0:
            calc_rt = calculations.residence_time_min
            ratio = abs(rt_min - calc_rt) / calc_rt

            # Build detailed concern with calculation provenance
            calc_detail = (
                f"[Calculation: τ = {calc_rt:.1f} min via "
                f"{calculations.kinetics_method}, "
                f"IF = {calculations.intensification_factor:.0f}×, "
                f"Da = {calculations.damkohler_mass:.2f}]"
            )

            if ratio > 0.5:
                # Major deviation → REJECT
                return CouncilMessage(
                    agent="KineticsAgent",
                    status="REJECT",
                    field="residence_time_min",
                    value=f"{rt_min:.1f} min",
                    concern=(
                        f"Proposal τ = {rt_min:.1f} min deviates {ratio*100:.0f}% "
                        f"from calculated τ = {calc_rt:.1f} min. {calc_detail}"
                    ),
                    revision_required=True,
                    suggested_revision=f"{calc_rt:.1f}",
                )
            elif ratio > 0.2:
                # Moderate deviation → WARNING
                return CouncilMessage(
                    agent="KineticsAgent",
                    status="WARNING",
                    field="residence_time_min",
                    value=f"{rt_min:.1f} min",
                    concern=(
                        f"Proposal τ = {rt_min:.1f} min differs {ratio*100:.0f}% "
                        f"from calculated τ = {calc_rt:.1f} min. {calc_detail}"
                    ),
                    revision_required=False,
                )
            else:
                # Good agreement
                return CouncilMessage(
                    agent="KineticsAgent",
                    status="ACCEPT",
                    field="residence_time_min",
                    value=(
                        f"{rt_min:.1f} min "
                        f"(calc: {calc_rt:.1f} min, Δ = {ratio*100:.0f}%) "
                        f"{calc_detail}"
                    ),
                    concern="",
                )

        # ── Fallback: compare with literature intensification ───────────
        if batch_time_h and batch_time_h > 0:
            proposed_factor = (batch_time_h * 60) / rt_min

            analogy_factors = []
            for a in analogies:
                full = a.get("full_record", {})
                if not full:
                    continue
                eng = full.get("translation_logic", {})
                if eng and eng.get("time_reduction_factor"):
                    analogy_factors.append(eng["time_reduction_factor"])
                batch_bl = full.get("batch_baseline", {})
                flow_opt = full.get("flow_optimized", {})
                bt = batch_bl.get("reaction_time_min")
                ft = flow_opt.get("residence_time_min")
                if bt and ft and ft > 0:
                    analogy_factors.append(bt / ft)

            median_factor = (
                statistics.median(analogy_factors) if analogy_factors
                else DEFAULT_INTENSIFICATION
            )
            ratio = proposed_factor / median_factor if median_factor else 1.0

            if ratio < INTENSIFICATION_LOW_BOUND:
                suggested_rt = (batch_time_h * 60) / median_factor
                return CouncilMessage(
                    agent="KineticsAgent",
                    status="REJECT",
                    field="residence_time_min",
                    value=f"{rt_min:.1f} min (IF = {proposed_factor:.0f}×)",
                    concern=(
                        f"IF = {proposed_factor:.0f}× is {ratio:.1f}× above "
                        f"literature median ({median_factor:.0f}×). "
                        "Risk of incomplete conversion."
                    ),
                    revision_required=True,
                    suggested_revision=f"{suggested_rt:.1f}",
                )
            if ratio > INTENSIFICATION_HIGH_BOUND:
                return CouncilMessage(
                    agent="KineticsAgent",
                    status="WARNING",
                    field="residence_time_min",
                    value=f"{rt_min:.1f} min (IF = {proposed_factor:.0f}×)",
                    concern=(
                        f"Conservative: proposed IF = {proposed_factor:.0f}×, "
                        f"literature median = {median_factor:.0f}×. "
                        "Could likely reduce τ."
                    ),
                    revision_required=False,
                )
            return CouncilMessage(
                agent="KineticsAgent",
                status="ACCEPT",
                field="residence_time_min",
                value=(
                    f"{rt_min:.1f} min "
                    f"(IF = {proposed_factor:.0f}×, "
                    f"median = {median_factor:.0f}×)"
                ),
                concern="",
            )

        # ── No batch time, no calculations → accept cautiously ──────────
        return CouncilMessage(
            agent="KineticsAgent",
            status="ACCEPT",
            field="residence_time_min",
            value=f"{rt_min:.1f} min",
            concern="No batch time or calculation available — treated as hypothesis.",
        )
