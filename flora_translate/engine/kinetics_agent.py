"""ENGINE — Kinetics Agent: validates residence time via intensification factor."""

import logging
import statistics

from flora_translate.config import (
    DEFAULT_INTENSIFICATION,
    INTENSIFICATION_HIGH_BOUND,
    INTENSIFICATION_LOW_BOUND,
)
from flora_translate.schemas import BatchRecord, CouncilMessage, FlowProposal, ProcessRecord

logger = logging.getLogger("flora.engine.kinetics")


class KineticsAgent:
    """Estimate whether proposed residence time is physically reasonable."""

    def run(
        self,
        batch_record: BatchRecord,
        proposal: FlowProposal,
        analogies: list[dict],
    ) -> CouncilMessage:
        batch_time_h = batch_record.reaction_time_h
        rt_min = proposal.residence_time_min

        if not rt_min or rt_min <= 0:
            return CouncilMessage(
                agent="KineticsAgent",
                status="REJECT",
                field="residence_time_min",
                value=str(rt_min),
                concern="Residence time is zero or not specified.",
                revision_required=True,
                suggested_revision="Set residence_time_min based on literature analogies (typical: 5-30 min).",
            )

        # Compute proposed intensification factor
        if batch_time_h and batch_time_h > 0:
            proposed_factor = (batch_time_h * 60) / rt_min
        else:
            proposed_factor = None

        # Gather intensification factors from analogies
        analogy_factors = []
        for a in analogies:
            full = a.get("full_record", {})
            if not full:
                continue
            eng = full.get("translation_logic", {})
            if eng and eng.get("time_reduction_factor"):
                analogy_factors.append(eng["time_reduction_factor"])
            # Also compute from batch/flow if available
            batch_bl = full.get("batch_baseline", {})
            flow_opt = full.get("flow_optimized", {})
            bt = batch_bl.get("reaction_time_min")
            ft = flow_opt.get("residence_time_min")
            if bt and ft and ft > 0:
                analogy_factors.append(bt / ft)

        # Determine reference median
        if analogy_factors:
            median_factor = statistics.median(analogy_factors)
        else:
            median_factor = DEFAULT_INTENSIFICATION

        # If no batch time provided, can only give general advice
        if proposed_factor is None:
            return CouncilMessage(
                agent="KineticsAgent",
                status="WARNING",
                field="residence_time_min",
                value=str(rt_min),
                concern=(
                    f"No batch reaction time provided — cannot compute intensification factor. "
                    f"Literature median is {median_factor:.0f}x. "
                    f"Proposed residence time of {rt_min:.1f} min treated as a hypothesis."
                ),
                revision_required=False,
            )

        # Compare to literature
        ratio = proposed_factor / median_factor if median_factor else 1.0

        if INTENSIFICATION_LOW_BOUND <= ratio <= INTENSIFICATION_HIGH_BOUND:
            return CouncilMessage(
                agent="KineticsAgent",
                status="ACCEPT",
                field="residence_time_min",
                value=f"{rt_min:.1f} min (intensification={proposed_factor:.0f}x)",
                concern="",
            )

        if ratio < INTENSIFICATION_LOW_BOUND:
            return CouncilMessage(
                agent="KineticsAgent",
                status="WARNING",
                field="residence_time_min",
                value=f"{rt_min:.1f} min (intensification={proposed_factor:.0f}x)",
                concern=(
                    f"Proposed residence time implies intensification_factor={proposed_factor:.0f}x, "
                    f"which is {ratio:.1f}x higher than literature median ({median_factor:.0f}x). "
                    f"Risk of incomplete conversion."
                ),
                revision_required=True,
                suggested_revision=(
                    f"Consider increasing residence_time_min to "
                    f"{(batch_time_h * 60) / median_factor:.1f} min "
                    f"(matching literature median intensification of {median_factor:.0f}x)."
                ),
            )

        # ratio > HIGH_BOUND → too conservative
        return CouncilMessage(
            agent="KineticsAgent",
            status="WARNING",
            field="residence_time_min",
            value=f"{rt_min:.1f} min (intensification={proposed_factor:.0f}x)",
            concern=(
                f"Proposed residence time is conservative. Literature suggests "
                f"shorter times are sufficient (median intensification={median_factor:.0f}x, "
                f"proposed={proposed_factor:.0f}x)."
            ),
            revision_required=False,
        )
