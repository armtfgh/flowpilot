"""ENGINE — Chemistry Validator (Layer 3).

Validates the FlowProposal AGAINST the ChemistryPlan to catch
chemistry errors that the hardware-focused agents miss:
  - Wavelength vs photocatalyst mismatch
  - Incompatible reagents in same stream
  - Missing deoxygenation for O2-sensitive reaction
  - Missing quench for reactive intermediates
  - Stream assignments contradict chemistry plan
  - Concentration sanity for mechanism type
"""

import logging

from flora_translate.schemas import (
    ChemistryPlan,
    CouncilMessage,
    FlowProposal,
)

logger = logging.getLogger("flora.engine.chemistry_validator")


class ChemistryValidator:
    """Validate flow proposal against the chemistry plan."""

    def run(
        self,
        proposal: FlowProposal,
        chemistry_plan: ChemistryPlan,
    ) -> list[CouncilMessage]:
        messages = []

        messages.extend(self._check_wavelength(proposal, chemistry_plan))
        messages.extend(self._check_incompatible_pairs(proposal, chemistry_plan))
        messages.extend(self._check_deoxygenation(proposal, chemistry_plan))
        messages.extend(self._check_quench(proposal, chemistry_plan))
        messages.extend(self._check_stream_assignments(proposal, chemistry_plan))
        messages.extend(self._check_concentration(proposal, chemistry_plan))
        messages.extend(self._check_light_sensitive(proposal, chemistry_plan))

        if not messages:
            messages.append(
                CouncilMessage(
                    agent="ChemistryValidator",
                    status="ACCEPT",
                    field="chemistry",
                    value="All chemistry checks passed",
                )
            )

        return messages

    def _check_wavelength(
        self, proposal: FlowProposal, plan: ChemistryPlan
    ) -> list[CouncilMessage]:
        """Verify proposed wavelength matches photocatalyst absorption."""
        if not plan.recommended_wavelength_nm or not proposal.wavelength_nm:
            return []

        diff = abs(proposal.wavelength_nm - plan.recommended_wavelength_nm)
        if diff > 50:
            return [
                CouncilMessage(
                    agent="ChemistryValidator",
                    status="REJECT",
                    field="wavelength_nm",
                    value=f"proposed={proposal.wavelength_nm}nm, catalyst needs={plan.recommended_wavelength_nm}nm",
                    concern=(
                        f"Proposed wavelength ({proposal.wavelength_nm}nm) is "
                        f"{diff:.0f}nm away from photocatalyst absorption maximum "
                        f"({plan.recommended_wavelength_nm}nm). "
                        f"{plan.wavelength_reasoning}"
                    ),
                    revision_required=True,
                    suggested_revision=f"Set wavelength_nm to {plan.recommended_wavelength_nm}.",
                )
            ]
        if diff > 30:
            return [
                CouncilMessage(
                    agent="ChemistryValidator",
                    status="WARNING",
                    field="wavelength_nm",
                    value=f"proposed={proposal.wavelength_nm}nm, optimal={plan.recommended_wavelength_nm}nm",
                    concern=(
                        f"Proposed wavelength is {diff:.0f}nm from optimal. "
                        f"May reduce photocatalytic efficiency."
                    ),
                )
            ]
        return []

    def _check_incompatible_pairs(
        self, proposal: FlowProposal, plan: ChemistryPlan
    ) -> list[CouncilMessage]:
        """Verify that incompatible reagent pairs are NOT in the same stream."""
        if not plan.incompatible_pairs or not proposal.streams:
            return []

        messages = []
        for pair in plan.incompatible_pairs:
            if len(pair) < 2:
                continue
            a, b = pair[0].lower(), pair[1].lower()

            for stream in proposal.streams:
                contents_lower = [c.lower() for c in stream.contents]
                contents_text = " ".join(contents_lower)
                a_in = any(a in c for c in contents_lower) or a in contents_text
                b_in = any(b in c for c in contents_lower) or b in contents_text

                if a_in and b_in:
                    messages.append(
                        CouncilMessage(
                            agent="ChemistryValidator",
                            status="REJECT",
                            field=f"stream_{stream.stream_label}",
                            value=f"'{pair[0]}' and '{pair[1]}' in same stream {stream.stream_label}",
                            concern=(
                                f"INCOMPATIBLE: '{pair[0]}' and '{pair[1]}' are in the "
                                f"same stream ({stream.stream_label}). The Chemistry Plan "
                                f"identified these as incompatible — they must be in "
                                f"separate streams to prevent premature reaction."
                            ),
                            revision_required=True,
                            suggested_revision=(
                                f"Move '{pair[1]}' to a separate stream from '{pair[0]}'."
                            ),
                        )
                    )
        return messages

    def _check_deoxygenation(
        self, proposal: FlowProposal, plan: ChemistryPlan
    ) -> list[CouncilMessage]:
        """Verify deoxygenation is specified if chemistry requires it."""
        if not plan.deoxygenation_required:
            return []

        has_degas = bool(proposal.deoxygenation_method)
        has_pre_step = any(
            "degas" in s.lower() or "deoxy" in s.lower() or "sparg" in s.lower()
            for s in proposal.pre_reactor_steps
        )

        if not has_degas and not has_pre_step:
            return [
                CouncilMessage(
                    agent="ChemistryValidator",
                    status="REJECT",
                    field="deoxygenation_method",
                    value="missing",
                    concern=(
                        f"Reaction is oxygen-sensitive but no deoxygenation specified. "
                        f"{plan.deoxygenation_reasoning}"
                    ),
                    revision_required=True,
                    suggested_revision="Add deoxygenation_method (N2 sparging or freeze-pump-thaw) and corresponding pre_reactor_step.",
                )
            ]
        return []

    def _check_quench(
        self, proposal: FlowProposal, plan: ChemistryPlan
    ) -> list[CouncilMessage]:
        """Verify quench is specified if chemistry requires it."""
        if not plan.quench_required:
            return []

        has_quench = any(
            "quench" in s.lower() for s in proposal.post_reactor_steps
        )

        if not has_quench:
            return [
                CouncilMessage(
                    agent="ChemistryValidator",
                    status="WARNING",
                    field="post_reactor_steps",
                    value="missing quench",
                    concern=(
                        f"Chemistry Plan requires quench with {plan.quench_reagent}. "
                        f"{plan.quench_reasoning}"
                    ),
                    revision_required=False,
                    suggested_revision=f"Add post-reactor inline quench with {plan.quench_reagent}.",
                )
            ]
        return []

    def _check_stream_assignments(
        self, proposal: FlowProposal, plan: ChemistryPlan
    ) -> list[CouncilMessage]:
        """Check that the proposal's stream assignments broadly match the plan's stream_logic."""
        if not plan.stream_logic or not proposal.streams:
            return []

        messages = []
        # Check each planned stream's reagents appear somewhere in the proposal
        for planned in plan.stream_logic:
            for reagent in planned.reagents:
                reagent_lower = reagent.lower()
                found = False
                for prop_stream in proposal.streams:
                    if any(reagent_lower in c.lower() for c in prop_stream.contents):
                        found = True
                        break
                if not found:
                    messages.append(
                        CouncilMessage(
                            agent="ChemistryValidator",
                            status="WARNING",
                            field=f"stream_{planned.stream_label}",
                            value=f"'{reagent}' not found in any stream",
                            concern=(
                                f"Chemistry Plan expects '{reagent}' in stream "
                                f"{planned.stream_label}, but it's missing from the "
                                f"proposal's stream assignments."
                            ),
                        )
                    )
        return messages

    def _check_concentration(
        self, proposal: FlowProposal, plan: ChemistryPlan
    ) -> list[CouncilMessage]:
        """Sanity check concentration for the mechanism type."""
        conc = proposal.concentration_M
        if not conc or conc <= 0:
            return []

        messages = []
        mech = plan.mechanism_type.lower() if plan.mechanism_type else ""

        # Radical chain reactions generally need higher concentration
        if "radical chain" in mech and conc < 0.01:
            messages.append(
                CouncilMessage(
                    agent="ChemistryValidator",
                    status="WARNING",
                    field="concentration_M",
                    value=f"{conc} M",
                    concern=(
                        "Radical chain reactions typically require [substrate] > 0.01M "
                        "for efficient chain propagation. Proposed concentration may be "
                        "too dilute."
                    ),
                )
            )

        # Very high concentration in photochemistry limits light penetration
        if conc > 0.5 and plan.mechanism_type:
            messages.append(
                CouncilMessage(
                    agent="ChemistryValidator",
                    status="WARNING",
                    field="concentration_M",
                    value=f"{conc} M",
                    concern=(
                        "High concentration (>0.5M) in photochemistry reduces light "
                        "penetration depth. Consider diluting to improve irradiation "
                        "uniformity, especially with narrow tubing."
                    ),
                )
            )

        return messages

    def _check_light_sensitive(
        self, proposal: FlowProposal, plan: ChemistryPlan
    ) -> list[CouncilMessage]:
        """Warn if light-sensitive reagents are in the feed — may decompose before reactor."""
        if not plan.light_sensitive_reagents or not proposal.streams:
            return []

        messages = []
        for reagent in plan.light_sensitive_reagents:
            r_lower = reagent.lower()
            for stream in proposal.streams:
                if any(r_lower in c.lower() for c in stream.contents):
                    messages.append(
                        CouncilMessage(
                            agent="ChemistryValidator",
                            status="WARNING",
                            field=f"stream_{stream.stream_label}",
                            value=f"'{reagent}' is light-sensitive",
                            concern=(
                                f"'{reagent}' is light-sensitive and in stream "
                                f"{stream.stream_label}. Ensure feed tubing is opaque "
                                f"or shielded from light before entering the reactor."
                            ),
                        )
                    )
        return messages
