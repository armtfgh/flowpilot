Schema-routing control — output-only benchmark supplement

run_followup.py is a frozen external harness. It does not modify production files. The primary campaign remains ablation_results/practical_input/claude_fair_bridge_20261014. The supplement is stored in this folder's campaign/ directory.

The full public case, scientific facts, inventory and required final output are unchanged. Only the final serialization envelope/field list is withheld from FlowPilot's intermediate scientific objective; the exact gas reference, residence-time closure, assumptions and disposition guidance remains. Native specialist output schemas are unchanged. Every primary outcome remains available.

The fixed population is all three chemistries, one additional FlowPilot attempt each. The original three one-shot outputs, packets and nine valid judge replies are reused exactly. No source repair, isolated pump-material patch, score-dependent regeneration or case selection is part of this control. All new calls use the original primary paid_usage.jsonl and its aggregate $10 spending guard.

Execution (root orchestrator owns paid calls)
  python run_followup.py --phase init
  python run_followup.py --phase generate --item 0
  python run_followup.py --phase generate --item 1
  python run_followup.py --phase generate --item 2
  python run_followup.py --phase prepare
  python run_followup.py --phase judge --item qwen
  python run_followup.py --phase judge --item openai
  python run_followup.py --phase judge --item claude
  python pin_baseline_grades.py --verify
  python run_followup.py --phase report

If a fixed judge cell exhausts its format-attempt budget, retain it as incomplete and continue only unattempted preplanned cells using the fixed three-cell sharding option. Do not regenerate designs, retry valid grades, relax evidence rules, or substitute a complete subset for the planned panel. Any N/A-score-only canonicalization must remain a separately declared supplement preserving raw replies and the strict missing-cell counts.

Provenance
- campaign/frozen/plan.json records the full preregistration.
- campaign/frozen/routing.json and objective_diff.txt preserve the exact intervention.
- campaign/frozen/helper_manifest.json freezes driver/tests; source_manifest.json freezes production and benchmark code.
- campaign/frozen/input_manifest.json pins original cases, rubric, historical judge source and reused outputs/packets.
- baseline_grade_manifest.json separately pins all nine original accepted one-shot grades, including raw response, prompt, telemetry, status and parsed response. Its helper never calls a model or changes a grade. Pinning was completed after all original baseline grades became available, before the corrected comparison is finalized.
- prompt_routing_audit.txt explains the observed failure and historical contrast.

Ten offline driver tests passed before freeze. Test coverage includes exact three-case selection/seeds, original one-shot selection, every semantic requirement retained verbatim, unchanged public case, one-attempt generation, no retry of valid grades, and complete equal-panel aggregation. The recorded control tests a harness-format routing hypothesis; it does not establish independent prospective superiority, chemical yield, or a pure effect of prompt length.
