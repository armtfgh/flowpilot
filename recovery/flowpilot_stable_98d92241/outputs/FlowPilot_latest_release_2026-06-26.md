**FlowPilot Agent Latest Release Notes**  
Release date: 2026-06-26  
   
 Release type: Flow-design accuracy, multiphase timing, and closed-loop refinement update  
   
 Primary target: Batch chemistry protocol to flow chemistry protocol translation  
**Executive Summary**  
This release upgrades FlowPilot from a one-shot batch-to-flow translator into a more engineering-aware design assistant. The main changes are:  
1. Heat-transfer calculations are now attached to generated flow proposals.  
2. Multi-step reaction timing is handled stage by stage instead of using one ambiguous global residence time.  
3. Gas-liquid additions now distinguish between inlet/STP gas settings and actual in-channel reactor flow.  
4. An incremental experimental feedback loop can ingest previous experimental results and generate a refined next design.  
5. The GUI now exposes the experiment loop and evidence-calibrated design ladder.  
6. The LLM translation path is more robust against malformed or overly long JSON outputs.  
The most important behavioral change is that experimental evidence can now override optimistic first-pass intensification. For the tetrahydroquinoline aerobic photo-oxidation case, this prevents the pipeline from keeping unrealistically short retention times after low-conversion KRICT results.  
**1. Heat-Transfer Calculations**  
**What Was Added**  
The deterministic design calculator now computes heat-transfer metrics for the proposed reactor and attaches them to the final proposal as heat_transfer_metrics.  
Implementation references:  
- flora_translate/design_calculator.py  
- DesignCalculations heat fields around lines 288-296  
- annotate_proposal_with_calculations() around lines 664-672  
- heat-transfer calculation block around lines 1938-2043  
- flora_translate/output_formatter.py around lines 114-120  
**What The Calculations Are**  
The heat-transfer module estimates whether the proposed flow reactor can remove reaction heat fast enough.  
Calculated quantities:  
- heat_generation_W: estimated reaction heat generation rate.  
- heat_removal_W: estimated reactor heat removal capacity.  
- thermal_damkohler: heat generation divided by heat removal, Da_th = Q_gen / Q_rem.  
- thermal_safe: Boolean safety flag based on Da_th < 1.  
- surface_to_volume: reactor surface-to-volume ratio, S/V = 4/d.  
- heat_transfer_area_m2: wall area available for heat removal.  
- UA_W_K: overall conductance, U * A.  
- heat_transfer_score: normalized heat-transfer margin score.  
Main equations:  
Q_gen = |DeltaH_r| * r * V_R  
 A_wall = pi * d * L  
 Q_rem = U * A_wall * DeltaT_lm  
 Da_th = Q_gen / Q_rem  
 S/V = 4 / d  
   
Current default assumptions:  
- DeltaH_r is estimated by chemistry class:  
  - photoredox/photochem: about -50 kJ/mol  
  - thermal: about -80 kJ/mol  
  - oxidation: about -100 kJ/mol  
  - hydrogenation: about -120 kJ/mol  
  - default: about -60 kJ/mol  
- Available heat-transfer coefficient constants:  
  - coil: 300 W/m2/K  
  - chip: 500 W/m2/K  
  - packed bed: 200 W/m2/K  
  - CSTR: 150 W/m2/K  
- The current calculator path uses the coil default for the heat-transfer step.  
- The log-mean temperature difference is estimated as 10 deg C.  
**Why It Matters**  
Previously, FlowPilot could recommend a residence time and reactor volume without explicitly reporting whether heat removal was plausible. Now each design reports a thermal risk signal. This is useful for exothermic oxidations, reductions, hydrogenations, and thermal scale-up cases.  
The output formatter includes the authoritative heat-transfer values in the final explanation:  
Heat transfer: UA=..., Da_th=..., score=...  
   
**2. Reaction-Time Calculation For Multi-Step Reactions**  
**What Was Added**  
The topology builder now detects multi-stage chemistry plans and builds a separate reactor zone for each stage.  
Implementation references:  
- flora_translate/main.py  
- _build_translate_topology() dispatch around lines 183-198  
- _build_multistep_topology() around lines 585-820  
- flora_translate/tests/test_multistage_topology.py  
**Behavior**  
If a ChemistryPlan contains more than one stage, FlowPilot uses _build_multistep_topology() instead of the single-step topology.  
The multi-step topology applies these rules:  
- Each chemistry stage gets its own reactor zone.  
- New feed additions are introduced at the correct stage.  
- Gas feeds are treated as MFCs instead of liquid pumps.  
- Quench/workup streams are not treated as reaction stages.  
- Each stage has its own inlet flow, residence time, tube diameter, and reactor volume.  
- Stage reactor volume is calculated from the local stage inlet flow:  
V_R,i = tau_i * Q_inlet,i  
   
**Residence-Time Allocation**  
The key correction is how global residence time is handled.  
If all stage residence times are explicitly provided in proposal.stage_parameters, those values are used. If only a global residence time is provided, FlowPilot treats that value as the total process residence time and allocates it across stages.  
Allocation method:  
- Use stage batch-time weights when available.  
- Use equal allocation when batch-time weights are not available.  
This prevents an unreviewed later stage from silently falling back to old batch/IF timing after the council has selected a global flow design.  
Validated behavior:  
- test_multistage_topology_allocates_global_council_tau_and_keeps_air_as_mfc() confirms that a two-stage photoredox/aerobic oxidation plan splits global tau into stage-specific tau values and keeps air as an MFC feed.  
**3. Gas-Liquid Addition And Residence-Time Basis**  
**What Was Added**  
The calculator and proposal annotation now distinguish:  
- liquid/substrate flow rate,  
- gas MFC setpoint at STP,  
- actual gas volume flow inside the pressurized reactor,  
- in-channel total actual residence time,  
- inlet/STP-equivalent residence time.  
Implementation references:  
- flora_translate/design_calculator.py  
- gas-liquid fields around lines 259-282  
- gas context calculation around lines 918-958  
- gas-liquid reactor sizing around lines 1638-1715  
- O2 transfer values around lines 1878-1910  
- proposal annotation around lines 620-643 and 644-663  
- flora_translate/experiment_loop.py fields around lines 25-42 and gas-summary helpers around lines 1005-1021  
- flora_translate/tests/test_multiphase_design.py  
**Behavior**  
For gas-liquid reactions, FlowPilot computes:  
- gas_flow_sccm: MFC setpoint at STP/inlet basis.  
- gas_flow_actual_mL_min: actual gas volumetric flow at reactor temperature and pressure.  
- gas_liquid_ratio: actual reactor gas/liquid volumetric ratio.  
- gas_holdup: estimated gas volume fraction.  
- liquid_holdup_volume_mL: liquid residence volume.  
- two_phase_multiplier: empirical pressure-drop multiplier for gas-liquid flow.  
- two_phase_pressure_drop_bar: pressure drop adjusted for gas-liquid flow.  
- o2_supply_mmol_min: O2 supplied by gas feed.  
- o2_required_mmol_min: stoichiometric O2 requirement.  
- o2_equiv_supplied: O2 equivalents supplied.  
- dissolved_o2_mM: approximate dissolved O2.  
- kLa_s: mass-transfer coefficient estimate.  
- o2_transfer_capacity_mmol_min: estimated transfer capacity.  
- o2_transfer_sufficiency: transfer capacity divided by O2 demand.  
The gas correction uses ideal-gas scaling from STP/inlet setpoint to actual reactor flow:  
Q_g,actual = Q_g,STP * (T_reactor / T_STP) * (P_STP / P_reactor)  
   
Gas holdup is estimated from the actual in-channel gas/liquid ratio:  
epsilon_g = Q_g / (Q_g + Q_L)  
   
The total reactor volume for a two-phase residence time is then corrected as:  
V_R = V_L / (1 - epsilon_g)  
   
**Why It Matters**  
This directly addresses the KHU/KRICT feedback. In gas-liquid flow, the time calculated from in-channel actual flow can be very different from the time calculated from inlet/STP flow settings. The agent now carries both bases so the user can compare:  
- what the reactor actually experiences,  
- what the MFC and pumps are set to at the inlet.  
The closed-loop package also preserves both:  
- residence_time_in_channel_min  
- residence_time_inlet_min  
- residence_time_basis  
**4. Incremental Experimental Feedback Agent**  
**What Was Added**  
A deterministic closed-loop refinement module was added. It accepts experimental results from one or more previous cycles, diagnoses why the design failed or succeeded, and creates the next design version.  
Implementation references:  
- flora_translate/experiment_loop.py  
- data models around lines 25-99  
- refine_from_experimental_campaign() around lines 120-145  
- extract_experiments_from_text() around lines 148-203  
- calibrate_experimental_campaign() around lines 206-260  
- campaign application helpers around lines 694-733  
- next-experiment package around lines 842-873  
- gas/liquid scaling helpers around lines 958-1021  
- automatic prompt hook in flora_translate/main.py around lines 1178-1205  
- GUI tab in pages/flora_design_unified.py around lines 235-244 and 390-475  
**Supported Inputs**  
Each experimental cycle can include actual run conditions:  
- residence time,  
- in-channel residence time,  
- inlet/STP residence time,  
- flow rate,  
- substrate flow,  
- gas in-channel flow,  
- gas STP/MFC flow,  
- gas equivalents,  
- temperature,  
- concentration,  
- tubing ID,  
- reactor volume,  
- BPR,  
- wavelength,  
- light power.  
Each cycle can include outcomes:  
- yield,  
- product percentage,  
- starting material percentage,  
- conversion,  
- selectivity,  
- pressure,  
- pressure drift,  
- clogging,  
- precipitation,  
- gas-liquid stability,  
- impurity notes.  
**Diagnosis Logic**  
The refinement module identifies common failure modes:  
- kinetic underconversion,  
- low yield,  
- selectivity loss,  
- solubility or fouling,  
- pressure instability,  
- gas-liquid instability,  
- experiment-calibrated kinetics.  
Depending on the diagnosis, it can revise:  
- residence time,  
- flow rate,  
- gas flow,  
- temperature,  
- concentration,  
- BPR,  
- tubing ID,  
- pre-reactor filtering,  
- safety flags,  
- next-experiment acceptance criteria.  
**Campaign Calibration**  
When multiple experiments are available, the module builds an evidence-calibrated design ladder.  
It uses the best observed response as an anchor and fits a simple apparent first-order response model versus in-channel residence time. The result includes:  
- best observed anchor,  
- next intermediate screen,  
- target estimate,  
- recommended in-channel residence time,  
- corresponding liquid and gas flow rates,  
- inlet/STP-equivalent residence time.  
This is deliberately conservative. It avoids letting one poor early run dominate the next design, but also prevents the LLM/council from ignoring measured low conversion.  
**Automatic Prompt Extraction**  
The main pipeline now scans the original user prompt for Entry N experimental blocks. If at least two usable entries are found, FlowPilot automatically applies evidence-calibrated closed-loop refinement before building the final topology.  
This is the new Step 6b:  
Step 6b: Applying evidence-calibrated closed-loop refinement from N experiments  
   
**5. GUI Updates**  
**What Was Added**  
The unified Streamlit GUI now has an Experiment Loop tab.  
Implementation references:  
- pages/flora_design_unified.py  
- tab list around lines 235-244  
- _render_experiment_loop() around lines 390-475  
- evidence ladder display around lines 657-674  
**GUI Capabilities**  
The GUI can now:  
- display the current design version,  
- collect actual lab conditions,  
- collect measured outcomes,  
- run the closed-loop refinement,  
- show parameter changes,  
- show the next experiment package,  
- show evidence-calibrated design ladders,  
- download the closed-loop campaign JSON.  
This makes the feature usable as an iterative design cycle rather than only as a script/API feature.  
**6. Other Recent Additions**  
**Model And Provider Configuration**  
The model aliases were updated:  
- upstream/default Claude Sonnet alias: claude-sonnet-4-6  
- Opus alias: claude-opus-4-6  
- downstream/council engine provider: openai  
- downstream OpenAI model: gpt-4o  
Implementation reference:  
- flora_translate/config.py around lines 20-50 and 129-133  
**More Robust Translation JSON Handling**  
The translation LLM now:  
- asks for compact JSON,  
- retries with stricter JSON-only instructions,  
- uses a repair call if the first outputs are malformed,  
- normalizes reactor volume, flow rate, and residence time consistency.  
Implementation reference:  
- flora_translate/translation_llm.py around lines 56-134  
