# KHU protocols and responses



Exact extracted text, not rewritten by an LLM. Images are flagged, not OCR-transcribed.

Only explicitly revised sets are selected; older designs are comparison material,

not wet-lab evidence or new constraints. No design run has been performed.



## Slide 1: Figure 5 batch protocol



Step 1: Giese reaction
A re-sealable pressure tube (13 × 100 mm) equipped with a magnetic stir bar was charged with (((4-methoxyphenyl)thio)methyl)trimethylsilane (1, 0.2 mmol, 1.0 equiv), acrylonitrile (2, 0.4 mmol, 2.0 equiv), and [Ir(dF(CF3)ppy)2(dtbpy)]PF6 (0.001 mmol, 0.5 mol %) under an argon atmosphere. A degassed solvent mixture (2.0 mL, 0.1 M with respect to 1) of EtOH:pH 9 buffer (5:1, v/v) was added, and the resulting light greenish-yellow mixture, under vigorous magnetic stirring, was positioned 3 cm from a pair of 5 W blue LEDs (λmax = 452 nm) and irradiated at room temperature (25 °C, maintained with a cooling fan to counteract heating from the LEDs) for 4 h 

Step 2: Oxidation reaction
After 4 h, the cap of the tube was removed to expose the reaction mixture to ambient air, providing atmospheric oxygen as the terminal oxidant, and stirring was continued under otherwise identical irradiation conditions for an additional 6 h.

Green Chem., 2025, 27, 3284–3292

Figure 5: Protocol



## Slide 2: Reference only



[Embedded image: not transcribed; inspect original slide.]

[Embedded image: not transcribed; inspect original slide.]

[Embedded image: not transcribed; inspect original slide.]

[Embedded image: not transcribed; inspect original slide.]

[Embedded image: not transcribed; inspect original slide.]

Figure 5: Questions 



## Slide 3: Reference only



Figure 5: Response set 1  from KHU

Respones to Q-OBJ-001: Please enter here —> Develop a safe, reproducible, and inventory-compatible two-stage flow process that maximizes the overall yield and selectivity to sulfoxide 4a. Stage-specific residence times and operating conditions may be adjusted, but the two stages should be optimized as one integrated process because the output of Stage 1 directly determines the feed composition entering Stage 2.

Respones to Q-HYP-001: Please enter here —> 
- Stage 1 requires effective oxygen exclusion and may be sensitive to photon delivery and residence time. 
- Incomplete Stage 1 conversion will directly reduce the amount of sulfide available for Stage 2 oxidation. 
- Stage 2 requires sufficient oxygen availability, and insufficient oxygen supply may limit oxidation conversion. 
- A practical excess of oxygen may be beneficial because not all supplied gas may be effectively transferred to or utilized in the liquid phase. 
- Oxygen flow, pressure, gas–liquid contact, and overall residence-time distribution should be considered together. 

Respones to Q-GAS-002: Please enter here —> No fixed value is specified by the batch protocol. Please select and justify an initial oxygen feed considering the oxidation requirement, gas utilization, gas–liquid transport, safety, and the available inventory. A practical excess of oxygen may be considered if oxygen availability is expected to limit Stage 2.

Respones to Q-GAS-003: Please enter here —> Introduce oxygen only after completion of the oxygen-free Giese stage, as the reaction stream enters Stage 2. Oxygen should not be introduced during Stage 1. 



## Slide 4: Reference only



Figure 5: Response set 1  from KHU

Parameter | FlowPilot Design

Flowrate Pump (Stream A) 
(mL/min) | 0.03129 

Gas Flowrate [at the inlet]
(mL/min) | 0.140259

BPR
(bar) | 2

Reactor Volume [stage 1 ]
(mL) | 5 mL – Manual Reactor PFA

Reactor Volume [stage 2 ]
(mL) | 10 mL – Manual Reactor FEP

Residence Time [Stage 1]
(min) | 159.796

Residence Time [Stage 2]
(min) | 58.2924

[Embedded image: not transcribed; inspect original slide.]

Models Used: 
Upstream: Claude-Opus-4.6
Downstream: Claude-Sonnet-4.6 



## Slide 5: Figure 5 revised set 1



Figure 5: Response set 1  from KHU (Integrated performance, Revised)

Respones to Q-OBJ-001: Please enter here —> Develop a safe, reproducible, and inventory-compatible two-stage flow process that maximizes the final yield of sulfoxide 4a while avoiding unnecessarily long residence times. Design the two stages as one connected process, because Stage 1 determines the composition entering Stage 2.

Respones to Q-HYP-001: Please enter here —> Stage 1 performance may depend on effective oxygen exclusion and photon delivery. Incomplete sulfide formation will limit the material available for subsequent oxidation. Stage 2 may depend on oxygen availability and gas–liquid transport. Use the batch reaction time as a reference rather than a fixed target, and determine the residence time based on flow performance. Balance photon delivery, oxygen feed, pressure, and stage-specific residence times based on the final sulfoxide yield and selectivity.

Respones to Q-GAS-002: Please enter here —> Use at least 2.0 equiv of oxygen relative to the limiting substrate; higher oxygen equivalents may be used if justified by the reaction and operating conditions.

Respones to Q-GAS-003: Please enter here —> Introduce oxygen only at the Stage 2 inlet. Stage 1 must remain oxygen-free.



## Slide 6: Reference only



Figure 5: Response set 2  from KHU

Respones to Q-OBJ-001: Please enter here —> Prioritize high conversion across the complete two-stage sequence and maximize the final yield of sulfoxide 4a. For the initial design, sufficient residence time should be allowed so that both stages can proceed to high conversion.

Respones to Q-HYP-001: Please enter here —> 
- Stage 1 should receive sufficient photon exposure and residence time to generate a high concentration of sulfide before the stream enters Stage 2. 
- Stage 2 may be particularly sensitive to oxygen availability. 
- A moderate excess of oxygen may improve oxidation if gas transfer or utilization is incomplete. 
- Increasing oxygen flow should not automatically be assumed to improve conversion because pressure, flow pattern, and residence time may change at the same time. 
- Conditions should be selected based on the final sulfoxide yield of the complete two-stage sequence. 

Respones to Q-GAS-002: Please enter here —> Use 2.0 equiv of oxygen relative to the limiting substrate as a moderate-excess starting point, while allowing pressure and residence-time conditions to be adjusted accordingly.

Respones to Q-GAS-003: Please enter here —> Introduce oxygen at Stage 2 only, after the oxygen-free Giese stage is complete. 



## Slide 7: Reference only



Figure 5: Response set 2  from KHU

20260907_135930_webapp

Parameter | FlowPilot Design

Flowrate Pump (Stream A) 
(mL/min) | 0.04182 

Gas Flowrate [at the inlet]
(mL/min) | 0.18746

BPR
(bar) | 2

Reactor Volume [stage 1 ]
(mL) | 10 mL – Manual Reactor FEP

Reactor Volume [stage 2 ]
(mL) | 20 mL – Manual Reactor FEP

Residence Time [Stage 1]
(min) | 239.12

Residence Time [Stage 2]
(min) | 87.2296

[Embedded image: not transcribed; inspect original slide.]

Models Used: 
Upstream: Claude-Opus-4.6
Downstream: Claude-Sonnet-4.6 



## Slide 8: Figure 5 revised set 2



Figure 5: Response set 2  from KHU (Conversion-focused, Revised)

Respones to Q-OBJ-001: Please enter here Prioritize high conversion through both stages and a high final yield of sulfoxide 4a. Choose the shortest residence times reasonably justified for this conversion-focused starting design rather than adding a large time excess by default. Keep reactor volumes and flow rates practical for flow screening.

Respones to Q-HYP-001: Please enter here —> Stage 1 requires sufficient photon delivery and residence time to generate enough sulfide for Stage 2. Use the batch reaction time as a reference rather than a fixed target, and determine the residence time based on flow performance. Stage 2 requires adequate oxygen delivery, but oxygen feed, pressure, gas fraction, and residence time are coupled. Residence time should be optimized based on the final sulfoxide yield and selectivity rather than increased by default.

Respones to Q-GAS-002: Please enter here —> Use at least 2.0 equiv of oxygen relative to the limiting substrate; higher oxygen equivalents may be used if justified by the reaction and operating conditions.

Respones to Q-GAS-003: Please enter here —> Introduce oxygen only at the Stage 2 inlet. Stage 1 must remain oxygen-free.



## Slide 9: Reference only



Figure 5: Response set 3  from KHU

Respones to Q-OBJ-001: Please enter here —> Identify a practical two-stage flow design that gives the best overall balance among conversion, residence time, oxygen delivery, and process stability. Different operating conditions may be used for the two stages, but the performance of the complete connected process should determine the final design.

Respones to Q-HYP-001: Please enter here —> 
- Stage 1 performance may depend on photon delivery and residence time. 
- Stage 2 performance may depend on oxygen availability and gas–liquid transport. 
- Excess oxygen may be useful, but its benefit is expected to depend on pressure, gas utilization, and residence time. 
- Pressure may increase oxygen dissolution while also changing the gas–liquid flow pattern. 
- FlowPilot should determine the most suitable combination of Stage 1 conversion, oxygen feed, pressure, and Stage 2 residence time for the overall process.

Respones to Q-GAS-002: Please enter here —> No fixed value. Please determine and justify an initial oxygen feed based on the reaction requirements and the available operating range.

Respones to Q-GAS-003: Please enter here —> Oxygen should first be introduced as the Stage 1 effluent enters the oxidation stage.



## Slide 10: Reference only



Figure 5: Response set 3  from KHU

20260907_142711_webapp

Parameter | FlowPilot Design

Flowrate Pump (Stream A) 
(mL/min) | 0.01246 

Gas Flowrate [at the inlet]
(mL/min) | 0.08378

BPR
(bar) | 2

Reactor Volume [stage 1 ]
(mL) | 5 mL – Manual Reactor PFA

Reactor Volume [stage 2 ]
(mL) | 10 mL – Manual Reactor FEP

Residence Time [Stage 1]
(min) | 52

Residence Time [Stage 2]
(min) | 104

[Embedded image: not transcribed; inspect original slide.]

Models Used: 
Upstream: Claude-Opus-4.6
Downstream: Claude-Sonnet-4.6 



## Slide 11: Figure 5 revised set 3



Figure 5: Response set 3  from KHU (Throughput / processing-time focused, Revised)

Respones to Q-OBJ-001: Please enter here —> Identify a practical two-stage flow design that balances final sulfoxide yield and selectivity with substrate throughput, total processing time, oxygen delivery, and stable operation. Prefer a compact, higher-throughput starting design where chemically justified.

Respones to Q-HYP-001: Please enter here —> Stage 1 may maintain sufficient sulfide formation at higher liquid throughput if photon delivery remains effective. Use the batch reaction time as a reference rather than a fixed target, and determine the residence time based on flow performance. In Stage 2, pressure and oxygen feed can alter oxygen availability, gas fraction, and contact time simultaneously. Evaluate the overall process based on substrate throughput, stage-specific residence times, and the final sulfoxide yield and selectivity. Shorter residence times may be explored, but should be validated experimentally.

Respones to Q-GAS-002: Please enter here —> Use at least 2.0 equiv of oxygen relative to the limiting substrate; higher oxygen equivalents may be used if justified by the reaction and operating conditions.

Respones to Q-GAS-003: Please enter here —> Introduce oxygen only at the Stage 2 inlet. Stage 1 must remain oxygen-free..



## Slide 12: Figure 6 batch protocol



 A 4.0 mL vial equipped with a PTFE stir bar was charged with 3-methyl-4-nitrobenzoic acid (1 equiv, 0.5 mmol), N,N-dimethylpyridin-4-amine (DMAP) (6.1 mg, 0.1 equiv, 0.05 mmol), and DPDTC (130. mg, 1.05 equiv, 0.525 mmol) under ambient atmosphere without argon or nitrogen purging.  2-MeTHF (1.0 mL, 0.5 M) was added, and the vial was capped and sealed with Teflon tape. The reaction vial was placed in a pre-heated oil bath maintained at 95 °C and stirred at 500–600 rpm for 30 min. The vial was removed from the oil bath and allowed to cool at room temperature for 5–10 min. Once cooled, the cap was removed and benzylamine (1.05 equiv, 0.525 mmol) and 2-MeTHF (0.25 mL, 0.4 M) were quickly added. The vial was capped, sealed with Teflon tape, and returned to the oil bath at 95 °C, and stirred at 500–600 rpm for another 30 min before workup. 

ACS Sustainable Chem. Eng. 2025, 13, 6646−6655

Figure 6: Protocol



## Slide 13: Reference only



Figure 6: Questions

[Embedded image: not transcribed; inspect original slide.]

[Embedded image: not transcribed; inspect original slide.]

[Embedded image: not transcribed; inspect original slide.]

[Embedded image: not transcribed; inspect original slide.]



## Slide 14: Reference only



Figure 6: Response set 1 from KHU

Respones to Q-OBJ-001: Please enter here —> Develop a safe, reproducible, and inventory-compatible two-stage telescoped flow process that gives high amide yield without isolating the thioester intermediate. The complete sequence should be designed and evaluated as one connected process, with suitable residence times and operating conditions determined from the overall process performance.

Respones to Q-HYP-001: Please enter here —>
- The extent of thioester formation in Stage 1 will directly affect Stage 2 amidation and the final amide yield. 
- Because the reaction temperature is above the normal boiling point of 2-MeTHF, sufficient pressure may be required to maintain a stable liquid-phase flow at 95 oC. 
- The brief cooling step before benzylamine addition may be a batch-handling step rather than a chemical requirement, so its necessity in continuous flow should be evaluated. 
- FlowPilot should determine suitable residence times and interstage conditions based on the performance of the complete connected process. 


Respones to Q-CHEM-001: Please enter here —> This is a two-stage DPDTC-mediated amide formation through a thioester intermediate. In Stage 1, 3-methyl-4-nitrobenzoic acid is converted to the corresponding 2-pyridyl thioester using DPDTC and DMAP. The intermediate is not isolated. In Stage 2, benzylamine reacts with the Stage 1 stream to form N-benzyl-3-methyl-4-nitrobenzamide. The amide C(O)–N bond is formed, and the thioester C(O)–S bond is cleaved during Stage 2.



## Slide 15: Reference only



Figure 6: Response set 1 from KHU

20260907_164042_webapp

[Embedded image: not transcribed; inspect original slide.]

Parameter | FlowPilot Design

Flowrate Pump (Stream A) 
(mL/min) | 0.149071 

Flowrate Pump (Stream B) 
(mL/min) | 0.037268

BPR
(bar) | 2

Reactor Volume [stage 1 ]
(mL) | 5 mL – Manual Reactor PFA

Reactor Volume [stage 2 ]
(mL) | 5 mL – Manual Reactor PFA

Residence Time [Stage 1]
(min) | 33.5411

Residence Time [Stage 2]
(min) | 26.8328

Models Used: 
Upstream: Claude-Opus-4.6
Downstream: Claude-Sonnet-4.6 



## Slide 16: Figure 6 revised set 1



Figure 6: Response set 1 from KHU (Integrated performance, Revised)

Respones to Q-OBJ-001: Please enter here —> Develop a safe, reproducible, and inventory-compatible two-stage telescoped flow process that maximizes the final amide yield while avoiding unnecessarily long residence times. Design the two stages as one connected process, because Stage 1 determines the composition entering Stage 2.

Respones to Q-CHEM-001: Please enter here —> This is a two-stage DPDTC-mediated amide formation through a thioester intermediate. In Stage 1, 3-methyl-4-nitrobenzoic acid is converted to the corresponding 2-pyridyl thioester using DPDTC and DMAP, without isolating the intermediate. In Stage 2, benzylamine is added to the Stage 1 effluent to form N-benzyl-3-methyl-4-nitrobenzamide. The amide C(O)–N bond is formed and the thioester C(O)–S bond is cleaved during Stage 2.

Respones to Q-HYP-001: Please enter here —> Stage 1 conversion will directly affect the amount of thioester available for Stage 2 amidation. The cooling step before benzylamine addition may be a batch-handling step rather than an essential reaction step and should be evaluated in flow. Use the batch temperature and reaction times as reference conditions rather than fixed targets, and determine the operating conditions based on flow performance. The Stream A/Stream B molar ratio should match the intended stoichiometry, taking both concentration and flow rate into account.



## Slide 17: Reference only



Figure 6: Response set 2 from KHU

Respones to Q-OBJ-001: Please enter here —> Maximize the final amide yield across the complete two-stage sequence. For the initial design, sufficient reaction time should be provided in both stages to achieve high conversion and stable operation.

Respones to Q-HYP-001: Please enter here —>
- High Stage 1 conversion is important because incomplete thioester formation will directly limit the amount of intermediate available for amidation. 
- Stage 2 should provide sufficient reaction time for conversion of the thioester intermediate to the amide product. 
- The overall design should be judged by the final amide yield rather than by the performance of either stage alone. 
- The need for cooling before benzylamine addition should be evaluated rather than assumed. 


Respones to Q-CHEM-001: Please enter here —> This is a two-stage DPDTC-mediated amide formation through a thioester intermediate. In Stage 1, 3-methyl-4-nitrobenzoic acid is converted to the corresponding 2-pyridyl thioester using DPDTC and DMAP. The intermediate is not isolated. In Stage 2, benzylamine reacts with the Stage 1 stream to form N-benzyl-3-methyl-4-nitrobenzamide. The amide C(O)–N bond is formed, and the thioester C(O)–S bond is cleaved during Stage 2.



## Slide 18: Reference only



Figure 6: Response set 2 from KHU

[Embedded image: not transcribed; inspect original slide.]

Parameter | FlowPilot Design

Flowrate Pump (Stream A) 
(mL/min) | 0.066667 

Flowrate Pump (Stream B) 
(mL/min) | 0.016667

BPR
(bar) | 2

Reactor Volume [stage 1 ]
(mL) | 2 mL – Manual Reactor PFA

Reactor Volume [stage 2 ]
(mL) | 5 mL – Manual Reactor PFA

Residence Time [Stage 1]
(min) | 30

Residence Time [Stage 2]
(min) | 60

Models Used: 
Upstream: Claude-Opus-4.6
Downstream: Claude-Sonnet-4.6 



## Slide 19: Figure 6 revised set 2



Figure 6: Response set 2 from KHU (Conversion-focused, Revised)

Respones to Q-OBJ-001: Please enter here —> Prioritize high conversion through both stages and a high final amide yield. Choose the shortest residence times reasonably justified for this conversion-focused starting design rather than adding a large time excess by default. Keep reactor volumes and flow rates practical for flow screening.

Respones to Q-CHEM-001: Please enter here —> This is a two-stage DPDTC-mediated amide formation through a thioester intermediate. In Stage 1, 3-methyl-4-nitrobenzoic acid is converted to the corresponding 2-pyridyl thioester using DPDTC and DMAP, without isolating the intermediate. In Stage 2, benzylamine is added to the Stage 1 effluent to form N-benzyl-3-methyl-4-nitrobenzamide. The amide C(O)–N bond is formed and the thioester C(O)–S bond is cleaved during Stage 2.

Respones to Q-HYP-001: Please enter here —> Stage 1 requires sufficient conversion to generate enough thioester for Stage 2. Stage 2 requires sufficient reaction time for conversion to the final amide. Use the batch temperature and reaction times as reference conditions rather than fixed targets, and determine the operating conditions based on flow performance. Evaluate the overall process based on the final amide yield rather than either stage alone. The Stream A/Stream B molar ratio should reflect the intended stoichiometry based on both concentration and flow rate.



## Slide 20: Reference only



Figure 6: Response set 3 from KHU 

Respones to Q-OBJ-001: Please enter here —> Identify a practical two-stage flow process that reduces the overall processing time while maintaining high final amide yield and stable continuous operation. FlowPilot should explore the residence time and interstage configuration without assuming that the batch timing must be retained.

Respones to Q-HYP-001: Please enter here —>
- Continuous heating and mixing in flow may allow the overall processing time to be shortened relative to batch, but the required residence time should be determined from the final process performance. 
- Any change in Stage 1 conversion will affect the composition of the stream entering Stage 2 and therefore the final amide yield. 
- The batch cooling step may not be necessary in continuous flow. Direct transfer from Stage 1 to benzylamine addition should be evaluated if chemically and operationally acceptable. 
- If cooling is required, a short controlled cooling section may be considered. 
- Temperature and pressure should be selected to maintain a stable liquid-phase process at the reaction temperature. 


Respones to Q-CHEM-001: Please enter here —> This is a two-stage DPDTC-mediated amide formation through a thioester intermediate. In Stage 1, 3-methyl-4-nitrobenzoic acid is converted to the corresponding 2-pyridyl thioester using DPDTC and DMAP. The intermediate is not isolated. In Stage 2, benzylamine reacts with the Stage 1 stream to form N-benzyl-3-methyl-4-nitrobenzamide. The amide C(O)–N bond is formed, and the thioester C(O)–S bond is cleaved during Stage 2.



## Slide 21: Reference only



Figure 6: Response set 3 from KHU

[Embedded image: not transcribed; inspect original slide.]

Parameter | FlowPilot Design

Flowrate Pump (Stream A) 
(mL/min) | 0.593326 

Flowrate Pump (Stream B) 
(mL/min) | 0.155748

BPR
(bar) | 2

Reactor Volume [stage 1 ]
(mL) | 10 mL – Manual Reactor PFA

Reactor Volume [stage 2 ]
(mL) | 5 mL – Manual Reactor PFA

Residence Time [Stage 1]
(min) | 15.8541

Residence Time [Stage 2]
(min) | 6.6749

Models Used: 
Upstream: Claude-Opus-4.6
Downstream: Claude-Sonnet-4.6 



## Slide 22: Figure 6 revised set 3



Figure 6: Response set 3 from KHU (Throughput / processing-time focused, Revised)

Respones to Q-OBJ-001: Please enter here —> Identify a practical two-stage flow design that balances final amide yield with substrate throughput, total processing time, and stable operation. Prefer a compact, higher-throughput starting design where chemically justified.

Respones to Q-CHEM-001: Please enter here —> This is a two-stage DPDTC-mediated amide formation through a thioester intermediate. In Stage 1, 3-methyl-4-nitrobenzoic acid is converted to the corresponding 2-pyridyl thioester using DPDTC and DMAP, without isolating the intermediate. In Stage 2, benzylamine is added to the Stage 1 effluent to form N-benzyl-3-methyl-4-nitrobenzamide. The amide C(O)–N bond is formed and the thioester C(O)–S bond is cleaved during Stage 2.

Respones to Q-HYP-001: Please enter here —> Stage 1 must maintain sufficient thioester formation as throughput is increased. Direct transfer from Stage 1 to benzylamine addition may avoid the batch cooling step if chemically and operationally acceptable. Use the batch temperature and reaction times as reference conditions rather than fixed targets. Evaluate the overall process based on substrate throughput, stage-specific residence times, and the final amide yield. Shorter residence times may be explored, but should be validated experimentally. The Stream A/Stream B molar ratio should match the intended stoichiometry based on both concentration and flow rate.

