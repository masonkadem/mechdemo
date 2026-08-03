# Feature reference

Spearman correlations with blood pressure on the VitalDB held-out split (57,600 segments, 144 subjects).

**Pooled vs within-subject.** Pooled correlations are dominated by between-subject differences: anything tracking age or body size correlates with BP without saying anything about a given person's pressure changing. The within-subject column is what a device needs. Where the two disagree in sign, the pooled value is the misleading one.

| feature | equation | plain meaning | r SBP (pooled / within) | r DBP (pooled / within) |
|---|---|---|---|---|
| `ppg_p10` | -- | 10th-percentile pulse height (how low the trough sits) | -0.287 / -0.353 | -0.296 / -0.369 |
| `rr_pnn50` | -- | proportion of large beat-interval changes | -0.127 / -0.095 | -0.255 / -0.106 |
| `ppg_p90` | -- | 90th-percentile pulse height | -0.274 / -0.377 | -0.245 / -0.367 |
| `decay_slope` | -- | how fast pressure falls after the peak | -0.271 / -0.209 | -0.238 / -0.231 |
| `t_e` | -- | timing of the dicrotic notch | -0.179 / -0.146 | -0.236 / -0.143 |
| `ppg_skew_g` | -- | pulse-shape asymmetry (lopsidedness of the wave) | -0.273 / -0.442 | -0.229 / -0.430 |
| `aix` | (S_peak - D_peak) / S_peak x 100% | augmentation index (how much reflected wave adds to the peak) | +0.182 / +0.150 | +0.225 / +0.180 |
| `reflect_idx` | D_peak / S_peak x 100% | strength of the reflected wave | +0.182 / +0.150 | +0.225 / +0.180 |
| `notch_depth` | depth of the dicrotic notch | depth of the dicrotic notch | +0.219 / +0.181 | +0.220 / +0.206 |
| `pat_peak` | -- | pulse arrival time to the pulse peak | -0.086 / -0.050 | -0.220 / -0.024 |
| `hr` | 60 / RR interval | heart rate | -0.126 / -0.026 | -0.214 / -0.016 |
| `rr_mean` | -- | average beat-to-beat interval | +0.135 / -0.024 | +0.214 / -0.007 |
| `pat_foot` | t(PPG foot) - t(R peak) | pulse arrival time to the pulse foot | -0.064 / -0.048 | -0.207 / -0.052 |
| `qrs_width` | -- | width of the ECG QRS complex | -0.084 / -0.076 | -0.205 / -0.088 |
| `r_count` | -- | number of heartbeats in the window | -0.135 / +0.013 | -0.204 / +0.019 |
| `rise` | upstroke duration | time from foot to peak (upstroke speed) | -0.080 / +0.042 | -0.197 / -0.002 |
| `xcorr_lag` | -- | ECG-PPG cross-correlation lag | -0.048 / -0.053 | -0.196 / -0.055 |
| `t_d` | -- | timing of the late reflection | -0.189 / -0.184 | -0.191 / -0.246 |
| `rr_rmssd` | -- | short-term beat-interval variability | -0.066 / -0.091 | -0.185 / -0.119 |
| `ppg_skew` | -- | pulse-shape asymmetry | -0.200 / -0.417 | -0.185 / -0.416 |
| `rr_cv` | -- | relative beat-interval variability | -0.061 / -0.069 | -0.183 / -0.065 |
| `age` | -- | age | +0.082 / +nan | -0.176 / +nan |
| `qrs_amp_std` | -- | variability of ECG R-wave height | -0.083 / -0.062 | -0.172 / -0.100 |
| `pow_hf` | -- | high-frequency power (respiratory band) | +0.078 / +0.123 | +0.171 / +0.088 |
| `ppg_p25` | -- | 25th-percentile pulse height | -0.243 / -0.307 | -0.167 / -0.353 |
| `sys_area` | -- | area under the systolic part of the pulse | -0.182 / -0.100 | -0.165 / -0.068 |
| `dia_area` | -- | area under the diastolic part | +0.182 / +0.100 | +0.165 / +0.068 |
| `sys_dia_ratio` | -- | systolic to diastolic area ratio | -0.181 / -0.100 | -0.165 / -0.064 |
| `dw25` | -- | width a quarter up the downstroke | +0.237 / +0.155 | +0.160 / +0.142 |
| `t_c` | -- | timing of the reflected-wave shoulder | -0.165 / -0.258 | -0.159 / -0.237 |
| `vpg_ms_area` | -- | area under the rise-rate curve | -0.004 / +0.011 | +0.147 / +0.050 |
| `ppg_kurt_g` | -- | pulse peakedness (sharp vs rounded) | -0.222 / -0.424 | -0.147 / -0.366 |
| `rr_sdnn` | -- | beat-to-beat interval variability | -0.038 / -0.057 | -0.147 / -0.071 |
| `t_vpg_max` | -- | time of the steepest rise | -0.108 / -0.009 | -0.146 / +0.039 |
| `qrs_amp_mean` | -- | average ECG R-wave height | +0.059 / -0.015 | +0.146 / -0.007 |
| `sw10` | -- | width near the base of the upstroke | -0.018 / +0.096 | -0.132 / +0.065 |
| `t_b` | -- | timing of the early rebound in wave curvature | -0.135 / -0.104 | -0.132 / -0.110 |
| `vpg_max` | -- | steepest rise rate of the pulse | -0.005 / -0.151 | +0.125 / -0.112 |
| `ppg_kurt` | -- | pulse peakedness | -0.165 / -0.431 | -0.124 / -0.399 |
| `peak_std` | -- | beat-to-beat peak-height variability | +0.014 / +0.030 | +0.119 / +0.028 |

## Theoretical route to blood pressure

| feature | why it should relate to BP |
|---|---|
| `aix` | stiffer arteries return the reflected wave sooner, so it merges with the systolic peak and augments it; AIx rises with age and vascular disease |
| `reflect_idx` | size of the reflected wave relative to the forward wave; a vascular tone and stiffness indicator |
| `notch_depth` | notch flattens as peripheral resistance and stiffness rise |
| `hr` | rate covaries with BP through autonomic drive, not through the arterial law; the audit identified it as the dominant shortcut |
| `pat_foot` | PAT = PEP + PTT. Moens-Korteweg predicts higher BP stiffens the artery and shortens transit, but PEP is cardiac and contaminates the interval |
| `rise` | faster upstroke with stiffer vessels |
| `age` | arteries stiffen with age; the strongest single predictor here |
| `hfd` | waveform complexity; no direct arterial-law route, included as a shape summary |
| `crest` | upstroke duration, set by how fast the ejected volume distends the vessel; shortens with arteriosclerosis |
| `apg_c_a` | falls with arterial stiffness |
| `bmi` | body composition covaries with pressure through several routes |
| `notch_time` | notch timing tracks aortic valve closure and wave reflection |
| `takazawa` | composite vascular ageing index |
| `apg_b_a` | b/a rises with arterial stiffness |
| `apg_d_a` | falls with arterial stiffness |
| `apg_e_a` | falls with arterial stiffness |
| `ushiro` | ageing index variant |

## Stiffness index

The published stiffness index is `height / delta_t`, the one index in this set with velocity units and therefore a direct pulse-wave-velocity proxy. **PulseDB does not carry height**, so the canonical index cannot be computed here. `delta_t_proxy` (timing alone) and `si_bmi_proxy` (BMI-scaled) are substitutes and are labelled as such; neither should be reported as SI.