# PRISM-F1 Audit Trail Demo

This document shows the per-prediction reasoning produced by PRISM-F1 for five test examples. Each prediction is decomposable into its 6 interpretable concept contributions plus a bias.

```

══════════════════════════════════════════════════════════════════════════════
  EXAMPLE 1 / 5  —  CONFIDENT PIT PREDICTION
══════════════════════════════════════════════════════════════════════════════

  Input row (id=453708):
    Driver:        COL
    Race:          Spanish Grand Prix 2025
    Lap:           33  (Stint 2, RaceProgress 0.43)
    Compound:      MEDIUM  (TyreLife=29)
    Position:      P14  (recent change: +3)
    Lap time:      83.873s  (delta +1.839, cumulative_deg -50.31)

  What PRISM-F1 perceives (6 named concepts):
    degradation_severity    0.9400  ███████████████████
    pace_decay_rate         0.0694  █
    strategic_window        0.7950  ████████████████
    track_position_risk     0.1622  ███
    undercut_pressure       0.9940  ████████████████████
    endgame_proximity       0.0000  

  How the decision is built (additive concept contributions):
    concept                    value ×   weight  =    contribution
    ----------------------------------------------------------------------
    degradation_severity      0.9400 ×  +1.0731  =         +1.0088  → pit
    pace_decay_rate           0.0694 ×  -0.1885  =         -0.0131  —
    strategic_window          0.7950 ×  +1.1990  =         +0.9532  → pit
    track_position_risk       0.1622 ×  -0.2383  =         -0.0386  —
    undercut_pressure         0.9940 ×  +1.6079  =         +1.5982  → pit
    endgame_proximity         0.0000 ×  -2.9945  =         -0.0001  —
    bias                             ×           =         -0.6079
    ----------------------------------------------------------------------
    TOTAL LOGIT                                            +2.9005
    sigmoid(logit) = pit_prob                                  0.9479

  Verdict: PRISM-F1 predicts PIT (probability 0.948)
  Primary drivers: undercut_pressure (+1.60) and degradation_severity (+1.01)

══════════════════════════════════════════════════════════════════════════════
  EXAMPLE 2 / 5  —  CONFIDENT STAY-OUT PREDICTION
══════════════════════════════════════════════════════════════════════════════

  Input row (id=585901):
    Driver:        SAR
    Race:          Qatar Grand Prix 2023
    Lap:           57  (Stint 5, RaceProgress 1.00)
    Compound:      HARD  (TyreLife=19)
    Position:      P17  (recent change: -1)
    Lap time:      90.057s  (delta +4.543, cumulative_deg -14.89)

  What PRISM-F1 perceives (6 named concepts):
    degradation_severity    0.0985  ██
    pace_decay_rate         0.1811  ████
    strategic_window        0.0003  
    track_position_risk     0.1120  ██
    undercut_pressure       0.0004  
    endgame_proximity       0.7023  ██████████████

  How the decision is built (additive concept contributions):
    concept                    value ×   weight  =    contribution
    ----------------------------------------------------------------------
    degradation_severity      0.0985 ×  +1.0731  =         +0.1057  → pit
    pace_decay_rate           0.1811 ×  -0.1885  =         -0.0341  —
    strategic_window          0.0003 ×  +1.1990  =         +0.0004  —
    track_position_risk       0.1120 ×  -0.2383  =         -0.0267  —
    undercut_pressure         0.0004 ×  +1.6079  =         +0.0006  —
    endgame_proximity         0.7023 ×  -2.9945  =         -2.1029  → stay
    bias                             ×           =         -0.6079
    ----------------------------------------------------------------------
    TOTAL LOGIT                                            -2.6650
    sigmoid(logit) = pit_prob                                  0.0651

  Verdict: PRISM-F1 predicts STAY OUT (probability 0.065)
  Primary drivers: endgame_proximity (-2.10) and degradation_severity (+0.11)

══════════════════════════════════════════════════════════════════════════════
  EXAMPLE 3 / 5  —  BORDERLINE 50/50 CALL
══════════════════════════════════════════════════════════════════════════════

  Input row (id=595611):
    Driver:        D210
    Race:          Spanish Grand Prix 2024
    Lap:           45  (Stint 3, RaceProgress 0.58)
    Compound:      HARD  (TyreLife=16)
    Position:      P6  (recent change: -3)
    Lap time:      80.853s  (delta -15.851, cumulative_deg -51.12)

  What PRISM-F1 perceives (6 named concepts):
    degradation_severity    0.0519  █
    pace_decay_rate         0.0082  
    strategic_window        0.5472  ███████████
    track_position_risk     0.4295  █████████
    undercut_pressure       0.0000  
    endgame_proximity       0.0000  

  How the decision is built (additive concept contributions):
    concept                    value ×   weight  =    contribution
    ----------------------------------------------------------------------
    degradation_severity      0.0519 ×  +1.0731  =         +0.0557  → pit
    pace_decay_rate           0.0082 ×  -0.1885  =         -0.0015  —
    strategic_window          0.5472 ×  +1.1990  =         +0.6561  → pit
    track_position_risk       0.4295 ×  -0.2383  =         -0.1024  → stay
    undercut_pressure         0.0000 ×  +1.6079  =         +0.0001  —
    endgame_proximity         0.0000 ×  -2.9945  =         -0.0000  —
    bias                             ×           =         -0.6079
    ----------------------------------------------------------------------
    TOTAL LOGIT                                            -0.0000
    sigmoid(logit) = pit_prob                                  0.5000

  Verdict: PRISM-F1 predicts STAY OUT (probability 0.500)
  Primary drivers: strategic_window (+0.66) and track_position_risk (-0.10)

══════════════════════════════════════════════════════════════════════════════
  EXAMPLE 4 / 5  —  STRONG END-OF-RACE ANTI-PIT SIGNAL
══════════════════════════════════════════════════════════════════════════════

  Input row (id=531463):
    Driver:        ZON
    Race:          São Paulo Grand Prix 2024
    Lap:           67  (Stint 2, RaceProgress 0.94)
    Compound:      INTERMEDIATE  (TyreLife=25)
    Position:      P2  (recent change: +0)
    Lap time:      81.189s  (delta +0.065, cumulative_deg -8.74)

  What PRISM-F1 perceives (6 named concepts):
    degradation_severity    0.7653  ███████████████
    pace_decay_rate         0.2513  █████
    strategic_window        0.0042  
    track_position_risk     0.4934  ██████████
    undercut_pressure       0.0223  
    endgame_proximity       0.8693  █████████████████

  How the decision is built (additive concept contributions):
    concept                    value ×   weight  =    contribution
    ----------------------------------------------------------------------
    degradation_severity      0.7653 ×  +1.0731  =         +0.8212  → pit
    pace_decay_rate           0.2513 ×  -0.1885  =         -0.0474  —
    strategic_window          0.0042 ×  +1.1990  =         +0.0051  —
    track_position_risk       0.4934 ×  -0.2383  =         -0.1176  → stay
    undercut_pressure         0.0223 ×  +1.6079  =         +0.0359  —
    endgame_proximity         0.8693 ×  -2.9945  =         -2.6031  → stay
    bias                             ×           =         -0.6079
    ----------------------------------------------------------------------
    TOTAL LOGIT                                            -2.5138
    sigmoid(logit) = pit_prob                                  0.0749

  Verdict: PRISM-F1 predicts STAY OUT (probability 0.075)
  Primary drivers: endgame_proximity (-2.60) and degradation_severity (+0.82)

══════════════════════════════════════════════════════════════════════════════
  EXAMPLE 5 / 5  —  STRONG UNDERCUT-PRESSURE PIT SIGNAL
══════════════════════════════════════════════════════════════════════════════

  Input row (id=595554):
    Driver:        D027
    Race:          Australian Grand Prix 2022
    Lap:           43  (Stint 2, RaceProgress 0.60)
    Compound:      HARD  (TyreLife=25)
    Position:      P6  (recent change: +4)
    Lap time:      84.927s  (delta +0.856, cumulative_deg +27.27)

  What PRISM-F1 perceives (6 named concepts):
    degradation_severity    0.3153  ██████
    pace_decay_rate         0.5132  ██████████
    strategic_window        0.6699  █████████████
    track_position_risk     0.7913  ████████████████
    undercut_pressure       0.9993  ████████████████████
    endgame_proximity       0.0000  

  How the decision is built (additive concept contributions):
    concept                    value ×   weight  =    contribution
    ----------------------------------------------------------------------
    degradation_severity      0.3153 ×  +1.0731  =         +0.3384  → pit
    pace_decay_rate           0.5132 ×  -0.1885  =         -0.0967  → stay
    strategic_window          0.6699 ×  +1.1990  =         +0.8032  → pit
    track_position_risk       0.7913 ×  -0.2383  =         -0.1886  → stay
    undercut_pressure         0.9993 ×  +1.6079  =         +1.6067  → pit
    endgame_proximity         0.0000 ×  -2.9945  =         -0.0000  —
    bias                             ×           =         -0.6079
    ----------------------------------------------------------------------
    TOTAL LOGIT                                            +1.8550
    sigmoid(logit) = pit_prob                                  0.8647

  Verdict: PRISM-F1 predicts PIT (probability 0.865)
  Primary drivers: undercut_pressure (+1.61) and strategic_window (+0.80)
```

## Final Decision Block Weights

| Concept | Weight |
|---|---|
| degradation_severity | +1.0731 |
| pace_decay_rate | -0.1885 |
| strategic_window | +1.1990 |
| track_position_risk | -0.2383 |
| undercut_pressure | +1.6079 |
| endgame_proximity | -2.9945 |
| **bias** | -0.6079 |
