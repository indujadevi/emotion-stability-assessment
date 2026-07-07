"""
emotion_model.py — Keystroke Emotion Classifier

Two classifiers work together:
  1. XGBoost trained on research-grounded synthetic data (for /api/results metrics)
  2. Rule-based classifier calibrated for real browser typing (for live /api/analyse)

Thresholds calibrated for real 10–30 second browser typing sessions.
Based on: Epp et al. 2011, Vizer et al. 2009, Ghosh et al. 2017.
"""

import os
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

EMOTION_LABELS = ["Calm", "Happy", "Sad", "Angry", "Anxious"]
LABEL_MAP      = {0: "Calm", 1: "Happy", 2: "Sad", 3: "Angry", 4: "Anxious"}


# ─────────────────────────────────────────────────────────────────────────────
# SCORING HELPER
# Returns a score based on where `val` falls relative to three thresholds.
# inv=True means LOW value is the positive signal.
# ─────────────────────────────────────────────────────────────────────────────
def _pts(val, t1, t2, t3, weight=1.0, inv=False):
    if inv:  # LOW val → positive score
        if   val <= t3: return  weight
        elif val <= t2: return  weight * 0.5
        elif val <= t1: return  weight * 0.15
        else:           return -weight * 0.3
    else:    # HIGH val → positive score
        if   val >= t3: return  weight
        elif val >= t2: return  weight * 0.5
        elif val >= t1: return  weight * 0.15
        else:           return -weight * 0.3


# ─────────────────────────────────────────────────────────────────────────────
# RULE-BASED CLASSIFIER — calibrated for real browser typing (10–30s sessions)
#
# Realistic browser ranges (from measurement):
#   Speed:   2–6 cps (avg ~3.5–4.5)
#   Dwell:   70–200 ms mean, std 8–50
#   Flight:  100–400 ms mean, std 15–120
#   Error:   0.01–0.20
#   Pauses:  0–7 raw count (NOT per-minute — sessions too short)
#   Backsp:  0–10 raw count
#   Burst:   3–16 cps
#   Caps:    0–9 raw count
#   Idle:    0–3000 ms
# ─────────────────────────────────────────────────────────────────────────────
def _rule_predict(m):
    speed  = float(m.get("TypingSpeed_cps",    0))
    dwell  = float(m.get("DwellTime_mean",     0))
    d_std  = float(m.get("DwellTime_std",      0))
    flight = float(m.get("FlightTime_mean",    0))
    f_std  = float(m.get("FlightTime_std",     0))
    errate = float(m.get("ErrorRate",          0))
    pauses = int(m.get("PauseFrequency",       0))
    burst  = float(m.get("BurstSpeed",         0))
    bksp   = int(m.get("BackspaceFrequency",   0))
    caps   = int(m.get("CapitalizationFreq",   0))
    dur    = max(float(m.get("SessionDuration", 1)), 0.5)
    idle   = float(m.get("IdleTime_mean",      0))

    # Use RAW COUNTS for pauses/backspaces — per-minute rates are
    # wildly unstable for 10–20 second sessions and were the #1 bug.

    sc = {e: 0.0 for e in EMOTION_LABELS}

    # ── CALM ─────────────────────────────────────────────────────────────
    # KEY signals: consistent rhythm (low d_std, low f_std),
    # low errors, few corrections, MODERATE speed (3–4.0 cps), few pauses.
    # Calm is the "steady, unremarkable" pattern — NOT fast, NOT slow.
    # FIX v4: Speed ceiling lowered to 4.0 (was 4.2) to avoid Calm/Happy overlap zone.
    #          Anti-penalty now fires at 4.1+ (was 4.4) — 4.1-4.4 is Happy territory.
    sc["Calm"] += _pts(d_std,   28,  20,  12,  2.5, inv=True)   # consistent dwell
    sc["Calm"] += _pts(f_std,   55,  38,  22,  2.0, inv=True)   # consistent flight
    sc["Calm"] += _pts(errate,  0.06, 0.04, 0.02, 2.5, inv=True)  # low errors
    sc["Calm"] += _pts(bksp,    3,   1,   0,   1.5, inv=True)   # few corrections
    sc["Calm"] += _pts(speed,   2.8, 3.5, 4.0, 1.5)             # moderate speed (cap at 4.0)
    sc["Calm"] += _pts(pauses,  3,   1,   0,   1.5, inv=True)   # few pauses
    sc["Calm"] += _pts(dwell,   100, 115, 140, 1.0)              # moderate dwell
    sc["Calm"] += _pts(caps,    4,   2,   0,   0.8, inv=True)   # not emphatic
    # Anti-signals: fast/slow/erratic → NOT calm
    # FIX v4: 4.1+ is Happy territory — penalise hard so Happy wins that range
    if speed >= 4.1: sc["Calm"] -= 2.5  # too fast → likely Happy or Angry
    if speed < 2.5:  sc["Calm"] -= 2.0  # too slow → likely Sad
    if d_std > 35:   sc["Calm"] -= 2.0
    if f_std > 70:   sc["Calm"] -= 1.5
    if errate > 0.10: sc["Calm"] -= 1.5
    if flight > 280: sc["Calm"] -= 1.5  # long flight gaps → likely Sad
    if dwell > 155:  sc["Calm"] -= 1.0  # heavy presses → likely Sad
    # FIX v4: dwell < 100 + low errors + good speed → more likely Happy than Calm
    if dwell < 100 and errate <= 0.03 and speed >= 3.8: sc["Calm"] -= 1.5

    # ── HAPPY ────────────────────────────────────────────────────────────
    # KEY signals: FASTER speed than calm (4.0+), consistent rhythm,
    # very few errors, fluid typing, good burst. Speed is THE differentiator
    # from Calm — happy people type faster AND smoother.
    # FIX v4: Lowered Happy speed thresholds: t1 3.8→3.5, t2 4.5→4.0, t3 6.0→5.5
    #          so that 4.0-4.4 cps gets meaningful Happy score (was getting 0.15×).
    #          Bonus trigger lowered from 4.5 to 4.0 to cover the common happy range.
    sc["Happy"] += _pts(speed,   3.5, 4.0, 5.5, 3.5)             # FASTER than calm (FIX v4)
    sc["Happy"] += _pts(d_std,   24,  16,  9,   2.5, inv=True)   # consistent
    sc["Happy"] += _pts(f_std,   45,  30,  16,  2.0, inv=True)   # consistent
    sc["Happy"] += _pts(errate,  0.05, 0.03, 0.01, 2.5, inv=True) # very low error
    sc["Happy"] += _pts(bksp,    2,   1,   0,   1.5, inv=True)   # few corrections
    sc["Happy"] += _pts(burst,   8,   10,  14,  2.0)              # decent burst
    sc["Happy"] += _pts(pauses,  2,   1,   0,   1.5, inv=True)   # very few pauses
    sc["Happy"] += _pts(dwell,   130, 110, 85,  1.5, inv=True)   # shorter dwell
    # Speed bonus: fast + consistent = happy (not angry which is fast + erratic)
    # FIX v4: Lower trigger from 4.5 → 4.0 to cover the real happy speed range
    if speed >= 4.0 and d_std < 22: sc["Happy"] += 2.0
    # Extra Happy signal: fast typing with near-zero errors and no pauses
    if speed >= 3.8 and errate <= 0.03 and pauses == 0: sc["Happy"] += 1.5
    # Anti-signals
    if speed < 2.5: sc["Happy"] -= 3.0  # too slow for happy
    if d_std > 30:  sc["Happy"] -= 1.5
    if errate > 0.10: sc["Happy"] -= 1.5
    # Happy vs Angry: happy has consistent rhythm, angry has short dwell + high speed
    if dwell <= 90 and d_std >= 28: sc["Happy"] -= 1.5  # erratic short-dwell = Angry not Happy

    # ── SAD ──────────────────────────────────────────────────────────────
    # KEY signals: SLOW speed, LONG dwell (heavy presses), LONG flight,
    # frequent pauses, low burst, high idle time.
    # Sadness detection threshold lowered: dwell 135+ and speed < 3 are strong signals.
    sc["Sad"] += _pts(speed,   3.2, 2.5, 1.5, 3.5, inv=True)    # SLOW (widened range)
    sc["Sad"] += _pts(dwell,   135, 165, 220, 3.0)               # LONG dwell (lowered t1)
    sc["Sad"] += _pts(flight,  220, 290, 420, 2.5)               # LONG flight (lowered t1)
    sc["Sad"] += _pts(idle,    350, 900, 2000, 2.0)              # long idle (lowered)
    sc["Sad"] += _pts(burst,   8,   5.5, 3,   2.0, inv=True)    # low burst
    sc["Sad"] += _pts(pauses,  2,   3,   5,   1.5)              # frequent pauses
    sc["Sad"] += _pts(caps,    3,   1,   0,   0.8, inv=True)    # not emphatic
    # Slow + long dwell combo is very strong sad indicator
    if speed <= 2.8 and dwell >= 145: sc["Sad"] += 2.0
    # Anti-signals
    if speed > 4.5:  sc["Sad"] -= 3.0  # too fast for sad
    if dwell < 115:  sc["Sad"] -= 2.0  # too short dwell for sad
    if burst > 10:   sc["Sad"] -= 1.5

    # ── ANGRY ────────────────────────────────────────────────────────────
    # KEY signals: FAST speed (4.0+ cps), SHORT dwell (stabbing), few/no pauses,
    # erratic rhythm, quick transitions. Caps and errors are supporting—not required.
    #
    # FIX 1: Lowered speed thresholds (4.0/4.8/6.0) — real angry typing is 4-5 cps,
    #         not the 4.5/5.5/7.0 which required near-impossible speeds.
    # FIX 2: Burst thresholds recalibrated for browser: browser sends max_keys_in_2s/2,
    #         so real burst maxes at ~6-8 cps. Old thresholds (10/13/16) were unreachable.
    # FIX 3: Caps made a supporting signal (weight 1.2, not 2.0) — most users don't
    #         hit CAPS even when angry, so it was unfairly penalizing angry detection.
    # FIX 4: Error rate threshold lowered (0.05/0.09/0.16) — angry typers often
    #         don't backspace at all (decisive), so low ErrorRate is not an anti-signal.
    sc["Angry"] += _pts(speed,  4.0, 4.8, 6.0, 3.5)              # FAST (FIX 1: lowered)
    sc["Angry"] += _pts(dwell,  110, 90,  70,  2.5, inv=True)    # SHORT dwell
    sc["Angry"] += _pts(burst,  5.0, 6.5, 8.5, 2.0)             # burst (FIX 2: recalibrated)
    sc["Angry"] += _pts(errate, 0.05, 0.09, 0.16, 1.5)          # errors (FIX 4: lowered)
    sc["Angry"] += _pts(caps,   2,   4,   6,   1.2)             # EMPHATIC caps (FIX 3: optional)
    sc["Angry"] += _pts(d_std,  22,  32,  45,  1.5)             # somewhat erratic
    sc["Angry"] += _pts(pauses, 2,   1,   0,   2.0, inv=True)   # decisive (boosted weight)
    sc["Angry"] += _pts(flight, 180, 140, 100, 1.0, inv=True)   # quick transitions
    # Speed + no pauses combo = strong angry signal (decisive fast typing)
    if speed >= 4.2 and pauses == 0: sc["Angry"] += 1.5
    # Short dwell + fast speed combo
    if speed >= 4.0 and dwell <= 100: sc["Angry"] += 1.0
    # Anti-signals
    if speed < 3.2:  sc["Angry"] -= 3.0   # too slow for angry
    if dwell > 160:  sc["Angry"] -= 2.0   # too long for stabbing
    if pauses > 4:   sc["Angry"] -= 1.5   # too hesitant for angry

    # ── ANXIOUS ──────────────────────────────────────────────────────────
    # KEY signals: VERY ERRATIC rhythm (high d_std, high f_std),
    # high corrections (backspaces), high error rate, hesitation pauses,
    # NOT fast (unlike angry). Key differentiator: variability + corrections.
    sc["Anxious"] += _pts(d_std,   30,  42,  60,  3.0)            # HIGH d variability
    sc["Anxious"] += _pts(f_std,   55,  85,  130, 3.0)            # HIGH f variability
    sc["Anxious"] += _pts(errate,  0.08, 0.13, 0.22, 2.5)         # high errors
    sc["Anxious"] += _pts(bksp,    3,   5,   8,   2.5)            # MANY corrections
    sc["Anxious"] += _pts(pauses,  2,   4,   6,   2.0)            # hesitation pauses
    sc["Anxious"] += _pts(idle,    200, 500, 1000, 1.0)           # some idle time
    # Speed limiter: very fast → Angry, not Anxious
    if speed > 5.5: sc["Anxious"] -= 2.0
    # Low variability → NOT anxious
    if d_std < 20:  sc["Anxious"] -= 2.0
    if f_std < 35:  sc["Anxious"] -= 1.5

    # ── DECISION ─────────────────────────────────────────────────────────
    best = max(sc, key=sc.get)

    # Softmax for confidence
    raw    = np.array([sc[e] for e in EMOTION_LABELS], dtype=float)
    exp_sc = np.exp(raw - raw.max())
    probs  = exp_sc / exp_sc.sum()
    conf   = round(float(probs[EMOTION_LABELS.index(best)]) * 100, 1)
    conf   = max(52.0, min(97.0, conf))

    # Build human-readable reasoning
    clues = []
    if speed >= 5.0:    clues.append(f"fast typing ({speed:.1f} cps)")
    elif speed <= 2.5:  clues.append(f"slow typing ({speed:.1f} cps)")
    else:               clues.append(f"moderate speed ({speed:.1f} cps)")

    if errate >= 0.12:  clues.append(f"high error rate ({errate*100:.0f}%)")
    elif errate <= 0.03:clues.append("very few corrections")

    if d_std >= 35:     clues.append("erratic key-hold rhythm")
    elif d_std <= 15:   clues.append("very consistent rhythm")

    if f_std >= 80:     clues.append("irregular key timing")
    elif f_std <= 25:   clues.append("smooth key flow")

    if pauses >= 4:     clues.append(f"frequent pauses ({pauses})")
    elif pauses == 0:   clues.append("no pauses")

    if burst >= 12:     clues.append(f"high burst speed ({burst:.1f} cps)")
    elif burst <= 5:    clues.append(f"low burst speed ({burst:.1f} cps)")

    if dwell >= 160:    clues.append(f"long key-hold ({dwell:.0f} ms)")
    elif dwell <= 90:   clues.append(f"short key-hold ({dwell:.0f} ms)")

    if caps >= 4:       clues.append("emphatic capitalisation")
    if bksp >= 5:       clues.append(f"many backspaces ({bksp})")

    return {
        "emotion":    best,
        "confidence": conf,
        "reasoning":  "Detected: " + ", ".join(clues[:4]) + ".",
    }


# ─────────────────────────────────────────────────────────────────────────────
# XGBoost Model — trained on research-grounded data for /api/results display
# ─────────────────────────────────────────────────────────────────────────────
def load_model():
    base = os.path.dirname(os.path.abspath(__file__))
    df   = pd.read_csv(os.path.join(base, "keystroke_emotion_dataset_balanced.csv"))

    feature_cols = [c for c in df.columns if c not in ("Emotion", "EmotionLabel")]
    X = df[feature_cols].fillna(0)
    y = df["EmotionLabel"]

    # Clip outliers
    for col in X.columns:
        lo, hi = np.percentile(X[col], [2, 98])
        X[col] = np.clip(X[col], lo, hi)

    # Add interaction features
    X["Speed_per_Dwell"] = X["TypingSpeed_cps"] / (X["DwellTime_mean"] + 1e-6)
    X["Error_per_Speed"] = X["ErrorRate"] / (X["TypingSpeed_cps"] + 1e-6)
    X["Dwell_std_ratio"] = X["DwellTime_std"] / (X["DwellTime_mean"] + 1e-6)
    X["Flight_std_ratio"] = X["FlightTime_std"] / (X["FlightTime_mean"] + 1e-6)

    le    = LabelEncoder()
    y_enc = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_enc, test_size=0.2, stratify=y_enc, random_state=42
    )

    xgb = XGBClassifier(
        n_estimators=300, learning_rate=0.08, max_depth=5,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric=["mlogloss", "merror"], random_state=42,
    )
    xgb.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False,
    )

    y_pred = xgb.predict(X_test)
    acc    = round(accuracy_score(y_test, y_pred) * 100, 2)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm     = confusion_matrix(y_test, y_pred)
    macro  = report.get("macro avg", {})
    evals  = xgb.evals_result()
    tl     = evals["validation_0"]["mlogloss"]
    vl     = evals["validation_1"]["mlogloss"]
    step   = max(1, len(tl) // 50)
    epochs = list(range(0, len(tl), step))

    class_metrics = {}
    for i, name in enumerate(EMOTION_LABELS):
        r = report.get(str(i), {})
        class_metrics[name] = {
            "precision": round(r.get("precision", 0) * 100, 1),
            "recall":    round(r.get("recall",    0) * 100, 1),
            "f1":        round(r.get("f1-score",  0) * 100, 1),
        }

    return {
        "accuracy":         acc,
        "precision":        round(macro.get("precision", 0) * 100, 2),
        "recall":           round(macro.get("recall",    0) * 100, 2),
        "f1":               round(macro.get("f1-score",  0) * 100, 2),
        "confusion_matrix": cm.tolist(),
        "class_metrics":    class_metrics,
        "training_curve": {
            "epochs":     epochs,
            "train_loss": [round(tl[i], 4) for i in epochs],
            "test_loss":  [round(vl[i], 4) for i in epochs],
        },
        "dataset_size":  len(df),
        "features_used": X.shape[1],
        "model":         "XGBoost · 300 trees + Rule Classifier (live)",
        "predict":       _rule_predict,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Smoke-testing rule classifier with realistic browser typing…\n")
    tests = [
        ("Calm", "Steady moderate typing, consistent rhythm, low errors", {
            "TypingSpeed_cps": 3.8, "DwellTime_mean": 125, "DwellTime_std": 14,
            "FlightTime_mean": 210, "FlightTime_std": 28, "ErrorRate": 0.03,
            "PauseFrequency": 1, "BackspaceFrequency": 1, "CapitalizationFreq": 1,
            "BurstSpeed": 8.5, "SessionDuration": 18, "IdleTime_mean": 200,
        }),
        ("Happy", "Fast fluid typing, very consistent, almost no errors", {
            "TypingSpeed_cps": 5.2, "DwellTime_mean": 100, "DwellTime_std": 10,
            "FlightTime_mean": 170, "FlightTime_std": 20, "ErrorRate": 0.02,
            "PauseFrequency": 0, "BackspaceFrequency": 0, "CapitalizationFreq": 2,
            "BurstSpeed": 12, "SessionDuration": 14, "IdleTime_mean": 80,
        }),
        ("Sad", "Very slow, long key holds, long gaps, many pauses", {
            "TypingSpeed_cps": 1.8, "DwellTime_mean": 190, "DwellTime_std": 20,
            "FlightTime_mean": 380, "FlightTime_std": 45, "ErrorRate": 0.04,
            "PauseFrequency": 5, "BackspaceFrequency": 1, "CapitalizationFreq": 0,
            "BurstSpeed": 4, "SessionDuration": 30, "IdleTime_mean": 1800,
        }),
        ("Angry", "Fast stabbing, short dwell, high burst, caps, errors", {
            "TypingSpeed_cps": 5.2, "DwellTime_mean": 82, "DwellTime_std": 35,
            "FlightTime_mean": 130, "FlightTime_std": 58, "ErrorRate": 0.10,
            "PauseFrequency": 0, "BackspaceFrequency": 3, "CapitalizationFreq": 5,
            "BurstSpeed": 7.5, "SessionDuration": 10, "IdleTime_mean": 50,
        }),
        ("Anxious", "Erratic rhythm, many corrections, hesitation pauses", {
            "TypingSpeed_cps": 3.2, "DwellTime_mean": 115, "DwellTime_std": 48,
            "FlightTime_mean": 230, "FlightTime_std": 100, "ErrorRate": 0.16,
            "PauseFrequency": 5, "BackspaceFrequency": 7, "CapitalizationFreq": 1,
            "BurstSpeed": 7, "SessionDuration": 20, "IdleTime_mean": 700,
        }),
        ("Calm", "Another calm session — moderate speed, steady rhythm", {
            "TypingSpeed_cps": 3.6, "DwellTime_mean": 122, "DwellTime_std": 17,
            "FlightTime_mean": 205, "FlightTime_std": 34, "ErrorRate": 0.03,
            "PauseFrequency": 0, "BackspaceFrequency": 1, "CapitalizationFreq": 2,
            "BurstSpeed": 9, "SessionDuration": 15, "IdleTime_mean": 150,
        }),
        ("Angry", "Moderately angry — fast, some errors, caps", {
            "TypingSpeed_cps": 4.5, "DwellTime_mean": 95, "DwellTime_std": 30,
            "FlightTime_mean": 155, "FlightTime_std": 50, "ErrorRate": 0.07,
            "PauseFrequency": 0, "BackspaceFrequency": 3, "CapitalizationFreq": 3,
            "BurstSpeed": 6.5, "SessionDuration": 12, "IdleTime_mean": 0,
        }),
        ("Anxious", "Moderate anxious — erratic, many backspaces", {
            "TypingSpeed_cps": 3.5, "DwellTime_mean": 120, "DwellTime_std": 40,
            "FlightTime_mean": 210, "FlightTime_std": 90, "ErrorRate": 0.13,
            "PauseFrequency": 3, "BackspaceFrequency": 6, "CapitalizationFreq": 1,
            "BurstSpeed": 8, "SessionDuration": 16, "IdleTime_mean": 500,
        }),
        ("Sad", "Moderate sad — slow, long dwell, pauses", {
            "TypingSpeed_cps": 2.5, "DwellTime_mean": 160, "DwellTime_std": 18,
            "FlightTime_mean": 300, "FlightTime_std": 40, "ErrorRate": 0.04,
            "PauseFrequency": 3, "BackspaceFrequency": 1, "CapitalizationFreq": 0,
            "BurstSpeed": 6, "SessionDuration": 25, "IdleTime_mean": 1200,
        }),
        ("Happy", "Moderate happy — good speed, low errors, fluid", {
            "TypingSpeed_cps": 4.5, "DwellTime_mean": 108, "DwellTime_std": 13,
            "FlightTime_mean": 185, "FlightTime_std": 24, "ErrorRate": 0.01,
            "PauseFrequency": 0, "BackspaceFrequency": 0, "CapitalizationFreq": 1,
            "BurstSpeed": 10, "SessionDuration": 16, "IdleTime_mean": 100,
        }),
    ]

    passed = 0
    for expected, desc, metrics in tests:
        r  = _rule_predict(metrics)
        ok = r["emotion"] == expected
        passed += ok
        mark = "✅" if ok else "❌"
        print(f"  {mark} {expected:8s} → {r['emotion']:8s} ({r['confidence']:5.1f}%)  {desc}")
    print(f"\n{passed}/{len(tests)} correct")
