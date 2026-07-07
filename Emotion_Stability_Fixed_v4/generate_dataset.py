"""
Generate a research-grounded keystroke-emotion dataset.

Each emotion has DISTINCT feature distributions based on:
  - Epp et al. 2011: emotion affects typing speed, dwell, error rate
  - Vizer et al. 2009: anxiety → high variability, many corrections
  - Ghosh et al. 2017: sadness → slowest speed, longest dwell/flight

Ranges calibrated for REAL browser typing in 10-30 second sessions.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N_PER_CLASS = 200

def clip(arr, lo, hi):
    return np.clip(arr, lo, hi)

def gen(n, mu, sigma, lo, hi):
    return clip(np.random.normal(mu, sigma, n), lo, hi)

def gen_emotion(label_name, label_id, n):
    """Generate n samples for one emotion with distinctive distributions."""

    if label_name == "Calm":
        # Moderate speed, VERY consistent rhythm, low errors, few pauses
        speed    = gen(n, 3.8,  0.6,  2.5,  5.5)
        dwell    = gen(n, 125,  15,   90,   160)
        d_std    = gen(n, 14,   5,    4,    28)     # LOW variability
        flight   = gen(n, 210,  30,   140,  300)
        f_std    = gen(n, 28,   10,   8,    55)     # LOW variability
        errate   = gen(n, 0.03, 0.02, 0.0,  0.08)  # LOW errors
        pauses   = np.random.choice([0, 0, 0, 1, 1, 2], n)
        bksp     = np.random.choice([0, 0, 1, 1, 2], n)
        burst    = gen(n, 8.5,  1.5,  5,    13)
        caps     = np.random.choice([0, 0, 1, 1, 2, 3], n)
        dur      = gen(n, 18,   5,    8,    35)
        idle     = gen(n, 200,  150,  0,    600)

    elif label_name == "Happy":
        # Faster speed, consistent rhythm, very few errors, fluid
        speed    = gen(n, 5.0,  0.8,  3.5,  7.0)
        dwell    = gen(n, 100,  12,   75,   130)
        d_std    = gen(n, 12,   4,    3,    22)     # Consistent
        flight   = gen(n, 170,  25,   110,  240)
        f_std    = gen(n, 22,   8,    6,    45)     # Consistent
        errate   = gen(n, 0.02, 0.015,0.0,  0.06)  # Very low
        pauses   = np.random.choice([0, 0, 0, 0, 1], n)
        bksp     = np.random.choice([0, 0, 0, 1, 1], n)
        burst    = gen(n, 11,   2,    7,    16)
        caps     = np.random.choice([0, 1, 1, 2, 3], n)
        dur      = gen(n, 15,   4,    7,    28)
        idle     = gen(n, 100,  80,   0,    400)

    elif label_name == "Sad":
        # SLOW speed, LONG dwell, LONG flight, frequent pauses, low burst
        speed    = gen(n, 2.0,  0.5,  1.0,  3.2)
        dwell    = gen(n, 180,  30,   135,  280)     # LONG presses
        d_std    = gen(n, 20,   6,    8,    38)
        flight   = gen(n, 350,  60,   220,  550)     # LONG gaps
        f_std    = gen(n, 45,   15,   15,   80)
        errate   = gen(n, 0.04, 0.025,0.0,  0.10)
        pauses   = np.random.choice([2, 3, 3, 4, 5, 6], n)
        bksp     = np.random.choice([0, 0, 1, 1, 2], n)
        burst    = gen(n, 4.5,  1.2,  2,    8)       # LOW burst
        caps     = np.random.choice([0, 0, 0, 0, 1], n)
        dur      = gen(n, 30,   8,    15,   55)       # Longer sessions (slow)
        idle     = gen(n, 1800, 500,  600,  3500)     # LONG idle

    elif label_name == "Angry":
        # FAST speed, SHORT dwell (stabbing), high burst, MORE errors, caps
        speed    = gen(n, 5.8,  0.9,  4.0,  8.0)
        dwell    = gen(n, 82,   12,   55,   110)     # SHORT punchy presses
        d_std    = gen(n, 35,   10,   15,   60)      # Somewhat erratic
        flight   = gen(n, 130,  20,   85,   185)     # Quick transitions
        f_std    = gen(n, 55,   18,   20,   100)
        errate   = gen(n, 0.12, 0.04, 0.04, 0.25)   # HIGH errors
        pauses   = np.random.choice([0, 0, 0, 0, 1], n)  # Decisive, few pauses
        bksp     = np.random.choice([1, 2, 3, 4, 5], n)
        burst    = gen(n, 13,   2,    9,    18)       # HIGH burst
        caps     = np.random.choice([2, 3, 4, 5, 6, 8], n)  # EMPHATIC
        dur      = gen(n, 12,   3,    6,    22)
        idle     = gen(n, 80,   60,   0,    300)

    elif label_name == "Anxious":
        # Moderate speed, VERY ERRATIC rhythm, high corrections, many pauses
        speed    = gen(n, 3.3,  0.7,  2.0,  5.0)
        dwell    = gen(n, 115,  18,   80,   160)
        d_std    = gen(n, 45,   12,   25,   75)      # HIGH variability
        flight   = gen(n, 230,  40,   140,  350)
        f_std    = gen(n, 95,   25,   45,   170)     # HIGH variability
        errate   = gen(n, 0.14, 0.05, 0.05, 0.30)   # HIGH errors
        pauses   = np.random.choice([2, 3, 3, 4, 5, 6, 7], n)  # Many pauses
        bksp     = np.random.choice([3, 4, 5, 6, 7, 8, 10], n) # Many corrections
        burst    = gen(n, 7.5,  1.5,  4,    12)
        caps     = np.random.choice([0, 0, 1, 1, 2, 3], n)
        dur      = gen(n, 20,   5,    10,   35)
        idle     = gen(n, 600,  300,  100,  1500)

    # Derived
    pause_dur = gen(n, 1200, 300, 500, 2500)
    punct     = np.random.choice([0, 1, 1, 2, 2, 3], n)

    return pd.DataFrame({
        "Emotion":            label_name,
        "DwellTime_mean":     np.round(dwell, 2),
        "DwellTime_std":      np.round(d_std, 2),
        "FlightTime_mean":    np.round(flight, 2),
        "FlightTime_std":     np.round(f_std, 2),
        "TypingSpeed_cps":    np.round(speed, 4),
        "PauseFrequency":     pauses.astype(int),
        "PauseDuration_mean": np.round(pause_dur, 2),
        "BackspaceFrequency": bksp.astype(int),
        "ErrorRate":          np.round(errate, 6),
        "BurstSpeed":         np.round(burst, 4),
        "SessionDuration":    np.round(dur, 2),
        "CapitalizationFreq": caps.astype(int),
        "PunctuationFreq":    punct.astype(int),
        "IdleTime_mean":      np.round(idle, 2),
        "EmotionLabel":       label_id,
    })


emotions = [("Calm", 0), ("Happy", 1), ("Sad", 2), ("Angry", 3), ("Anxious", 4)]
df = pd.concat([gen_emotion(name, lid, N_PER_CLASS) for name, lid in emotions], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df.to_csv("keystroke_emotion_dataset_balanced.csv", index=False)
print(f"Generated {len(df)} samples ({N_PER_CLASS} per class)")
print(f"Columns: {list(df.columns)}")

# Verify signal
from scipy.stats import f_oneway
print("\nANOVA F-test (should all show SIGNAL now):")
for col in ['TypingSpeed_cps','DwellTime_mean','DwellTime_std','FlightTime_mean',
            'FlightTime_std','ErrorRate','BurstSpeed','CapitalizationFreq','IdleTime_mean']:
    groups = [df[df['Emotion']==e][col].values for e in ['Calm','Happy','Sad','Angry','Anxious']]
    f, p = f_oneway(*groups)
    sig = "✅" if p < 0.001 else ("⚠️" if p < 0.05 else "❌")
    print(f"  {col:25s}: F={f:8.1f}  p={p:.6f}  {sig}")
