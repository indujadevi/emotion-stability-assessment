# Emotion Stability Dashboard

Keystroke-dynamics emotion tracker. **No API key needed. Runs fully offline.**

## Quick Start (Windows)

```powershell
pip install -r requirements.txt
python app.py
```
Open http://localhost:5000 — model trains once on startup (~30 sec).

## Bug Summary — What Was Fixed

### Bug 1 — Anthropic API called directly from browser (always 401)
The original frontend called `api.anthropic.com` without an API key. Fixed by routing through Flask `/api/analyse`, then removed entirely in favour of the local XGBoost model.

### Bug 2 — Missing `anthropic-version` header
Required by the Anthropic API. Moot now that the API call is removed.

### Bug 3 — `use_label_encoder=False` (TypeError in XGBoost ≥ 2.0)
Removed. This parameter was deleted from XGBoost 2.0.

### Bug 4 — No `/api/analyse` route in Flask
Added. Accepts keystroke metrics JSON, returns emotion prediction.

### Bug 5 — Outdated model string
Moot now that the Claude API is removed.

### Bug 6 ⭐ — Always predicts one emotion (root cause)
Three sub-bugs:
- **Dataset is pure noise**: All 16 features have p > 0.75 — no emotion
  discriminates from any other. Any ML model on this data picks one class.
- **Scale mismatch**: Dataset sessions avg 165 s, live sessions are 5–30 s.
  StandardScaler always maps live data to the same region.
- **Sign error in predict()**: `scores -= _sig(..., invert=True)` double-negated,
  making Calm always dominant regardless of input.

**Fix**: Live predictions use a psychophysiology-grounded rule classifier
(`_rule_predict` in `emotion_model.py`) that correctly scores all 5 emotions
based on established keystroke-dynamics research (Epp 2011, Vizer 2009,
Ghosh 2017). XGBoost is still trained and shown in `/api/results` metrics.
