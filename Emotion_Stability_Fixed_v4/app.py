"""
Emotion Stability Dashboard — Flask backend
No API key needed — uses the local XGBoost model for classification.

Run:
  python app.py
  Then open http://localhost:5000
"""
from flask import Flask, render_template, jsonify, request
from emotion_model import load_model

app = Flask(__name__)

# Train once at startup and keep in memory
print("⏳ Training XGBoost model… (~30 seconds on first run)")
MODEL = load_model()
print(f"✅ Model ready — accuracy: {MODEL['accuracy']}%")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyse", methods=["POST"])
def api_analyse():
    """
    Accept keystroke metrics from the frontend and return an emotion
    prediction using the local XGBoost model — no internet or API key needed.
    """
    body = request.get_json(force=True)
    metrics = body.get("metrics")
    if not metrics:
        return jsonify({"error": "Missing 'metrics' in request body"}), 400

    result = MODEL["predict"](metrics)   # returns {emotion, confidence, note}
    return jsonify(result)


@app.route("/api/results")
def api_results():
    """Return overall model performance metrics."""
    return jsonify({k: v for k, v in MODEL.items() if k != "predict"})


if __name__ == "__main__":
    print("🚀 Emotion Stability Dashboard → http://localhost:5000")
    app.run(debug=False, host="0.0.0.0", port=5000)
