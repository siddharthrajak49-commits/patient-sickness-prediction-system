import pickle
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# --- Load saved objects ---
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model_columns.pkl", "rb") as f:
    model_columns = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

with open("health_reports.pkl", "rb") as f:
    health_reports = pickle.load(f)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get patient data from form
        input_data = {col: request.form.get(col) for col in request.form}
        df = pd.DataFrame([input_data])

        # Convert numeric values
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                pass

        # Align with training columns
        df = pd.get_dummies(df)
        df = df.reindex(columns=model_columns, fill_value=0)

        # Predict probabilities and class
        probs = model.predict_proba(df)[0]
        pred_class = model.predict(df)[0]
        label = label_encoder.inverse_transform([pred_class])[0]

        # Build probability report
        prob_report = {
            label_encoder.inverse_transform([i])[0]: f"{p*100:.1f}%"
            for i, p in enumerate(probs)
        }

        # Fetch health report
        report = health_reports[label]

        # Add confidence of prediction
        confidence = max(probs) * 100

        return render_template(
            "result.html",
            prediction=label,
            confidence=f"{confidence:.1f}%",
            probabilities=prob_report,
            problem=report["problem"],
            suggestion=report["suggestion"]
        )

    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    import sys, os, socket

    # Default port
    port = int(os.environ.get("PORT", 5002))

    # Check if custom port is passed in args like --port=5003
    for arg in sys.argv:
        if arg.startswith("--port="):
            port = int(arg.split("=")[1])

    # Try to bind to the port, otherwise find a free one
    s = socket.socket()
    while True:
        try:
            s.bind(("0.0.0.0", port))
            s.close()
            break
        except OSError:
            port += 1  # increment to next port if in use

    app.run(host="0.0.0.0", port=port, debug=True)