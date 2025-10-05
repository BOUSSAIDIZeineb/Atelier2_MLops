from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os

app = FastAPI(title="Churn Prediction API")

# ----- Config -----
MODEL_PATH = "models/churn_model.joblib"

# ----- Input schema -----
class PredictRequest(BaseModel):
    data: list  # Liste de dicts, chaque dict = une ligne à prédire

# ----- Load model once at startup -----
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Le modèle n'a pas été trouvé à {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

# ----- Prediction route -----
@app.post("/predict")
def predict(request: PredictRequest):
    if not request.data:
        raise HTTPException(status_code=400, detail="Aucune donnée fournie pour la prédiction.")
    
    # Convertir la liste de dicts en DataFrame
    df = pd.DataFrame(request.data)

    try:
        # Si le modèle supporte predict_proba
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df)[:, 1]  # Probabilité pour la classe 1
            preds = (proba >= 0.5).astype(int)  # Threshold = 0.5
        else:
            preds = model.predict(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur pendant la prédiction: {str(e)}")

    return {"predictions": preds.tolist()}
