from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import numpy as np

import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

app = FastAPI()

# -------------------------
# Config
# -------------------------
DATA_PATH = "data"
MODEL_PATH = "models"
os.makedirs(MODEL_PATH, exist_ok=True)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Encoders
place_encoder = LabelEncoder()
state_encoder = LabelEncoder()

# Models dict
models = {}
transport_encoder = None
sentiment_encoder = None

# Month mapping
MONTH_MAPPING = {
    "January":1, "February":2, "March":3, "April":4,
    "May":5, "June":6, "July":7, "August":8,
    "September":9, "October":10, "November":11, "December":12
}

# -------------------------
# Load data
# -------------------------
def load_data():
    tourism = pd.read_csv(os.path.join(DATA_PATH, "Tourism_Fact.csv"))
    transport = pd.read_csv(os.path.join(DATA_PATH, "Transport_Fact.csv"))
    social = pd.read_csv(os.path.join(DATA_PATH, "Social_Fact.csv"))
    infra = pd.read_csv(os.path.join(DATA_PATH, "Infra_Fact.csv"))
    season = pd.read_csv(os.path.join(DATA_PATH, "Seasonal_Fact.csv"))
    state_dim = pd.read_csv(os.path.join(DATA_PATH, "State_Dim.csv"))  # State_ID, State_Name

    # Merge datasets
    tourism = tourism.merge(state_dim, on="State_ID", how="left")
    transport = transport.merge(state_dim, on="State_ID", how="left")
    social = social.merge(state_dim, on="State_ID", how="left")
    infra = infra.merge(state_dim, on="State_ID", how="left")
    season = season.merge(state_dim, on="State_ID", how="left")

    # Merge numeric data into tourism for model training
    tourism = tourism.merge(infra[['State_ID','Famous_Place','No_of_Hotels','Budget_Stay_pct','Luxury_Stay_pct']], 
                            on=['State_ID','Famous_Place'], how='left')
    tourism = tourism.merge(season[['State_ID','Month_ID','Famous_Place','Avg_Temperature_C']], 
                            left_on=['State_ID','Famous_Place','Year_ID'], 
                            right_on=['State_ID','Famous_Place','Month_ID'], how='left')
    tourism.rename(columns={
        'Avg_Temperature_C':'Temperature',
        'Budget_Stay_pct':'Budget_Hotel_Price',
        'Luxury_Stay_pct':'Luxury_Hotel_Price'
    }, inplace=True)

    # Fit encoders
    place_encoder.fit(tourism["Famous_Place"])
    state_encoder.fit(tourism["State_Name"])

    # Encode categorical fields
    tourism["Place_enc"] = place_encoder.transform(tourism["Famous_Place"])
    tourism["State_enc"] = state_encoder.transform(tourism["State_Name"])

# Add encoding for social as well
    social["State_enc"] = state_encoder.transform(social["State_Name"])

    return tourism, transport, social

# -------------------------
# Train models
# -------------------------
@app.get("/train", response_class=HTMLResponse)
def train_models(request: Request):
    global transport_encoder, sentiment_encoder
    tourism, transport, social = load_data()

    X = tourism[["State_enc", "Year_ID", "Place_enc"]]

    targets_reg = ["Domestic_Visitors","Foreign_Visitors","Avg_Stay_Days",
                   "Luxury_Hotel_Price","Budget_Hotel_Price","No_of_Hotels","Temperature"]

    metrics = {}  # Store accuracy/R2 for all models

    # Train regressors
    for target in targets_reg:
        if target in tourism.columns:
            y = tourism[target]
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            joblib.dump(model, os.path.join(MODEL_PATH, f"{target}.joblib"))
            models[target] = model

            # Evaluate
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            metrics[target] = {"R2": round(r2, 3), "RMSE": round(rmse, 2)}

    # Train transport classifier
    if "Preferred_Transport" in transport.columns:
        Xt = transport[["State_enc", "Year_ID"]]
        yt = transport["Preferred_Transport"]
        transport_encoder = LabelEncoder()
        yt_enc = transport_encoder.fit_transform(yt)
        model_t = RandomForestClassifier(n_estimators=100, random_state=42)
        model_t.fit(Xt, yt_enc)
        joblib.dump((model_t, transport_encoder), os.path.join(MODEL_PATH, "transport.joblib"))
        models["transport"] = model_t

        # Evaluate
        y_pred_t = model_t.predict(Xt)
        acc = accuracy_score(yt_enc, y_pred_t)
        metrics["Preferred_Transport"] = {"Accuracy": round(acc, 3)}

    # Train sentiment classifier
    if "Sentiment" in social.columns:
        Xs = social[["State_enc", "Year_ID"]]
        ys = social["Sentiment"]
        sentiment_encoder = LabelEncoder()
        ys_enc = sentiment_encoder.fit_transform(ys)
        model_s = RandomForestClassifier(n_estimators=100, random_state=42)
        model_s.fit(Xs, ys_enc)
        joblib.dump((model_s, sentiment_encoder), os.path.join(MODEL_PATH, "sentiment.joblib"))
        models["sentiment"] = model_s

        # Evaluate
        y_pred_s = model_s.predict(Xs)
        acc = accuracy_score(ys_enc, y_pred_s)
        metrics["Sentiment"] = {"Accuracy": round(acc, 3)}

    # Save encoders
    joblib.dump(place_encoder, os.path.join(MODEL_PATH, "place_encoder.joblib"))
    joblib.dump(state_encoder, os.path.join(MODEL_PATH, "state_encoder.joblib"))

    return templates.TemplateResponse("train.html", {
    "request": request,
    "message": "✅ Models trained successfully!",
    "metrics": metrics  # your metrics dictionary from training
})

# Get places by state (API)
# -------------------------
@app.get("/places/{state}", response_class=JSONResponse)
def get_places(state: str):
    tourism, _, _ = load_data()
    places = tourism[tourism["State_Name"]==state]["Famous_Place"].unique().tolist()
    return {"places": sorted(places)}

# -------------------------
# Predict (GET)
# -------------------------
@app.get("/predict", response_class=HTMLResponse)
def predict_get(request: Request):
    tourism, _, _ = load_data()
    states = sorted(tourism["State_Name"].unique())
    places = sorted(tourism["Famous_Place"].unique())
    
    # Empty initial data for charts
    yearly_data = []
    transport_counts = []
    sentiment_counts = []

    return templates.TemplateResponse("predict.html", {
        "request": request,
        "states": states,
        "places": places,
        "results": None,
        "yearly_data": yearly_data,
        "transport_counts": transport_counts,
        "sentiment_counts": sentiment_counts
    })

# -------------------------
# Predict (POST)
# -------------------------
@app.post("/predict", response_class=HTMLResponse)
def predict_post(request: Request,
                 state: str = Form(...),
                 place: str = Form(...),
                 month: str = Form(...)):

    tourism, transport, social = load_data()  # Load full data

    states = sorted(tourism["State_Name"].unique())
    places = sorted(tourism[tourism["State_Name"] == state]["Famous_Place"].unique())

    month_num = MONTH_MAPPING.get(month.capitalize(), 1)

    try:
        state_enc = state_encoder.transform([state])[0]
    except:
        state_enc = 0
    try:
        place_enc = place_encoder.transform([place])[0]
    except:
        place_enc = 0

    X_input = [[state_enc, month_num, place_enc]]

    results = {}

    # Predict numeric targets
    targets = ["Domestic_Visitors","Foreign_Visitors","Avg_Stay_Days",
               "Luxury_Hotel_Price","Budget_Hotel_Price","No_of_Hotels","Temperature"]
    for target in targets:
        if target not in models and os.path.exists(os.path.join(MODEL_PATH, f"{target}.joblib")):
            models[target] = joblib.load(os.path.join(MODEL_PATH, f"{target}.joblib"))
        if target in models:
            results[target] = round(float(models[target].predict(X_input)[0]),2)

    # -------------------------
    # Prepare data for visualizations
    # Year-wise numeric data for selected state and place
    yearly_df = tourism[(tourism["State_Name"]==state) & (tourism["Famous_Place"]==place)]
    yearly_data = yearly_df[["Year_ID","Domestic_Visitors","Foreign_Visitors"]].to_dict(orient="records")

    # Transport counts for pie chart (synthetic)
    transport_state_df = transport[transport["State_Name"]==state].copy()
    if "Preferred_Transport" not in transport_state_df.columns:
        # Create synthetic transport preference
        transport_state_df["Preferred_Transport"] = np.where(
            transport_state_df["No_of_Railway_Stations"] > transport_state_df["No_of_Airports"], "Train", "Air"
        )
    transport_counts_df = transport_state_df["Preferred_Transport"].value_counts().reset_index()
    transport_counts_df.columns = ["type","count"]
    transport_counts = transport_counts_df.to_dict(orient="records")

    # Sentiment counts for pie chart
    if "Sentiment_Score" in social.columns:
        sentiment_counts_df = social[social["State_Name"]==state]["Sentiment_Score"].value_counts().reset_index()
        sentiment_counts_df.columns = ["type","count"]
        sentiment_counts = sentiment_counts_df.to_dict(orient="records")
    else:
        sentiment_counts = []

    return templates.TemplateResponse("predict.html", {
        "request": request,
        "results": results,
        "states": states,
        "places": places,
        "selected_state": state,
        "selected_place": place,
        "selected_month": month,
        "yearly_data": yearly_data,
        "transport_counts": transport_counts,
        "sentiment_counts": sentiment_counts
    })

# -------------------------
# Index
# -------------------------
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
