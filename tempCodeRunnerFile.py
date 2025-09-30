@app.route("/train", methods=["GET", "POST"])
def train():
    global model, X, y

    message = None
    if request.method == "POST":
        # Load datasets
        tourism = pd.read_csv(os.path.join(UPLOAD_FOLDER, "Tourism_Fact.csv"))
        seasonal = pd.read_csv(os.path.join(UPLOAD_FOLDER, "Seasonal_Fact.csv"))
        social = pd.read_csv(os.path.join(UPLOAD_FOLDER, "Social_Fact.csv"))
        transport = pd.read_csv(os.path.join(UPLOAD_FOLDER, "Transport_Fact.csv"))

        # Merge on keys
        df = tourism.merge(seasonal, on=["State_ID", "Year_ID", "Famous_Place"], how="left")
        df = df.merge(social, on=["State_ID", "Year_ID", "Famous_Place"], how="left")
        df = df.merge(transport, on=["State_ID", "Year_ID", "Famous_Place"], how="left")

        # Features/target
        target = "Domestic_Visitors"
        X = df.drop(columns=drop_cols, errors="ignore")
        y = df[target]

        # Encode categorical columns
        for col in X.columns:
            if X[col].dtype == "object":
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))

        X = X.fillna(0)

        # Train Random Forest
        model = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            max_depth=None,
            n_jobs=-1
        )
        model.fit(X, y)

        joblib.dump(model, model_path)
        message = "🌲✅ Random Forest model trained successfully!"

    return render_template("train.html", message=message)
