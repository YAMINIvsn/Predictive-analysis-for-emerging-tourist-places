import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Path where your CSV files are stored
data_folder = "data"   # change this to your folder path

results = []

for file in os.listdir(data_folder):
    if file.endswith(".csv"):
        file_path = os.path.join(data_folder, file)
        
        try:
            # Load dataset
            df = pd.read_csv(file_path)
            
            # Drop empty or irrelevant columns if any
            df = df.dropna(axis=0)  # remove rows with NaN
            
            # Assume last column is target
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
            
            # Encode categorical features
            for col in X.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
            
            # If target is categorical, skip (we're doing regression only)
            if y.dtype == "object":
                print(f"⚠️ Skipping {file} because target is not numeric")
                continue
            
            # Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Try Linear Regression
            lin_model = LinearRegression()
            lin_model.fit(X_train, y_train)
            y_pred_lin = lin_model.predict(X_test)
            r2_lin = r2_score(y_test, y_pred_lin)
            rmse_lin = np.sqrt(mean_squared_error(y_test, y_pred_lin))
            
            # Try Random Forest
            rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
            rf_model.fit(X_train, y_train)
            y_pred_rf = rf_model.predict(X_test)
            r2_rf = r2_score(y_test, y_pred_rf)
            rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
            
            # Pick best model
            if r2_rf > r2_lin:
                best_model = "RandomForest"
                best_r2, best_rmse = r2_rf, rmse_rf
            else:
                best_model = "LinearRegression"
                best_r2, best_rmse = r2_lin, rmse_lin
            
            # Save results
            results.append({
                "File": file,
                "Best Model": best_model,
                "R2": round(best_r2, 4),
                "RMSE": round(best_rmse, 2),
                "Linear R2": round(r2_lin, 4),
                "RF R2": round(r2_rf, 4)
            })
            
            print(f"✅ Processed {file} | Best Model: {best_model} | R²={best_r2:.3f}, RMSE={best_rmse:.2f}")
        
        except Exception as e:
            print(f"❌ Error with {file}: {e}")

# Save summary results
results_df = pd.DataFrame(results)
results_df.to_csv("model_results_summary.csv", index=False)

print("\n📊 All results saved to 'model_results_summary.csv'")
