import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn import metrics

def parse_duration(duration_str):
    """
    Parses an ISO 8601 duration string (e.g., 'PT7M37S', 'PT1H23M28S', 'PT45S')
    into total duration in seconds.
    """
    if pd.isna(duration_str):
        return 0
    # Pattern to match: P[#D][T[#H][#M][#S]]
    pattern = re.compile(r'P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?')
    match = pattern.match(duration_str)
    if not match:
        return 0
    days = int(match.group(1)) if match.group(1) else 0
    hours = int(match.group(2)) if match.group(2) else 0
    minutes = int(match.group(3)) if match.group(3) else 0
    seconds = int(match.group(4)) if match.group(4) else 0
    return days * 86400 + hours * 3600 + minutes * 60 + seconds

def main():
    print("="*60)
    print("YouTube Adview Prediction Pipeline")
    print("="*60)

    # 1. Load Datasets
    train_path = "train_lyst1717074532669.csv"
    test_path = "test_lyst1717074532669.csv"

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"Error: Make sure {train_path} and {test_path} are in the current folder!")
        return

    print("Loading data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"Train set shape: {train_df.shape}")
    print(f"Test set shape: {test_df.shape}")

    # 2. Data Cleaning & Preprocessing
    print("\nCleaning data...")
    # Convert 'F' (missing) to NaN in both datasets
    numeric_cols = ['views', 'likes', 'dislikes', 'comment']
    for col in numeric_cols:
        train_df[col] = pd.to_numeric(train_df[col].replace('F', np.nan), errors='coerce')
        test_df[col] = pd.to_numeric(test_df[col].replace('F', np.nan), errors='coerce')

    # Convert 'adview' to numeric in train set
    train_df['adview'] = pd.to_numeric(train_df['adview'], errors='coerce')

    # Drop rows with NaN in the training set
    train_df = train_df.dropna().reset_index(drop=True)
    print(f"Train set shape after dropping NaNs: {train_df.shape}")

    # Impute NaNs in the test set (using column medians from training set)
    # This ensures we don't lose any test rows and can predict for all test videos
    medians = {col: train_df[col].median() for col in numeric_cols}
    for col in numeric_cols:
        test_df[col] = test_df[col].fillna(medians[col])

    # Convert duration to seconds
    print("Parsing video durations...")
    train_df['duration_sec'] = train_df['duration'].apply(parse_duration)
    test_df['duration_sec'] = test_df['duration'].apply(parse_duration)

    # Extract date features from 'published'
    print("Extracting date features...")
    train_df['published'] = pd.to_datetime(train_df['published'], errors='coerce')
    test_df['published'] = pd.to_datetime(test_df['published'], errors='coerce')

    # Fill any invalid dates with a default date
    default_date = train_df['published'].dropna().min()
    train_df['published'] = train_df['published'].fillna(default_date)
    test_df['published'] = test_df['published'].fillna(default_date)

    for df in [train_df, test_df]:
        df['pub_year'] = df['published'].dt.year
        df['pub_month'] = df['published'].dt.month
        df['pub_day'] = df['published'].dt.day
        df['pub_weekday'] = df['published'].dt.weekday

    # Encode categorical 'category'
    print("Encoding categorical categories...")
    le_cat = LabelEncoder()
    train_df['category'] = le_cat.fit_transform(train_df['category'])
    # Safely handle unseen categories in the test set
    test_df['category'] = test_df['category'].map(lambda s: s if s in le_cat.classes_ else le_cat.classes_[0])
    test_df['category'] = le_cat.transform(test_df['category'])

    # Drop columns that are no longer needed
    cols_to_drop = ['vidid', 'duration', 'published']
    train_features = train_df.drop(columns=cols_to_drop + ['adview'])
    train_target = train_df['adview']

    test_features = test_df.drop(columns=cols_to_drop)

    print("\nFeature Columns used for training:")
    print(list(train_features.columns))

    # 3. Exploratory Data Analysis Plots
    print("\nGenerating correlation heatmap...")
    plt.figure(figsize=(10, 8))
    # Combine features and target to see correlations
    corr_df = train_features.copy()
    corr_df['adview'] = train_target
    sns.heatmap(corr_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Feature Correlation Heatmap (YouTube Adview Dataset)")
    plt.savefig("correlation_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved heatmap to correlation_heatmap.png")

    # 4. Train/Validation Split & Scaling
    X_train, X_val, y_train, y_val = train_test_split(train_features, train_target, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(test_features)

    # 5. Model Training & Evaluation
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=12, n_jobs=-1, random_state=42),
        "SVR": SVR(C=10.0, epsilon=0.2),
        "Neural Network (MLP)": MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    }

    results = []
    best_rmse = float('inf')
    best_model_name = None
    best_model_obj = None

    print("\nTraining and evaluating models...")
    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_val_scaled)
        
        # Calculate metrics
        mae = metrics.mean_absolute_error(y_val, preds)
        mse = metrics.mean_squared_error(y_val, preds)
        rmse = np.sqrt(mse)
        
        results.append({
            "Model": name,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse
        })
        
        print(f"    Validation RMSE: {rmse:.2f} | MAE: {mae:.2f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_model_name = name
            best_model_obj = model

    results_df = pd.DataFrame(results)
    print("\nComparison of Model Performance:")
    print(results_df.to_string(index=False))

    # Plot Model Performance comparison
    plt.figure(figsize=(10, 5))
    sns.barplot(x="Model", y="RMSE", data=results_df, palette="viridis")
    plt.title("Model Comparison - Root Mean Squared Error (RMSE)")
    plt.ylabel("RMSE (Lower is Better)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig("model_performance.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved model comparison plot to model_performance.png")

    # 6. Save the Best Model
    print(f"\nBest Model: {best_model_name} (RMSE = {best_rmse:.2f})")
    model_save_path = "youtube_adview_prediction_model.joblib"
    joblib.dump({
        'model': best_model_obj,
        'scaler': scaler,
        'le_cat': le_cat,
        'features': list(train_features.columns)
    }, model_save_path)
    print(f"Saved best model and scaler to: {model_save_path}")

    # 7. Predict on Test Set
    print("\nGenerating predictions on the test set...")
    test_preds = best_model_obj.predict(X_test_scaled)
    # Ensure no negative predictions (adviews cannot be negative)
    test_preds = np.clip(test_preds, 0, None)
    
    # Save predictions to CSV
    predictions_df = pd.DataFrame({
        'vidid': test_df['vidid'],
        'predicted_adview': test_preds
    })
    
    predictions_csv_path = "predictions.csv"
    predictions_df.to_csv(predictions_csv_path, index=False)
    print(f"Successfully saved test predictions to: {predictions_csv_path}")
    print("="*60)

if __name__ == "__main__":
    main()
