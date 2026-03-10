from src.data_loader import load_data
from src.feature_engineering import feature_engineering
from src.model import train_and_evaluate
from src.visualization import plot_results
import os

def main():
    print("Starting Netflix Popularity Prediction Pipeline...")
    
    # 1. Load Data
    df = load_data()
    print(f"Loaded {len(df)} titles.")
    
    # 2. Feature Engineering
    df = feature_engineering(df)
    print("Feature engineering complete.")
    
    # 3. Model Training
    best_model, X_test, y_test, predictions = train_and_evaluate(df)
    
    # 4. Visualization
    plot_results(y_test, predictions, best_model, X_test)
    print("Plots saved to plots/ folder.")
    
    print("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
