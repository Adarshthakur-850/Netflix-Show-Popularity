import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pandas as pd

def plot_results(y_test, predictions, model, X_test):
    print("Generating plots...")
    if not os.path.exists("plots"):
        os.makedirs("plots")
        
    # 1. Actual vs Predicted (Best Model - usually XGB or RF)
    # Let's take the last prediction set (likely XGBoost or RF) for plotting if dict passed
    # Actually, let's strictly use XGBoost or Random Forest predictions if available
    
    y_pred = predictions.get('XGBoost', predictions.get('Random Forest', list(predictions.values())[0]))
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
    plt.plot([0, 100], [0, 100], 'r--', lw=2)
    plt.xlabel("Actual Popularity")
    plt.ylabel("Predicted Popularity")
    plt.title("Actual vs Predicted Popularity")
    plt.grid(True)
    plt.savefig("plots/actual_vs_predicted.png")
    plt.close()
    
    # 2. Feature Importance
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        features = X_test.columns
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances[indices], y=[features[i] for i in indices], palette="viridis")
        plt.title("Feature Importance")
        plt.xlabel("Importance Score")
        plt.tight_layout()
        plt.savefig("plots/feature_importance.png")
        plt.close()
