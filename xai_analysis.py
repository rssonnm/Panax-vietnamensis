import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import shap
import os

# Create output directory
output_dir = 'results'
os.makedirs(output_dir, exist_ok=True)

# 1. Load Data
df = pd.read_excel('data/samVN.xlsx')

# 2. Select Features for analysis
# We want to see which active ingredients influence 'Khoi_luong'
features = df.columns.difference(['Mau', 'Khoi_luong']).tolist()
X = df[features]
y = df['Khoi_luong']

# Setting aesthetic style
sns.set_theme(style="white", palette="muted")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

# 3. Correlation Heatmap
plt.figure(figsize=(10, 8), dpi=150)
correlation_matrix = df[features + ['Khoi_luong']].corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(
    correlation_matrix, 
    mask=mask, 
    annot=True, 
    fmt=".2f", 
    cmap='coolwarm', 
    vmax=1, vmin=-1, 
    center=0,
    square=True, 
    linewidths=.5, 
    cbar_kws={"shrink": .8}
)
plt.title('Correlation Heatmap: Active Ingredients vs Khoi luong', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'heatmap_correlation.png'), bbox_inches='tight')
plt.close()

# 4. xAI Analysis using Random Forest and SHAP
# Standardize features (optional for RF but good for consistency)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_df = pd.DataFrame(X_scaled, columns=features)

# Train Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_df, y)

# Calculate SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_df)

# 4.1 Feature Importance Plot (Standard)
importances = model.feature_importances_
feat_importances = pd.Series(importances, index=features).sort_values(ascending=True)

plt.figure(figsize=(10, 8), dpi=150)
feat_importances.plot(kind='barh', color='skyblue', edgecolor='navy')
plt.title('Random Forest Feature Importance (Influence on Khoi luong)', fontsize=16, fontweight='bold')
plt.xlabel('Importance Score')
plt.ylabel('Active Ingredients')
plt.grid(axis='x', linestyle=':', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'rf_feature_importance.png'), bbox_inches='tight')
plt.close()

# 4.2 SHAP Summary Plot
plt.figure(figsize=(10, 8), dpi=150)
shap.summary_plot(shap_values, X_df, show=False)
plt.title('SHAP Analysis: How Ingredients Influence Weight', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'shap_summary_plot.png'), bbox_inches='tight')
plt.close()

print("Heatmap and xAI plots generated successfully in 'results/' directory.")

# Analysis summary
print("\nTop Features by Importance:")
print(feat_importances.sort_values(ascending=False))
