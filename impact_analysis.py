import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import os

# Create output directory
output_dir = 'results'
os.makedirs(output_dir, exist_ok=True)

# 1. Load Data
df = pd.read_excel('data/samVN.xlsx')

# 2. Select Features (Active Ingredients) and Target (Weight)
features = df.columns.difference(['Mau', 'Khoi_luong']).tolist()
X = df[features]
y = df['Khoi_luong']

# 3. Fit Linear Regression model
model = LinearRegression()
model.fit(X, y)

# 4. Get Coefficients
# The coefficient tells us the change in y (Weight in grams) for a 1-unit (1%) change in x.
coefficients = pd.DataFrame({
    'Ingredient': features,
    'Coefficient (g per 1%)': model.coef_
}).sort_values(by='Coefficient (g per 1%)', ascending=False)

# 5. Visualizing the Coefficients
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 8), dpi=150)
colors = ['#2ecc71' if c > 0 else '#e74c3c' for c in coefficients['Coefficient (g per 1%)']]
bars = plt.barh(coefficients['Ingredient'], coefficients['Coefficient (g per 1%)'], color=colors, edgecolor='black', alpha=0.8)

# Add value labels
for bar in bars:
    width = bar.get_width()
    label_x_pos = width + (1 if width > 0 else -1)
    plt.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:+.2f}g', 
             va='center', ha='left' if width > 0 else 'right', fontweight='bold', fontsize=12)

plt.axvline(0, color='black', linewidth=1.5)
plt.title('Mathematical Impact: Change in Weight (g) per 1% Ingredient Change', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Change in Weight (grams)', fontsize=14)
plt.ylabel('Active Ingredient', fontsize=14)
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'feature_impact_plus_minus.png'), bbox_inches='tight')
plt.close()

# 6. Save Coefficients to a markdown-ready CSV/Text for the summary
print("Mathematical Coefficients (Impact per 1% concentration change):")
print(coefficients.to_string(index=False))

print(f"\nIntercept (Base Weight): {model.intercept_:.2f}g")
print("\nPlot saved as 'feature_impact_plus_minus.png' in the results folder.")
