import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import os


output_dir = 'results'
os.makedirs(output_dir, exist_ok=True)


data = pd.read_excel('data/samVN.xlsx')

features = data.columns.difference(['Mau', 'Khoi_luong']).tolist()
X = data[features]
y = data['Khoi_luong']

model = LinearRegression()
model.fit(X, y)

coefficients = pd.DataFrame({
    'Ingredient': features,
    'Coefficient (g per 1%)': model.coef_
}).sort_values(by='Coefficient (g per 1%)', ascending=False)


sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 8), dpi=150)
colors = ['#2ecc71' if c > 0 else '#e74c3c' for c in coefficients['Coefficient (g per 1%)']]
bars = plt.barh(coefficients['Ingredient'], coefficients['Coefficient (g per 1%)'], color=colors, edgecolor='black', alpha=0.8)


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

print(coefficients.to_string(index=False))

print(f"\nIntercept (Base Weight): {model.intercept_:.2f}g")

