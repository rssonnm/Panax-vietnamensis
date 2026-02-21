import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

output_dir = 'results'
os.makedirs(output_dir, exist_ok=True)

data = pd.read_excel('data/samVN.xlsx')

bins = [0, 30, 60, 90, np.inf]
labels = ['0-30', '30-60', '60-90', '>90']
data['Weight_Group'] = pd.cut(data['Khoi_luong'], bins=bins, labels=labels)

features = data.columns.difference(['Mau', 'Khoi_luong', 'Weight_Group']).tolist()
X = data[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=3)
pca_result = pca.fit_transform(X_scaled)

data['PCA1'] = pca_result[:, 0]
data['PCA2'] = pca_result[:, 1]
data['PCA3'] = pca_result[:, 2]

explained_variance = pca.explained_variance_ratio_
print(f"Explained variance: {explained_variance}")

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

colors = sns.color_palette("muted", len(labels))
color_map = dict(zip(labels, colors))

from matplotlib.patches import Ellipse

def draw_confidence_ellipse(x, y, ax, n_std=2.0, facecolor='none', **kwargs):
    """
    Draw a confidence ellipse for a set of data points.
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    import matplotlib.transforms as transforms
    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


plt.figure(figsize=(12, 10), dpi=1500)
ax = plt.gca()

sns.scatterplot(
    data=data, 
    x='PCA1', y='PCA2', 
    hue='Weight_Group', 
    style='Weight_Group',
    s=120, 
    alpha=0.9,
    edgecolor='w',
    ax=ax
)

for label in labels:
    subset = data[data['Weight_Group'] == label]
    if len(subset) > 2:  
        draw_confidence_ellipse(
            subset['PCA1'], subset['PCA2'], ax, 
            n_std=2.0, 
            edgecolor=color_map[label], 
            linewidth=2, 
            linestyle='--', 
            label=f'Ellipse {label}'
        )

plt.title('PCA 2D Score Plot with Confidence Ellipses', fontsize=18, fontweight='bold', pad=20)
plt.xlabel(f'PCA1 ({explained_variance[0]*100:.1f}%)', fontsize=14)
plt.ylabel(f'PCA2 ({explained_variance[1]*100:.1f}%)', fontsize=14)
plt.legend(title='Weight Group (g)', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'pca_2d_score_ellipses.png'), bbox_inches='tight')
plt.close()

fig = plt.figure(figsize=(12, 10), dpi=150)
ax = fig.add_subplot(111, projection='3d')

for label in labels:
    subset = data[data['Weight_Group'] == label]
    ax.scatter(
        subset['PCA1'], subset['PCA2'], subset['PCA3'],
        label=label,
        s=100,
        alpha=0.8,
        edgecolor='w',
        color=color_map[label]
    )

ax.set_title('PCA 3D Score Plot - SamVN Grouping', fontsize=18, fontweight='bold')
ax.set_xlabel(f'PCA1 ({explained_variance[0]*100:.1f}%)')
ax.set_ylabel(f'PCA2 ({explained_variance[1]*100:.1f}%)')
ax.set_zlabel(f'PCA3 ({explained_variance[2]*100:.1f}%)')
ax.view_init(elev=25, azim=45)
ax.legend(title='Weight Group (g)', loc='center left', bbox_to_anchor=(1.07, 0.5))
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'pca_3d_score.png'), bbox_inches='tight')
plt.close()

plt.figure(figsize=(14, 11), dpi=150)
ax = plt.gca()

# Plot scores
sns.scatterplot(
    data=data, 
    x='PCA1', y='PCA2', 
    hue='Weight_Group', 
    alpha=0.4, 
    s=80, 
    edgecolor=None,
    ax=ax
)

loading_data = pd.DataFrame(pca.components_.T[:, :2], columns=['PC1', 'PC2'], index=features)
scale_factor = 5.0  # Scale arrows for visibility

for feature in features:
    x_val = loading_data.loc[feature, 'PC1'] * scale_factor
    y_val = loading_data.loc[feature, 'PC2'] * scale_factor
    plt.arrow(0, 0, x_val, y_val, color='darkred', alpha=0.7, width=0.015, head_width=0.1)
    plt.text(x_val * 1.15, y_val * 1.15, feature, color='darkred', ha='center', va='center', fontsize=13, fontweight='bold')

plt.axhline(0, color='black', linestyle='-', alpha=0.2)
plt.axvline(0, color='black', linestyle='-', alpha=0.2)
plt.title('PCA Biplot (Scores + Loadings Mixed)', fontsize=20, fontweight='bold', pad=25)
plt.xlabel(f'PCA1 ({explained_variance[0]*100:.1f}%)', fontsize=14)
plt.ylabel(f'PCA2 ({explained_variance[1]*100:.1f}%)', fontsize=14)
plt.legend(title='Weight Group (g)', loc='upper left')
plt.grid(True, linestyle=':', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'pca_biplot.png'), bbox_inches='tight')
plt.close()

plt.figure(figsize=(12, 10), dpi=150)
for feature in features:
    plt.arrow(0, 0, loading_data.loc[feature, 'PC1'], loading_data.loc[feature, 'PC2'], 
              color='blue', alpha=0.6, width=0.005, head_width=0.03)
    plt.text(loading_data.loc[feature, 'PC1'] * 1.1, loading_data.loc[feature, 'PC2'] * 1.1, 
             feature, color='black', ha='center', va='center', fontsize=12)

plt.axhline(0, color='black', linestyle='--', alpha=0.3)
plt.axvline(0, color='black', linestyle='--', alpha=0.3)
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.title('PCA Loading Plot', fontsize=18, fontweight='bold')
plt.xlabel('PC1 Component Loading')
plt.ylabel('PC2 Component Loading')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'pca_loading.png'), bbox_inches='tight')
plt.close()


correlations = data[features + ['Khoi_luong']].corr()['Khoi_luong'].sort_values(ascending=False)
print("\nCorrelation with Khoi_luong:")
print(correlations)
