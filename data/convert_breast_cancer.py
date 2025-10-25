import pandas as pd

# Define column names from UCI dataset documentation
columns = [
    "id", "diagnosis", "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
    "compactness_worst", "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
]

# Read raw .data file (no headers)
df = pd.read_csv('data/wdbc.data', header=None, names=columns)

# Drop ID column
df = df.drop(columns=['id'])

# Convert 'diagnosis' column to binary
df['label'] = df['diagnosis'].map({'M': 1, 'B': 0})
df = df.drop(columns=['diagnosis'])

# Save cleaned CSV
df.to_csv('data/breast_cancer_clean.csv', index=False)

print("✅ Saved clean dataset as data/breast_cancer_clean.csv")
print(df.head())