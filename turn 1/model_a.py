import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
import geopandas as gpd
import rasterio
from rasterio.plot import show
from rasterio.mask import mask
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import shap

# Step 1: Generate Dummy Satellite Data and Shapefile

# Create a dummy raster image (for demonstration purposes)
height, width = 100, 100
data = np.random.randint(0, 255, size=(height, width)).astype(np.uint8)
transform = rasterio.transform.from_origin(0, 0, 1, 1)
new_dataset = rasterio.open(
    'dummy_satellite.tif',
    'w',
    driver='GTiff',
    height=height,
    width=width,
    count=1,
    dtype=rasterio.uint8,
    crs='epsg:4326',
    transform=transform,
)
new_dataset.write(data, 1)
new_dataset.close()

# Create a dummy shapefile for agricultural fields
polygons = [
    Polygon([(0, 0), (0, 20), (20, 20), (20, 0), (0, 0)]),
    Polygon([(30, 10), (30, 30), (50, 30), (50, 10), (30, 10)]),
]

gdf = gpd.GeoDataFrame({'geometry': polygons}, crs='epsg:4326')
gdf.to_file('dummy_fields.shp')

# Step 2: Load and preprocess data

# Load satellite data
with rasterio.open('dummy_satellite.tif') as src:
    image = src.read(1)
    affine = src.transform
    crs = src.crs

# Mask the satellite data with the shapefile
masked_image, out_transform = mask(dataset=src, shapes=gdf['geometry'], crop=True)
masked_image = masked_image.squeeze()

# Flatten the masked image data for ML analysis
X = masked_image.flatten()

# Generate dummy yield data for the fields (same length as X)
y = np.random.randint(1000, 5000, size=len(X))

# Step 3: Perform Machine Learning Analysis

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Initialize a Random Forest Regressor
regressor = RandomForestRegressor(random_state=42)

# Train the model
regressor.fit(X_train.reshape(-1, 1), y_train)

# Step 4: Evaluate the model
y_pred = regressor.predict(X_test.reshape(-1, 1))
mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse}")

# Step 5: Interpret the model using SHAP
explainer = shap.TreeExplainer(regressor)
shap_values = explainer.shap_values(X)

shap.summary_plot(shap_values, X, feature_names=['Satellite Index'])

plt.show()
