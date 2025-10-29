# model_selection.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler

# Load data
train = pd.read_csv("train_sample.csv")

# Fix data types
for col in ['vehicle_density', 'population_density']:
    train[col] = pd.to_numeric(train[col], errors='coerce')

# Log transform for skewed feature
train['event_count'] = train['event_count'].apply(lambda x: np.log1p(x))

# Split X and y
X = train.drop(columns='travel_time')
y = train['travel_time']

# Define columns
cat_cols = ['start_point', 'end_point', 'time_of_day', 'day_of_week', 'weather']
num_cols_std = ['historical_delay_factor', 'public_transport_availability', 'is_holiday']
num_cols_mm = ['traffic_condition', 'vehicle_density', 'population_density', 'event_count']

# Transformers
cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy="most_frequent")),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

num_std_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('scaler', StandardScaler())
])

num_mm_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('scaler', MinMaxScaler())
])

preprocessor = ColumnTransformer(transformers=[
    ('cat', cat_transformer, cat_cols),
    ('num_std', num_std_transformer, num_cols_std),
    ('num_mm', num_mm_transformer, num_cols_mm)
])

# Define models to evaluate
models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1),
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
    'SGDRegressor': SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
}

# Evaluate each model
for name, model in models.items():
    pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('model', model)
    ])

    scores = cross_val_score(pipeline, X, y, scoring='r2', cv=5)
    print(f"{name}: R2 Mean = {scores.mean():.4f}, Std = {scores.std():.4f}")
