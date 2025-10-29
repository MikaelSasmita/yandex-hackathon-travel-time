import pandas as pd
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

# Load data
train = pd.read_csv("train_sample.csv")
test = pd.read_csv("test_sample.csv")

# Fix data types
for col in ['vehicle_density', 'population_density']:
    train[col] = pd.to_numeric(train[col], errors='coerce')
    test[col] = pd.to_numeric(test[col], errors='coerce')

# Log transform event_count
train['event_count'] = np.log1p(train['event_count'])
test['event_count'] = np.log1p(test['event_count'])

# Log transform target
y_train = np.log1p(train['travel_time'])
X_train = train.drop(columns='travel_time')
X_test = test.copy()

# Feature engineering
for df in [X_train, X_test]:
    df['route'] = df['start_point'] + '->' + df['end_point']
    df['same_area'] = (df['start_point'] == df['end_point']).astype(int)
    
# Columns for preprocessing
cat_cols = ['time_of_day', 'day_of_week', 'weather', 'route']
num_std_cols = ['historical_delay_factor', 'public_transport_availability', 'is_holiday', 'same_area']
num_mm_cols = ['traffic_condition', 'vehicle_density', 'population_density', 'event_count']

# Pipelines
cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])
num_std_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
num_mm_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', MinMaxScaler())
])

# Preprocessor
preprocessor = ColumnTransformer(transformers=[
    ('cat', cat_transformer, cat_cols),
    ('num_std', num_std_transformer, num_std_cols),
    ('num_mm', num_mm_transformer, num_mm_cols)
])

# Final model pipeline
pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('model', SGDRegressor(
        loss='huber',
        alpha=0.000000001,
        epsilon=1.2,
        learning_rate='constant',
        eta0=0.02,
        early_stopping=True,
        validation_fraction=0.15, 
        n_iter_no_change=4,
        tol=0.000001,
        random_state=42
    ))
])

# Train and predict
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
y_pred = np.expm1(y_pred)

# Save submission
pd.DataFrame(y_pred).to_csv("submission.csv", index=False, header=True, float_format='%.6f')
print("submission.csv berhasil dibuat.")
