import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

try:
    from xgboost import XGBRegressor
    xgb_installed = True
except ImportError:
    xgb_installed = False

# Load data
df = pd.read_csv('data/house_prices.csv')

# Features and target
X = df.drop('price', axis=1)
y = df['price']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'LinearRegression': LinearRegression(),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
}
if xgb_installed:
    models['XGBoost'] = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    results[name] = rmse
    print(f'{name} RMSE: {rmse:.2f}')

best_model_name = min(results, key=results.get)
best_model = models[best_model_name]

print(f'\nBest model: {best_model_name} (RMSE: {results[best_model_name]:.2f})')

# Save the best model
joblib.dump(best_model, 'app/models/model.pkl')
print('Best model saved to app/models/model.pkl')
