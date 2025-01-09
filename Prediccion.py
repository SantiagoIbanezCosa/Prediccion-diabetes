import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.experimental import enable_iterative_imputer  # Importar enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
import numpy as np

# ...existing code...

# Abrir el archivo CSV
df = pd.read_csv('diabetes.csv')

# Revisar la calidad de los datos
print(df.describe())
print(df.isnull().sum())

# Imputación avanzada con IterativeImputer
imputer = IterativeImputer(max_iter=10, random_state=42)
df[df.columns] = imputer.fit_transform(df[df.columns])

# Ajustar la normalización/escala de las características
scaler = PowerTransformer()
df[df.columns] = scaler.fit_transform(df[df.columns])

# Crear una matriz de relación
correlation_matrix = df.corr()

# Visualizar la matriz de relación
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()

# Preparar los datos
X = df.drop('Outcome', axis=1)  # Reemplaza 'Outcome' con el nombre de la columna objetivo
y = df['Outcome']  # Reemplaza 'Outcome' con el nombre de la columna objetivo

# Verificar si X y y no están vacíos
if X.empty or y.empty:
    raise ValueError("X o y están vacíos después de la eliminación de características. Revisa los datos de entrada.")

# Selección de características basada en la importancia de los árboles
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_train, y_train)
importances = rf.feature_importances_
indices = importances.argsort()[::-1]  # Ordenar por importancia

# Mostrar las características más importantes
for f in range(X_train.shape[1]):
    print(f"{X.columns[indices[f]]}: {importances[indices[f]]}")

# Regresión Lineal con validación cruzada
lr = LinearRegression()
scores = cross_val_score(lr, X_train, y_train, cv=5, scoring='r2')
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print("Linear Regression Results:")
print("Cross-validated R2 Scores:", scores)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred_lr))
print("R2 Score:", r2_score(y_test, y_pred_lr))

# Árbol de Decisión con ajuste de hiperparámetros
dt = DecisionTreeRegressor()
param_grid = {'max_depth': [3, 5, 7, 10], 'min_samples_split': [2, 5, 10]}
grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)
best_dt = grid_search.best_estimator_
y_pred_dt = best_dt.predict(X_test)
print("\nDecision Tree Results:")
print("Best Parameters:", grid_search.best_params_)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred_dt))
print("R2 Score:", r2_score(y_test, y_pred_dt))

# XGBoost con ajuste de hiperparámetros usando RandomizedSearchCV
xgb = XGBRegressor()
param_grid = {'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1, 0.2], 'n_estimators': [100, 200, 300]}
random_search = RandomizedSearchCV(xgb, param_grid, cv=5, n_iter=10, scoring='r2', random_state=42)
random_search.fit(X_train, y_train)
best_xgb = random_search.best_estimator_
y_pred_xgb = best_xgb.predict(X_test)
print("\nXGBoost Results:")
print("Best Parameters:", random_search.best_params_)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred_xgb))
print("R2 Score:", r2_score(y_test, y_pred_xgb))

# Evaluación de modelos con más métricas
def evaluate_model(y_test, y_pred):
    print(f'MAE: {mean_absolute_error(y_test, y_pred)}')
    print(f'MSE: {mean_squared_error(y_test, y_pred)}')
    print(f'RMSE: {mean_squared_error(y_test, y_pred) ** 0.5}')
    print(f'R2: {r2_score(y_test, y_pred)}')

print("\nLinear Regression Evaluation:")
evaluate_model(y_test, y_pred_lr)

print("\nDecision Tree Evaluation:")
evaluate_model(y_test, y_pred_dt)

print("\nXGBoost Evaluation:")
evaluate_model(y_test, y_pred_xgb)

# Probar con otros modelos adicionales (Random Forest)
# Random Forest
rf_model = RandomForestRegressor(n_estimators=100)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("\nRandom Forest Evaluation:")
evaluate_model(y_test, y_pred_rf)

# Ensamblado de Modelos (Stacking)
estimators = [
    ('lr', LinearRegression()),
    ('dt', DecisionTreeRegressor()),
    ('xgb', XGBRegressor())
]
stacking_regressor = StackingRegressor(estimators=estimators, final_estimator=RandomForestRegressor())
stacking_regressor.fit(X_train, y_train)
y_pred_stacking = stacking_regressor.predict(X_test)
print("\nStacking Regressor Evaluation:")
evaluate_model(y_test, y_pred_stacking)

# ...existing code...
