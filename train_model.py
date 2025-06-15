import pandas as pd 
import numpy as np
df = pd.read_csv('C:\\Users\\Hi\\Downloads\\archive\\Turbine_Data.csv')

print(df)

df.rename(columns={df.columns[0]: 'Timestamp'}, inplace=True)
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

print(df.info())

missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0].sort_values(ascending=False)
print("Missing values:\n", missing_values)
df_imputed = df.copy()

df_imputed.fillna(method='ffill', inplace=True)
df_imputed.fillna(method='bfill', inplace=True)

print(df_imputed.isnull().sum())
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    df_clean = df[(df[column] >= lower_limit) & (df[column] <= upper_limit)]
    return df_clean

for col in ['WindSpeed', 'RotorRPM', 'ActivePower']:
    df_imputed = remove_outliers_iqr(df_imputed, col)

print("Shape after outlier removal:", df_imputed.shape)
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.boxplot(y=df['WindSpeed'], ax=axes[0], color='skyblue')
axes[0].set_title('Before Outlier Removal - WindSpeed')

sns.boxplot(y=df_imputed['WindSpeed'], ax=axes[1], color='lightgreen')
axes[1].set_title('After Outlier Removal - WindSpeed')

plt.tight_layout()
plt.show()
from sklearn.preprocessing import StandardScaler

df_scaled = df_imputed.copy()

numeric_cols = df_scaled.select_dtypes(include=['float64', 'int64']).columns
features_to_scale = df_scaled[numeric_cols]

scaler = StandardScaler()
scaled_array = scaler.fit_transform(features_to_scale)

scaled_df = pd.DataFrame(scaled_array, columns=numeric_cols)
scaled_df['Timestamp'] = df_imputed['Timestamp'].values  # Add timestamp back

print(scaled_df.head())
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.histplot(df_imputed['RotorRPM'], ax=axes[0], kde=True, color='coral')
axes[0].set_title('Original RotorRPM Distribution')
axes[0].set_xlabel('RotorRPM')

sns.histplot(scaled_df['RotorRPM'], ax=axes[1], kde=True, color='mediumseagreen')
axes[1].set_title('Scaled RotorRPM Distribution')
axes[1].set_xlabel('RotorRPM (Standardized)')

plt.tight_layout()
plt.show()
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.histplot(df_imputed['RotorRPM'], ax=axes[0], kde=True, color='coral')
axes[0].set_title('Original RotorRPM Distribution')
axes[0].set_xlabel('RotorRPM')

sns.histplot(scaled_df['RotorRPM'], ax=axes[1], kde=True, color='mediumseagreen')
axes[1].set_title('Scaled RotorRPM Distribution')
axes[1].set_xlabel('RotorRPM (Standardized)')

plt.tight_layout()
plt.show()
import matplotlib.pyplot as plt
import seaborn as sns

numeric_data = df_imputed.select_dtypes(include=['float64', 'int64'])

correlation_matrix = numeric_data.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', square=True)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()
numeric_data = df_imputed.select_dtypes(include='number')  

target = 'ActivePower'  

correlations = numeric_data.corr()[target].abs().sort_values(ascending=False)

top_features = correlations.drop(target).head(5).index.tolist()

print("Selected Features for Modeling:", top_features)
from sklearn.model_selection import train_test_split

X = df_imputed[top_features]  
y = df_imputed[target]        

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training features shape:", X_train.shape)
print("Test features shape:", X_test.shape)
print("Training labels shape:", y_train.shape)
print("Test labels shape:", y_test.shape)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import pandas as pd

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "KNN": KNeighborsRegressor()
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    results.append({
        'Model': name,
        'MAE': mae,
        'MSE': mse,
        'R2': r2,
        'MAPE': mape
    })

results_df = pd.DataFrame(results)

# Display
print(results_df)
import matplotlib.pyplot as plt
import math

sample_range = 100

n_models = len(models)
cols = 2
rows = math.ceil(n_models / cols)

# Create figure
plt.figure(figsize=(14, rows * 4))

for i, (name, model) in enumerate(models.items(), start=1):
    preds = model.predict(X_test)

    plt.subplot(rows, cols, i)
    plt.plot(y_test.values[:sample_range], label='Actual', marker='o', linewidth=1)
    plt.plot(preds[:sample_range], label='Predicted', marker='x', linewidth=1)
    
    plt.title(f"{name}", fontsize=12, weight='bold')
    plt.xlabel("Sample Index")
    plt.ylabel("Active Power")
    plt.legend()
    plt.grid(True)

plt.suptitle(" Actual vs Predicted Power (First 100 Samples)", fontsize=16, weight='bold', y=1.02)
plt.tight_layout()
plt.show()
r2_key = next((col for col in results_df.columns if 'r2' in col.lower()), None)

if r2_key:
    best_by_mae = results_df.loc[results_df['MAE'].idxmin()]
    best_by_mse = results_df.loc[results_df['MSE'].idxmin()]
    best_by_r2 = results_df.loc[results_df[r2_key].idxmax()]
    best_by_mape = results_df.loc[results_df['MAPE'].idxmin()]

    print(" Best Model by MAE:", best_by_mae['Model'])
    print(" Best Model by MSE:", best_by_mse['Model'])
    print(f" Best Model by {r2_key}:", best_by_r2['Model'])
    print(" Best Model by MAPE:", best_by_mape['Model'])
else:
    print(" 'R²' or similar column not found in the results.")
import matplotlib.pyplot as plt

best_model = models['KNN']  
y_pred_knn = best_model.predict(X_test)

plt.figure(figsize=(12, 6))
plt.plot(y_test.values[:100], label='Actual', marker='o', linestyle='-')
plt.plot(y_pred_knn[:100], label='Predicted', marker='x', linestyle='--')
plt.title("KNN Model - Actual vs Predicted (First 100 Samples)")
plt.xlabel("Sample Index")
plt.ylabel("Active Power")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

knn_pred = y_pred_knn

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=knn_pred, alpha=0.4, label='Predicted Points', color='mediumblue')
sns.lineplot(x=y_test, y=y_test, color='red', label='Ideal Fit Line', linewidth=2)
sns.regplot(x=y_test, y=knn_pred, scatter=False, color='green', label='Best Fit Line', ci=None)

# Metrics
knn_r2 = r2_score(y_test, knn_pred)
knn_mae = mean_absolute_error(y_test, knn_pred)
knn_mape = np.mean(np.abs((y_test - knn_pred) / y_test)) * 100

# Plot Labels
plt.title("K-Nearest Neighbors: Predicted vs Actual")
plt.xlabel("Actual Active Power")
plt.ylabel("Predicted Active Power")
plt.legend()
plt.grid(True)

# Add metric annotations
plt.text(
    x=np.percentile(y_test, 80),
    y=np.percentile(knn_pred, 35),  
    s=f"R²: {knn_r2:.4f}\nMAE: {knn_mae:.2f}\nMAPE: {knn_mape:.2f}%",
    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black", lw=1),
    fontsize=11,
    ha='left',
    va='top',
    color='black'
)

plt.tight_layout()
plt.show()
from sklearn.neighbors import KNeighborsRegressor
import joblib

best_model = KNeighborsRegressor()
best_model.fit(X_train, y_train)

joblib.dump(best_model, "best_knn_model.pkl")
print("KNN model saved as 'best_knn_model.pkl'")
results_df.to_csv("model_comparison_report.csv", index=False)
print("Report saved as model_comparison_report.csv")