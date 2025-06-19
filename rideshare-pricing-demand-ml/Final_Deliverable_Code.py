# Load the Datasets
# Step 1
# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import folium
from folium.plugins import HeatMap
import numpy as np
from tabulate import tabulate
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, make_scorer, precision_score, recall_score, roc_auc_score, roc_curve, auc, ConfusionMatrixDisplay, mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDClassifier
from prophet import Prophet
from pmdarima import auto_arima
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline


# Load rideshare dataset
rideshare = pd.read_csv("rideshare_kaggle.csv")

# Load income dataset
income = pd.read_csv("us_income_zipcode.csv")

# Load the zipcode and location datase
location = pd.read_excel("ZIP_Locale_Detail.xls")

# Step 2: Prepare and clean location data
location_clean = location.rename(columns={
    'LOCALE NAME': 'location_name',
    'DELIVERY ZIPCODE': 'zipcode'
})
location_clean['location_name'] = location_clean['location_name'].str.strip().str.lower()
location_clean['zipcode'] = location_clean['zipcode'].astype(str).str.zfill(5)
location_lookup = location_clean[['location_name', 'zipcode']].drop_duplicates()

# Step 3: Clean rideshare source/destination and drop old ZIPs if any
rideshare = rideshare.copy()
rideshare = rideshare.drop(columns=[col for col in rideshare.columns if 'zip' in col], errors='ignore')
rideshare['source_clean'] = rideshare['source'].str.strip().str.lower()
rideshare['destination_clean'] = rideshare['destination'].str.strip().str.lower()

# Step 4: Map ZIP codes to source/destination
rideshare = rideshare.merge(location_lookup, how='left', left_on='source_clean', right_on='location_name')
rideshare = rideshare.rename(columns={'zipcode': 'source_zip'})
rideshare = rideshare.drop(columns=['location_name'])

rideshare = rideshare.merge(location_lookup, how='left', left_on='destination_clean', right_on='location_name')
rideshare = rideshare.rename(columns={'zipcode': 'destination_zip'})
rideshare = rideshare.drop(columns=['location_name'])

# Step 5: Prepare income dataset
income['zipcode'] = income['ZIP'].astype(str).str.zfill(5)

# Step 6: Merge income info with source ZIP (avoid col name conflict)
rideshare = rideshare.drop(columns=['zipcode'], errors='ignore')
merged = rideshare.merge(
    income[['zipcode', 'Households Median Income (Dollars)']],
    how='left',
    left_on='source_zip',
    right_on='zipcode'
)

# Step 7: Verify the merge
print("Total rides:", len(merged))
print("Matched income rows:", merged['Households Median Income (Dollars)'].notnull().sum())

# Reduce dataset size by 90% using random sampling
merged = merged.sample(frac=0.1, random_state=42).reset_index(drop=True)
# Confirm new size
print("Sampled dataset size:", len(merged))

# Price vs. Income Bracket Visualization
# Step 1: Prepare and clean the data
bracket_data = merged[['price', 'Households Median Income (Dollars)']].dropna()

# Step 2: Define quantile-based bins and capture the bin ranges
bracket_data['income_bracket'], bin_edges = pd.qcut(
    bracket_data['Households Median Income (Dollars)'],
    q=5,
    retbins=True
)

# Step 3: Format income range labels
labels = [
    f"${int(bin_edges[i]):,} - ${int(bin_edges[i + 1]):,}"
    for i in range(len(bin_edges) - 1)
]

bracket_data['income_bracket'] = pd.cut(
    bracket_data['Households Median Income (Dollars)'],
    bins=bin_edges,
    labels=labels,
    include_lowest=True
)

# Step 4: Calculate average price per bracket
avg_prices = bracket_data.groupby('income_bracket', observed=False)['price'].mean().reset_index()

# Step 5: Plot
plt.figure(figsize=(10, 6))
sns.set(style="whitegrid")
bar = sns.barplot(
    data=avg_prices,
    x='income_bracket',
    y='price',
    palette='Blues_d'
)

for index, row in avg_prices.iterrows():
    bar.text(index, row['price'] + 0.2, f"${row['price']:.2f}", ha='center', fontsize=10)

# Title
plt.title('Average Ride Price by Household Income Range', fontsize=14, fontweight='bold')

# Surge Rate by Income Level

# Step 1: Create surge flag
merged['is_surge'] = merged['surge_multiplier'] > 1

# Step 2: Drop missing values
surge_data = merged[['is_surge', 'Households Median Income (Dollars)']].dropna()

# Step 3: Bin income into quantile-based brackets
surge_data['income_bracket'], bin_edges = pd.qcut(
    surge_data['Households Median Income (Dollars)'],
    q=5,
    retbins=True
)

# Step 4: Create cleaner labels
labels = [
    f"${int(bin_edges[i]):,} – ${int(bin_edges[i+1]):,}"
    for i in range(len(bin_edges)-1)
]
surge_data['income_bracket'] = pd.qcut(
    surge_data['Households Median Income (Dollars)'],
    q=5,
    labels=labels
)

# Step 5: Calculate surge rates
surge_rates = surge_data.groupby('income_bracket', observed=True)['is_surge'].mean().reset_index()

# Step 6: Plot
plt.figure(figsize=(10, 6))
sns.set(style="whitegrid")
barplot = sns.barplot(
    data=surge_rates,
    x='income_bracket',
    y='is_surge',
    palette='Blues_d',
    legend=False
)

# Add percentage labels on top of each bar
for i, val in enumerate(surge_rates['is_surge']):
    barplot.text(i, val + 0.001, f"{val:.2%}", ha='center', va='bottom', fontsize=10)

# Customize chart appearance
plt.title('Surge Rate by Household Income Range', fontsize=14, fontweight='bold')
plt.xlabel('Household Income Range', fontsize=12)
plt.ylabel('Proportion of Rides with Surge Pricing', fontsize=12)
plt.xticks(rotation=15)
plt.ylim(0, surge_rates['is_surge'].max() + 0.01)
plt.tight_layout()
plt.show()

# Spatial Surge Analysis Using Coordinates
# Step 1: Filter for surge pricing rides
surge_rides = merged[merged['surge_multiplier'] > 1]

# Drop rows with missing or zero coordinates
surge_rides = surge_rides.dropna(subset=['latitude', 'longitude'])
surge_rides = surge_rides[(surge_rides['latitude'] != 0) & (surge_rides['longitude'] != 0)]

# Step 2: Create map centered in Boston
surge_map = folium.Map(location=[42.3601, -71.0589], zoom_start=12)

# Step 3: Prepare data for HeatMap
heat_data = surge_rides[['latitude', 'longitude']].values.tolist()

# Step 4: Add heatmap layer
HeatMap(heat_data, radius=10, blur=15, max_zoom=13).add_to(surge_map)

# Step 5: Show map
surge_map

# Temporal Patterns: Table and Heatmap of Hour vs Day vs Average Surge Rate
# Step 1: Convert datetime column to actual datetime format
merged['datetime'] = pd.to_datetime(merged['datetime'])

# Step 2: Extract actual day of the week
merged['weekday'] = merged['datetime'].dt.weekday  # 0 = Monday, 6 = Sunday

# Step 3: Define day type
merged['day_type'] = merged['weekday'].apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')

# Step 4: Group and calculate average ride price
day_price = merged.groupby('day_type')['price'].mean().reset_index()
day_price['price'] = day_price['price'].round(2)
day_price['price'] = day_price['price'].apply(lambda x: f"${x:.2f}")

# Step 5: Display the table
print("\n")
print(tabulate(day_price, headers=["Day Type", "Average Ride Price"], tablefmt="github"))

# Create binary surge indicator
merged['is_surge'] = merged['surge_multiplier'] > 1

# Ensure 'datetime' is a datetime type
merged['datetime'] = pd.to_datetime(merged['datetime'])

# Extract actual date and weekday label
merged['day_date'] = merged['datetime'].dt.date
merged['day_of_week'] = merged['datetime'].dt.strftime('%a (%m/%d)')  # e.g., Mon (12/01)

# Create pivot table: Hour vs Day Label
pivot_data = merged.pivot_table(
    index='hour',
    columns='day_of_week',
    values='is_surge',
    aggfunc='mean'
)

# Reorder columns by actual date
ordered_cols = sorted(pivot_data.columns, key=lambda x: pd.to_datetime(x.split('(')[-1][:-1], format="%m/%d"))
pivot_data = pivot_data[ordered_cols]

# Plot heatmap
plt.figure(figsize=(14, 7))
sns.set(style='whitegrid')
ax = sns.heatmap(
    pivot_data,
    cmap='Reds',
    annot=True,
    fmt=".2f",
    linewidths=0.5,
    cbar_kws={'label': 'Surge Rate'}
)

plt.title('Surge Frequency: Hour of Day vs Day of Week 2018', fontsize=14, fontweight='bold')
plt.xlabel('Day of Week')
plt.ylabel('Hour of Day')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Compare Average Price by Weather Type
# Group by weather condition and calculate average ride price
weather_price = merged.groupby('short_summary')['price'].mean().reset_index()

# Round the prices for presentation
weather_price['price'] = weather_price['price'].round(2).apply(lambda x: f"${x:.2f}")

# Sort by price
weather_price = weather_price.sort_values(by='price', ascending=False)

# Display as table
print("\n")
print(tabulate(
    weather_price.reset_index(drop=True),
    headers=["Weather Condition", "Average Ride Price ($)"],
    tablefmt="github"
))

# Surge Frequency by Weather Condition
# Step 1: Create binary surge flag
merged['is_surge'] = merged['surge_multiplier'] > 1

# Step 2: Group by weather condition and calculate surge rate
weather_surge = merged.groupby('short_summary')['is_surge'].mean().reset_index()

# Step 3: Sort and format the result
weather_surge['is_surge'] = (weather_surge['is_surge'] * 100).round(2).apply(lambda x: f"{x:.2f}%")
weather_surge = weather_surge.sort_values(by='is_surge', ascending=False)

# Step 4: Display as table
print("\n")
print(tabulate(
    weather_surge.reset_index(drop=True),
    headers=["Weather Condition", "Surge Frequency (%)"],
    tablefmt="github"
))

# Average Ride Price by Temperature Range
# Step 1: Bin temperature into intervals
merged['temp_bin'] = pd.cut(
    merged['temperature'],
    bins=[-10, 30, 50, 70, 90, 110],
    labels=['<30°F', '30–50°F', '50–70°F', '70–90°F', '90–110°F']
)

# Step 2: Group and clean
temp_price_summary = (
    merged.groupby('temp_bin')['price']
    .mean()
    .round(2)
    .dropna()
    .reset_index()
)

# Step 3: Plot line chart
plt.figure(figsize=(8, 5))
sns.set(style="whitegrid")

plt.plot(temp_price_summary['temp_bin'], temp_price_summary['price'], marker='o', linewidth=2, color='teal')
for i, row in temp_price_summary.iterrows():
    plt.text(i, row['price'] + 0.03, f"${row['price']:.2f}", ha='center', fontsize=9)

plt.title('Trend of Average Ride Price by Temperature Range', fontsize=14, fontweight='bold')
plt.xlabel('Temperature Range (°F)', fontsize=12)
plt.ylabel('Average Ride Price ($)', fontsize=12)
plt.ylim(temp_price_summary['price'].min() - 0.1, temp_price_summary['price'].max() + 0.2)
plt.tight_layout()
plt.show()

# -------------------------------
# Weather → Ride Price Regression
# -------------------------------
# Step 1: Select relevant weather + price data
regression_features = ['temperature', 'humidity', 'windSpeed', 'precipProbability']
regression_data = merged[regression_features + ['price']].dropna()

# Explore correlation between weather variables and price
print(regression_data.corr())
sns.heatmap(regression_data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix: Weather & Price")
plt.tight_layout()
plt.show()

# Log-transform price to reduce skewness
regression_data['log_price'] = np.log1p(regression_data['price'])

# Use the log-transformed price as target
X_reg = regression_data[regression_features]
y_reg = regression_data['log_price']

# Step 2: Train-test split
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Step 3: Train model
linreg = LinearRegression()
linreg.fit(X_train_reg, y_train_reg)

# Step 4: Predict and evaluate
y_pred_reg = linreg.predict(X_test_reg)

y_pred_log = linreg.predict(X_test_reg)
y_pred_actual = np.expm1(y_pred_log)
y_test_actual = np.expm1(y_test_reg)  # also transform test labels back

mae = mean_absolute_error(y_test_actual, y_pred_actual)
rmse = mean_squared_error(y_test_actual, y_pred_actual, squared=False)
mse = mean_squared_error(y_test_actual, y_pred_actual)
r2 = r2_score(y_test_actual, y_pred_actual)

print("\nWeather → Price Linear Regression Metrics")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MSE: {mse:.2f}")
print(f"R²: {r2:.2f}")

# -------------------------------
# Distance → Ride Price Regression
# -------------------------------
# Select distance and price
distance_corr_data = merged[['distance', 'price']].dropna()

# Generate and print correlation matrix
print(distance_corr_data.corr())

# Plot heatmap
sns.heatmap(distance_corr_data.corr(), annot=True, cmap="Blues")
plt.title("Correlation Matrix: Distance & Price")
plt.tight_layout()
plt.show()

# Step 1: Prepare the data
distance_data = merged[['distance', 'price']].dropna()
distance_data['log_price'] = np.log1p(distance_data['price'])

X_dist = distance_data[['distance']]
y_dist_log = distance_data['log_price']
y_dist_actual = distance_data['price']

# Step 2: Train-test split
X_train_dist, X_test_dist, y_train_dist_log, y_test_dist_log = train_test_split(X_dist, y_dist_log, test_size=0.2, random_state=42)

# Step 3: Linear Regression
linreg_dist = LinearRegression()
linreg_dist.fit(X_train_dist, y_train_dist_log)
y_pred_dist_log_lr = linreg_dist.predict(X_test_dist)
y_pred_dist_lr = np.expm1(y_pred_dist_log_lr)
y_test_dist_actual = np.expm1(y_test_dist_log)

# Step 4: Gradient Boosting Regressor
gbr_dist = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gbr_dist.fit(X_train_dist, y_train_dist_log)
y_pred_dist_log_gbr = gbr_dist.predict(X_test_dist)
y_pred_dist_gbr = np.expm1(y_pred_dist_log_gbr)

# Step 5: Evaluation Function
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)

    print(f"\n{model_name} Performance (Distance → Price):")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.2f}")

# Step 6: Evaluate both models
evaluate_model(y_test_dist_actual, y_pred_dist_lr, "Linear Regression")
evaluate_model(y_test_dist_actual, y_pred_dist_gbr, "Gradient Boosting Regressor")

# -------------------------------------
# All Features → Ride Price Regression
# -------------------------------------
# Step 1: Select relevant features and target
regression_features_all = [
    'distance', 'hour', 'temperature', 'humidity',
    'windSpeed', 'precipProbability', 'Households Median Income (Dollars)'
]

regression_data_all = merged[regression_features_all + ['price']].dropna()

# Remove extreme price outliers (adjust if needed)
regression_data_all = regression_data_all[regression_data_all['price'] <= 100]

# Step 2: Define X and y
X_all = regression_data_all[regression_features_all]
y_all = regression_data_all['price']

# Step 3: Train-test split
X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

# Step 4: Linear Regression model
linreg = LinearRegression()
linreg.fit(X_train_all, y_train_all)
y_pred_all = linreg.predict(X_test_all)

# Step 5: Evaluate the model
mae = mean_absolute_error(y_test_all, y_pred_all)
mse = mean_squared_error(y_test_all, y_pred_all)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_all, y_pred_all)

print("\nLinear Regression Performance (All Features → Price):")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")

# Step 5: Plot actual vs predicted prices
plt.figure(figsize=(8, 6))
plt.scatter(y_test_all, y_pred_all, alpha=0.4, color='royalblue', edgecolors='k')
plt.plot([y_test_all.min(), y_test_all.max()], [y_test_all.min(), y_test_all.max()], color='red', linestyle='--')
plt.xlabel('Actual Ride Price ($)', fontsize=12)
plt.ylabel('Predicted Ride Price ($)', fontsize=12)
plt.title('Linear Regression: Actual vs Predicted Ride Prices', fontsize=14, fontweight='bold')
plt.grid(True)
plt.tight_layout()
plt.show()

# =====================
# Model Pipeline
# =====================
# Step 1: Prepare Features and Target
merged['is_surge'] = (merged['surge_multiplier'] > 1).astype(int)
features = ['distance', 'hour', 'temperature', 'Households Median Income (Dollars)']
df_ml = merged[features + ['is_surge']].dropna()
X = df_ml[features]
y = df_ml['is_surge']

# Step 2: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Step 3: Apply SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
print("\nBalanced Training Set Class Distribution:\n", pd.Series(y_train_balanced).value_counts())

# Step 4: Scale + PCA once
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)

X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, stratify=y, test_size=0.2, random_state=42)

# Step 5: Define and Train Optimized Models
models = {
    "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=200, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(64,), max_iter=200, early_stopping=True, random_state=42)
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Report Mean Cross-Validation Score for Random Forest
cv_scores = cross_val_score(models["Random Forest"], X, y, cv=cv, scoring='f1')
print(f"\nMean F1 Score (3-fold CV): {cv_scores.mean():.4f}")

scorer = f1_score

for name, model in models.items():
    print(f"\n{name} Training & Evaluation:")

    # Use SMOTE-balanced data for imbalance-sensitive models
    if name in ["Gradient Boosting", "Neural Network", "Logistic Regression"]:
        model.fit(X_train_balanced, y_train_balanced)
    else:
        model.fit(X_train, y_train)

    # Threshold tuning block
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test)[:, 1]
        y_pred = (y_scores >= 0.5).astype(int)
    else:
        y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, zero_division=0))
    print("Recall:", recall_score(y_test, y_pred, zero_division=0))
    print("F1 Score:", f1_score(y_test, y_pred, zero_division=0))
    print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))
    print("ROC AUC:", roc_auc_score(y_test, y_scores))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues', ax=ax, values_format='d')
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    plt.title(f"Confusion Matrix: {name}")
    plt.tight_layout()
    plt.show()

    if name == "Random Forest":
        importances = model.feature_importances_
        feature_importance = pd.DataFrame({
            "Feature": X.columns,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        print("\nTop Features by Importance (Random Forest):")
        print(feature_importance)

        # Visualize Feature Importance
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_importance.head(10), x="Importance", y="Feature")
        plt.title("Top 10 Feature Importances (Random Forest)", fontsize=14, fontweight='bold')
        plt.xlabel("Importance Score")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.show()

# SVM with PCA features

# Step 1: Resample the training PCA data
smote = SMOTE(random_state=42)
X_train_pca_resampled, y_train_pca_resampled = smote.fit_resample(X_train_pca, y_train_pca)

# Step 2: Define LinearSVC
linear_svm = LinearSVC(random_state=42, max_iter=2000, dual=False)

# Step 3: Wrap it in CalibratedClassifierCV
calibrated_svm = CalibratedClassifierCV(estimator=linear_svm, cv=3)

# Step 4: Fit the calibrated SVM
calibrated_svm.fit(X_train_pca_resampled, y_train_pca_resampled)

# Step 5: Predict
y_scores_svm = calibrated_svm.predict_proba(X_test_pca)[:, 1]
y_pred_svm = (y_scores_svm >= 0.5).astype(int)

# Step 6: Evaluation
print("\nSVM (LinearSVC + Calibration + SMOTE) Evaluation:")
print("Accuracy:", accuracy_score(y_test_pca, y_pred_svm))
print("Precision:", precision_score(y_test_pca, y_pred_svm, zero_division=0))
print("Recall:", recall_score(y_test_pca, y_pred_svm, zero_division=0))
print("F1 Score:", f1_score(y_test_pca, y_pred_svm, zero_division=0))
print("Classification Report:\n", classification_report(y_test_pca, y_pred_svm, zero_division=0))
print("ROC AUC:", roc_auc_score(y_test_pca, y_scores_svm))

# Confusion Matrix
cm_svm = confusion_matrix(y_test_pca, y_pred_svm)
disp_svm = ConfusionMatrixDisplay(confusion_matrix=cm_svm)
disp_svm.plot(cmap='Blues')
plt.title("Confusion Matrix: LinearSVC + Calibration + SMOTE")
plt.tight_layout()
plt.show()

param_grid_rf = {
    'n_estimators': [50, 100],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}

# Use 3-fold CV for speed
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Define model and GridSearch
rf = RandomForestClassifier(random_state=42)

random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_grid_rf,
                                   n_iter=5, scoring='f1', cv=cv, random_state=42, n_jobs=-1)
random_search.fit(X_train, y_train)

print("\nBest Params (RandomizedSearchCV):", random_search.best_params_)
print("Best F1 Score from RandomizedSearchCV:", random_search.best_score_)

# Step 6: ROC Curves for all models
plt.figure(figsize=(10, 6))
for name, model in models.items():
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test)[:, 1]
    else:
        y_scores = model.decision_function(X_test)

    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

# Plot settings
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve for All Models")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# ===========================
# Forecasting Ride Demand
# ===========================

# === ARIMA MODEL ===
print("ARIMA Model: Hourly Ride Demand Forecast")

# Step 1: Prepare hourly ride count time series
merged['datetime'] = pd.to_datetime(merged['datetime'])
hourly_demand = merged.set_index('datetime').resample('H').size().asfreq('H').fillna(0)

# Step 2: Train-test split (last 24 hours as test)
train = hourly_demand[:-24]
test = hourly_demand[-24:]

# Step 3: Train Auto ARIMA model
arima_model = auto_arima(train, seasonal=False, trace=True, error_action='ignore', suppress_warnings=True)
arima_forecast = arima_model.predict(n_periods=24)

# Step 4: Evaluate ARIMA
arima_rmse = mean_squared_error(test, arima_forecast, squared=False)
arima_mape = mean_absolute_percentage_error(test, arima_forecast)

print(f"ARIMA RMSE: {arima_rmse:.2f}")
print(f"ARIMA MAPE: {arima_mape:.2%}")

# === PROPHET MODEL ===
print("\n Prophet Model: 3-Day Ride Demand Forecast")

# Step 1: Re-aggregate data for Prophet
ride_counts = (
    merged.groupby(pd.Grouper(key='datetime', freq='H'))
    .size()
    .reset_index()
    .rename(columns={'datetime': 'ds', 0: 'y'})
)

# Step 2: Train Prophet
prophet_model = Prophet()
prophet_model.fit(ride_counts)

# Step 3: Forecast next 72 hours
future = prophet_model.make_future_dataframe(periods=72, freq='H')
forecast = prophet_model.predict(future)

# Step 4: Plot
fig1 = prophet_model.plot(forecast)
plt.title("Prophet Forecast: Next 3 Days of Ride Demand", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Step 5: Prophet Evaluation (backtest on train data)
actual_vs_pred = ride_counts.merge(forecast[['ds', 'yhat']], on='ds')
prophet_rmse = mean_squared_error(actual_vs_pred['y'], actual_vs_pred['yhat'], squared=False)
prophet_mape = mean_absolute_percentage_error(actual_vs_pred['y'], actual_vs_pred['yhat'])

print(f"Prophet RMSE: {prophet_rmse:.2f}")
print(f"Prophet MAPE: {prophet_mape:.2%}")
