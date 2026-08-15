import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


df = pd.read_csv("netflix_imdb_modeling_V1.csv", low_memory=False)
df = df[(df["netflix_season_number"] == 1) &
    (df["target_next_season_views"].notna()) &
    (df["netflix_views"] > 0)].copy()
df["retention"] = (df["target_next_season_views"]/ df["netflix_views"])


features = ["imdb_average_rating", "imdb_num_votes"]
df = df.dropna(subset=features + ["retention"])

X = df[features]
y = df["retention"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

###Model Evaluation###

mae = mean_absolute_error( y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))


print("Mean Absolute Error:", round(mae * 100, 2), "percentage points")
print("Root Mean Squared Error:", round(rmse * 100, 2),"percentage points")


###Baseline Evaluation###

median_retention = y_train.median()
baseline_predictions = np.full(len(y_test), median_retention)
baseline_mae = mean_absolute_error(y_test, baseline_predictions)
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_predictions))


print("\nBaseline Results")
print("Median Retention:", round(median_retention * 100, 2), "%")
print("Baseline MAE:", round(baseline_mae * 100, 2), "percentage points")
print("Baseline RMSE:", round(baseline_rmse * 100, 2), "percentage points")


#Comparing model to baseline
improvement = ((baseline_mae - mae)/ baseline_mae) * 100
print("\nImprovement over baseline:", round(improvement, 2), "%")


print("\nCoefficients:")
for feature, coefficient in zip(features, model.coef_):
    print(feature, ":", coefficient)

print("Intercept:", model.intercept_)
print("Shows over 100% retention:", (df["retention"] > 1).sum())
print("Shows over 200% retention:", (df["retention"] > 2).sum())

plt.hist(df["retention"], bins=50)
plt.xlabel("Season 2 / Season 1 Retention")
plt.ylabel("Number of Series")
plt.title("Distribution of Viewer Retention")
plt.show()

plt.hist(df[df["retention"] <= 2]["retention"], bins=40)
plt.xlabel("Season 2 / Season 1 Retention")
plt.ylabel("Number of Series")
plt.title("Viewer Retention Distribution (0–200%)")
plt.show()
