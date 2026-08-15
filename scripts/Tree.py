import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (balanced_accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay)


df = pd.read_csv("netflix_imdb_modeling_V1.csv", low_memory=False)
df = df[(df["netflix_season_number"] == 1) &
    (df["target_next_season_views"].notna()) &
    (df["netflix_views"] > 0)].copy()
df["retention"] = (df["target_next_season_views"]/ df["netflix_views"])

#0 is High retention (70% or higher)
#1 is Low retention (below 70%)
df["low_retention"] = (df["retention"] < 0.70).astype(int)

features = ["imdb_average_rating", "imdb_num_votes"]
df = df.dropna(subset=features + ["low_retention"])

X = df[features]
y = df["low_retention"]
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.30, random_state=101, stratify=y)

print("Class Counts:")
print(y.value_counts())

print("\nClass Percentages:")
print(y.value_counts(normalize=True) * 100)

model = DecisionTreeClassifier( max_depth=3, min_samples_leaf=20, random_state=101)
model.fit(X_train, y_train)
predictions = model.predict(X_test)


###Model Evaluation###

balanced_accuracy = balanced_accuracy_score( y_test, predictions)
f1 = f1_score( y_test, predictions, pos_label=1)


print("\nDecision Tree Results")
print("Balanced Accuracy:", round(balanced_accuracy * 100, 2), "%")
print("F1 Score for Low Retention:", round(f1 * 100, 2), "%")


majority_class = y_train.mode()[0]
baseline_predictions = np.full(len(y_test), majority_class)
baseline_balanced_accuracy = balanced_accuracy_score( y_test, baseline_predictions)
baseline_f1 = f1_score(y_test, baseline_predictions, pos_label=1, zero_division=0)


print("\nBaseline Results")
print("Majority Class:", majority_class)
print("Baseline Balanced Accuracy:", round(baseline_balanced_accuracy * 100, 2), "%")
print("Baseline F1 Score:", round(baseline_f1 * 100, 2), "%")


cm = confusion_matrix(y_test, predictions)
print("\nConfusion Matrix:")
print(cm)

display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["70% or Higher", "Below 70%"])
display.plot()
plt.title("Decision Tree Confusion Matrix")
plt.show()
plt.figure(figsize=(16, 8))
plot_tree(model, feature_names=features, class_names=["70% or Higher", "Below 70%"],
    filled=True,
    rounded=True,
    fontsize=10)
plt.title("Decision Tree for Viewer Retention")
plt.show()

print("\nFeature Importance:")
for feature, importance in zip(features, model.feature_importances_):
    print(feature, ":", round(importance, 4))

