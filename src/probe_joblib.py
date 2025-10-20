import joblib
import matplotlib.pyplot as plt

# Load the saved model dictionary
model_dict = joblib.load("precision_prediction_model.joblib")

# Extract features and their importances
features = model_dict["features"]
importances = model_dict["feature_importances"]

# Print them out
for feat, imp in zip(features, importances):
    print(f"{feat}: {imp}")

# Plot the feature importances
plt.figure(figsize=(6,4))
plt.bar(features, list(importances.values()))
plt.ylabel("Importance")
plt.title("Feature Importances in DecisionTreeRegressor")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plot.pdf")
