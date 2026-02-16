import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
data = pd.read_csv("dataset/pcos_dataset.csv")

X = data.drop("PCOS", axis=1)
y = data["PCOS"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Save model
with open("model/pcos_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model trained and saved successfully!")
