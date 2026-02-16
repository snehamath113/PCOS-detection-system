import pickle
import numpy as np

# Load model
with open("model/pcos_model.pkl", "rb") as file:
    model = pickle.load(file)

# Sample input
# Age, BMI, Insulin, Testosterone, FSH, LH
input_data = np.array([[26, 30.5, 17.8, 1.0, 5.1, 10.0]])

prediction = model.predict(input_data)

if prediction[0] == 1:
    print("PCOS Detected")
else:
    print("No PCOS Detected")
