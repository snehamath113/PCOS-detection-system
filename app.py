import pickle
import numpy as np

with open("model/pcos_model.pkl", "rb") as file:
    model = pickle.load(file)

print("PCOS Detection System")

age = float(input("Enter Age: "))
bmi = float(input("Enter BMI: "))
insulin = float(input("Enter Insulin Level: "))
testosterone = float(input("Enter Testosterone Level: "))
fsh = float(input("Enter FSH Level: "))
lh = float(input("Enter LH Level: "))

input_data = np.array([[age, bmi, insulin, testosterone, fsh, lh]])

result = model.predict(input_data)

if result[0] == 1:
    print("Result: PCOS Detected")
else:
    print("Result: No PCOS Detected")
