import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

# Training data
X = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(-1, 1)
y = np.array([0, 0, 0, 1, 1, 1, 1, 1])

# Create model
model = LogisticRegression()

# Train model
model.fit(X, y)
joblib.dump(model,"student_model.pkl")

# User input
hours = float(input("Enter study hours: "))

# Prediction
prediction = model.predict([[hours]])

# Output
if prediction[0] == 1:
    print("Result: PASS ✅")
else:
    print("Result: FAIL ❌")