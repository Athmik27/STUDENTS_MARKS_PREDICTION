
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Sample dataset
data = {
    "Hours": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    "Marks": [15, 20, 35, 40, 50, 60, 65, 80, 90]
}

df = pd.DataFrame(data)

# Features and target
X = df[["Hours"]]
y = df["Marks"]

# Train the model
model = LinearRegression()
model.fit(X, y)

# Predict marks
hours = float(input("Enter study hours: "))
prediction = model.predict(pd.DataFrame({"Hours": [hours]}))

print(f"Predicted marks: {prediction[0]:.2f}")

# Visualize the data
plt.scatter(X["Hours"], y, label="Actual marks")
plt.plot(X["Hours"], model.predict(X), color="red",
         label="Prediction line")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Student Marks Prediction")
plt.legend()
plt.savefig("prediction.png")
plt.show()
