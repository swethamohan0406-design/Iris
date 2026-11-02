
import streamlit as st
import joblib
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

model = joblib.load("iris_model.pkl")

st.title("Iris Flower Prediction")

# Load Iris dataset for the graph
iris = load_iris()
X = iris.data
y = iris.target

# Display graph
st.subheader("Iris Dataset Visualization")

plt.figure(figsize=(8, 5))

plt.scatter(X[y == 0, 2], X[y == 0, 3], label="Setosa")
plt.scatter(X[y == 1, 2], X[y == 1, 3], label="Versicolor")
plt.scatter(X[y == 2, 2], X[y == 2, 3], label="Virginica")

plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("Iris Flower Classification")
plt.legend()
plt.grid()

st.pyplot(plt)
plt.close()

# User input
st.subheader("Predict Your Flower")

sl = st.number_input("Sepal Length", min_value=0.0)
sw = st.number_input("Sepal Width", min_value=0.0)
pl = st.number_input("Petal Length", min_value=0.0)
pw = st.number_input("Petal Width", min_value=0.0)

if st.button("Predict"):
    pred = model.predict([[sl, sw, pl, pw]])
    flowers = ["Setosa", "Versicolor", "Virginica"]
    st.success(flowers[pred[0]])
