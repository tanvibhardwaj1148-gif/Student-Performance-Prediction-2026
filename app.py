import streamlit as st
import pandas as pd
import joblib

# Page title
st.title("Student Performance Prediction System")

st.write("Predict student final marks using the trained Machine Learning model.")

# Load trained model
model = joblib.load("Student_performance_model.pkl")

# Load dataset
df = pd.read_csv("Student_data.csv")

st.subheader("Student Dataset")

# Show dataset
st.dataframe(df.head())

st.subheader("Make Prediction")

# Select a student from the dataset
student_index = st.number_input(
    "Enter Student Index (0 to 394)",
    min_value=0,
    max_value=len(df) - 1,
    value=0
)

if st.button("Predict Final Marks"):

    # Encode dataset
    df_encoded = pd.get_dummies(df, drop_first=True)

    # Remove target column
    X = df_encoded.drop("G3", axis=1)

    # Select student
    student_data = X.iloc[[student_index]]

    # Predict
    prediction = model.predict(student_data)

    # Actual marks
    actual_marks = df.iloc[student_index]["G3"]

    st.success(
        f"Predicted Final Marks: {prediction[0]:.2f}"
    )

    st.info(
        f"Actual Final Marks: {actual_marks}"
    )
