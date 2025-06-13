import streamlit as st
import pandas as pd
import pickle

# Title of the app
st.title("Titanic Survival Prediction App")
st.write("Predict survival chances based on passenger details.")

# Load the trained model
@st.cache_resource
def load_model():
    with open('titanic_model.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()

# User inputs for prediction
st.sidebar.header("Passenger Details")
Pclass = st.sidebar.selectbox("Passenger Class", options=[1, 2, 3], index=2)
Age = st.sidebar.slider("Age", min_value=1, max_value=100, value=30)
SibSp = st.sidebar.number_input("Number of Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
Parch = st.sidebar.number_input("Number of Parents/Children Aboard", min_value=0, max_value=10, value=0)
Fare = st.sidebar.slider("Fare", min_value=0.0, max_value=500.0, value=50.0, step=0.1)
Sex_male = st.sidebar.radio("Gender", options=["Male", "Female"]) == "Male"
Embarked_Q = st.sidebar.radio("Embarked at Q?", options=["No", "Yes"]) == "Yes"
Embarked_S = st.sidebar.radio("Embarked at S?", options=["No", "Yes"]) == "Yes"

# Prepare input data
def prepare_input():
    return pd.DataFrame({
        'PassengerId': [0],  # Add a placeholder if not relevant, or drop from the model training
        'Pclass': [Pclass],
        'Age': [Age],
        'SibSp': [SibSp],
        'Parch': [Parch],
        'Fare': [Fare],
        'male': [1 if Sex_male else 0],
        'Q': [1 if Embarked_Q else 0],
        'S': [1 if Embarked_S else 0]
    })



input_data = prepare_input()

# Prediction
if st.button("Predict Survival"):
    prediction = model.predict(input_data)
    st.write("### Prediction:")
    if prediction[0] == 1:
        st.success("The passenger is likely to survive.")
    else:
        st.error("The passenger is unlikely to survive.")
