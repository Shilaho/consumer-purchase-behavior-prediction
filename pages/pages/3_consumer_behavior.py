import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Set page title
st.set_page_config(page_title="Customer Behavior Prediction")

# Define function to load the dataset
@st.cache_data
def load_data(file):
    data = pd.read_csv(file)
    return data.copy()

# Define function to preprocess the data
def preprocess_data(data):
    # Use LabelEncoder to encode non-numeric string values in the 'gender' column
    le = LabelEncoder()
    data['gender'] = le.fit_transform(data['gender'])
    return data, le

# Define function to train the model
def train_model(data):
    # Split the dataset into input features and target variable
    X = data.drop('behavior', axis=1)
    y = data['behavior']

    # Train a Random Forest classifier on the dataset
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)

    return clf

# Define function to predict customer behavior
def predict_behavior(age, gender, income, credit_score, balance):
    # Encode the 'gender' value using the LabelEncoder
    gender = le.transform([gender])[0]

    # Create a DataFrame with the input features
    input_data = pd.DataFrame({'age': [age],
                               'gender': [gender],
                               'income': [income],
                               'credit_score': [credit_score],
                               'balance': [balance]})

    # Make the prediction using the trained model
    prediction = model.predict(input_data)

    return prediction[0]

# Define the app layout
st.title("Customer Behavior Prediction")
st.markdown("This app predicts whether a customer will exhibit a certain behavior based on their demographic and financial information.")

# Add a file uploader for the user to upload their own dataset
uploaded_file = st.file_uploader("Upload a CSV file containing customer data", type="csv")

if uploaded_file is not None:
    # Load the dataset and preprocess it
    data = load_data(uploaded_file)
    data, le = preprocess_data(data.copy())

    # Train the model
    model = train_model(data)

    # Add input widgets for the user to enter the customer information
    age = st.slider("Age", 18, 100, 25, 1)
    gender = st.selectbox("Gender", ["Male", "Female"])
    income = st.number_input("Annual Income", min_value=0, max_value=1000000, step=1000, value=50000)
    credit_score = st.slider("Credit Score", 300, 850, 650, 1)
    balance = st.number_input("Account Balance", min_value=0, max_value=1000000, step=1000, value=5000)

    # When the user clicks the "Predict" button, make the prediction and show the result
    if st.button("Predict"):
        prediction = predict_behavior(age, gender, income, credit_score, balance)
        if prediction == "Yes":
            st.markdown("This customer is likely to exhibit the purchase behavior.")
        else:
            st.markdown("This customer is unlikely to exhibit the purchase behavior.")

else:
    st.write("Please upload a CSV file.")