import streamlit as st
import pandas as pd
import pickle

# Load the saved model
with open('bank_churn_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Function to predict churn
def predict_churn(features):
    # Create a DataFrame with the input features
    input_df = pd.DataFrame([features])
    # Use the model to make predictions
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)
    return prediction[0], probability[0][1]

# Main function to run the Streamlit app
def main():
    st.title('Bank Churn Prediction')

    # Input features
    st.sidebar.header('Input Features')
    credit_score = st.sidebar.slider('Credit Score', 300, 850, 650)
    age = st.sidebar.slider('Age', 18, 100, 35)
    tenure = st.sidebar.slider('Tenure', 0, 20, 5)
    balance = st.sidebar.slider('Balance', 0, 250000, 50000)
    num_of_products = st.sidebar.slider('Number of Products', 1, 4, 2)
    has_credit_card = st.sidebar.selectbox('Has Credit Card', ['No', 'Yes'])
    is_active_member = st.sidebar.selectbox('Is Active Member', ['No', 'Yes'])
    estimated_salary = st.sidebar.slider('Estimated Salary', 0, 200000, 100000)
    geography = st.sidebar.selectbox('Geography', ['France', 'Germany', 'Spain'])
    gender = st.sidebar.selectbox('Gender', ['Female', 'Male'])

    # Convert categorical features to numeric
    gender = 1 if gender == 'Male' else 0
    geography = 0 if geography == 'France' else 1 if geography == 'Germany' else 2
    has_credit_card = 1 if has_credit_card == 'Yes' else 0
    is_active_member = 1 if is_active_member == 'Yes' else 0

    # Make prediction
    features = [credit_score, geography, gender, age, tenure, balance, num_of_products, has_credit_card, is_active_member, estimated_salary]
    prediction, probability = predict_churn(features)

    # Display prediction
    st.subheader('Prediction')
    if prediction == 0:
        st.write('The customer is likely to stay with the bank.')
    else:
        st.write('The customer is likely to churn from the bank.')

    # Display probability
    st.subheader('Churn Probability')
    st.write(f'The probability of churn is {probability:.2f}')

if __name__ == '__main__':
    main()
