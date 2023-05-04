import streamlit as st

st.set_page_config(
    page_title="FAQ",
    page_icon="👋",
)


st.header('What is the Consumer Behavior Prediction App?')
q1_expander = st.expander('Click to reveal answer')
with q1_expander:
    st.write('The Consumer Behavior Prediction App is a machine learning application that can predict consumer behavior based on historical data and various demographic features.')

st.header('What kind of data can I use with this app?')
q2_expander = st.expander('Click to reveal answer')
with q2_expander:
    st.write('The app can handle various types of data, including demographic data (age, gender, income, etc.), behavioral data (past purchase history, clickstream data, etc.), and psychographic data (personality traits, values, interests, etc.).')

st.header('What kind of predictions can I make with this app?')
q3_expander = st.expander('Click to reveal answer')
with q3_expander:
    st.write('The app can make a wide range of predictions, such as the likelihood of a customer making a purchase, the customer lifetime value, and the most effective marketing channel to reach a particular customer.')

st.header('How accurate are the predictions?')
q4_expander = st.expander('Click to reveal answer')
with q4_expander:
    st.write('The accuracy of the predictions depends on the quality of the data used and the machine learning algorithms used in the app. We use state-of-the-art machine learning techniques and regularly update our algorithms to ensure the highest accuracy possible.')

st.header('How do I use the app?')
q5_expander = st.expander('Click to reveal answer')
with q5_expander:
    st.write('To use the app, simply upload your data to the app, select the prediction you want to make, and run the app. The app will generate a report with the predicted outcomes.')

st.header('Is my data safe with the app?')
q6_expander = st.expander('Click to reveal answer')
with q6_expander:
    st.write('Yes, your data is safe with the app. We use industry-standard security protocols to ensure the confidentiality and integrity of your data.')

st.header('How much does it cost to use the app?')
q7_expander = st.expander('Click to reveal answer')
with q7_expander:
    st.write('We offer both free and paid plans for the app. The free plan allows you to make a limited number of predictions per month, while the paid plans offer unlimited predictions and additional features.')

st.header('How do I get started?')
q8_expander = st.expander('Click to reveal answer')
with q8_expander:
    st.write('To get started, simply sign up for an account on our website and follow the instructions to upload your data and start making predictions.')
