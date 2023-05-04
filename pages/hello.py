import streamlit as st

st.set_page_config(
    page_title="Consumer Behavior Prediction App Onboarding",
    page_icon="👋",
)

# Define header and subheader
st.write("# Welcome to Consumer Behavior Prediction App 👋")
st.write("## Predicting Consumer Purchase Behavior from Browsing Data")

# Add images to showcase consumer behavior prediction
st.write("### Consumer Behavior Prediction")
st.image("https://d3caycb064h6u1.cloudfront.net/wp-content/uploads/2021/08/customerbehavior-scaled.jpg", width=500)
st.write("This image showcases how our app can predict a consumer's purchase behavior based on their browsing data.")

# Add images to showcase sales prediction
st.write("### Sales Prediction")
st.image("https://cdnwebsite.databox.com/wp-content/uploads/2022/04/21041435/sales-forecast-report.jpg", width=500)
st.write("This image showcases how our app can predict future sales based on consumer behavior data.")

# Add images to showcase data input
st.write("### Data Input")
st.image("https://res.cloudinary.com/practicaldev/image/fetch/s--fQrX-ukH--/c_imagga_scale,f_auto,fl_progressive,h_900,q_auto,w_1600/https://dev-to-uploads.s3.amazonaws.com/i/v92yxv9kzhhtdxtvkc46.png", width=500)
st.write("This image showcases how users can input their own data into the app to make predictions.")

# Add images to showcase effectiveness analysis
st.write("### Effectiveness Analysis")
st.image("https://editor.analyticsvidhya.com/uploads/975301.jfif", width=500)
st.write("This image showcases how our app analyzes the effectiveness of its predictions.")

# Add a sidebar with additional information or options
st.sidebar.success("Select a Task above.")
