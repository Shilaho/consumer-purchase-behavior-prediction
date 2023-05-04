import streamlit as st
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
import datetime as dt 
import matplotlib.pyplot as plt 
import seaborn as sns 
from matplotlib import rcParams
from IPython.display import Image
from IPython.core.display import HTML 
from itertools import product
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from scipy import interp
from itertools import cycle
from IPython.core.display import HTML 
import plotly.express as px
import plotly.offline as py
import plotly.graph_objs as go
import warnings 
warnings.filterwarnings('ignore')

# set page title and favicon
st.set_page_config(page_title='Dataset Analysis', page_icon='📊')

# display sidebar message
st.sidebar.markdown('💡✨ Dive Deep into Understanding Your Dataset')

# display main title and subheader
st.title('Dataset Analysis')
st.subheader('Data is the next Gold')

online_file = st.file_uploader("Upload Online Dataset", type=["csv"])
order_file = st.file_uploader("Upload Order Dataset", type=["csv"])

if online_file is not None and order_file is not None:
    online = pd.read_csv(online_file) # Load the Non-transicational dataset
    order = pd.read_csv(order_file) # Load the Transicational dataset
    
    # Data Exploration 
    st.subheader('Data Exploration')
    st.write('Online Dataset')
    st.write(online.head())
    st.write('Order Dataset')
    st.write(order.head())
    
     # Data Analysis   
    st.write(f'The most recent time of online session: {online.dt.max()}')
    st.write(f'The start time of online session: {online.dt.min()}')
    st.write(f'The most recent time of online order: {order.orderdate.max()}')
    st.write(f'The start time of online order: {order.orderdate.min()}')
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(x='prodcat1', data=order, ax=ax)
    ax.set_xlabel('Prodcat1', fontsize=15)
    ax.set_ylabel('Count', fontsize=15)
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.set_title('Count of Prodcat1', fontsize=15)
    

    # Display plot in Streamlit
    st.subheader("Popularity of prodast1 in terms of the number of orders")
    st.pyplot(fig)
        
    
    # create catplot using seaborn
    fig = sns.catplot(x="category", kind="count", height=5, data=online)
    
    # set x-axis label font size to 15
    fig.set_xlabels("Procat1", fontsize=15)

    # set y-axis label font size to 15
    fig.set_ylabels("Count", fontsize=15)

    # set x-axis tick label font size to 10
    plt.xticks(fontsize=10)

    # display plot in Streamlit
    st.subheader("Popularity of prodcat1 in terms of the number of browsing sessions")
    st.pyplot(fig)
    
    
    
    # Create catplot
    fig = sns.catplot(x="event1", kind="count", height=5, data=online)

    # Set font size of x-axis labels
    for ax in fig.axes.flat:
        plt.setp(ax.get_xticklabels(), fontsize=10)

    # Set x and y axis labels and title
    plt.xlabel('event1', fontsize=15)
    plt.ylabel('Count', fontsize=15)
    plt.title('Number of sessions per category of event1')

    # Display plot in Streamlit
    st.subheader("Customer Behavior")
    st.write("Which channel is more popular in terms of event1 and event2?")
    st.pyplot(fig)

else:
    st.warning("Please upload both datasets.")