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
import pickle
warnings.filterwarnings('ignore')

st.title('Consumer Behaviour Prediction')
st.subheader('A Neural Network approach For Enhanced Business Strategy And Consumer Satisfaction')

# display sidebar message
#st.sidebar.markdown('💡✨ Dive Deep into Predicting using your Dataset')


online_file = st.file_uploader("Upload Online Dataset", type=["csv"])
order_file = st.file_uploader("Upload Order Dataset", type=["csv"])

if online_file is not None and order_file is not None:
    online = pd.read_csv(online_file) # Load the Non-transicational dataset
    order = pd.read_csv(order_file) # Load the Transicational dataset

    if st.checkbox('Show online dataset info'):
        st.write(online.info())
        
    if st.checkbox('Show order dataset info'):
        st.write(order.info())
        
    st.subheader('Online dataset')
    st.write(online.head())

    st.subheader('Order dataset')
    st.write(order.head())
    
       # Convert the data type of dt and orderdate
    order['orderdate'] = pd.to_datetime(order['orderdate']) 
    online['dt'] = pd.to_datetime(online['dt']) 
    # Truncate order data only between 2016-01-01 to 2017-12-31
    order = order[order['orderdate'] < '2018-01-01']

    # Get the max purchase date for each customer
    user_segment = order.groupby('custno').orderdate.max().reset_index()
    # Calculate the number of inactive numbers of each customer 
    user_segment['Recency'] = (user_segment['orderdate'].max() - user_segment['orderdate']).dt.days
    # Calculate the Frequency and revenue for each customer 
    rfmTable = order.groupby('custno').agg({'ordno': lambda x: len(x), 
                                            'revenue': lambda x: x.sum()}).reset_index().rename(
                                            columns={"ordno": "Frequency", "revenue": "Revenue"})
    rfmTable = pd.merge(rfmTable, user_segment, on='custno').drop(['orderdate'], axis=1)
    
    # Functions for RMF score 
    def RScore(x,p,d):
        """ Generate the Regency Score based on the quantile values"""
        if x<=d[p][0.25] :
            return 1 
        elif x<=d[p][0.5]:
            return 2 
        elif x<=d[p][0.75]:
            return 3 
        else :
            return 4 
    def FMScore(x,p,d):
        """ Generate the Frequency and Monetary Score based on the quantile values """
        if x<=d[p][0.25] :
            return 4 
        elif x<=d[p][0.5]:
            return 4 
        elif x<=d[p][0.75]:
            return 2 
        else :
            return 1
        
    if st.checkbox('Show RFM table'):
        st.write(rfmTable)
        
           
    # Add segment numbers to RMF table based on the quantile values.
    quantiles = rfmTable.quantile(q=[0.25,0.5,0.75])
    quantiles = quantiles.to_dict()
    rfmTable['r_quartile'] = rfmTable['Recency'].apply(RScore, args=('Recency',quantiles,))
    rfmTable['f_quartile'] = rfmTable['Frequency'].apply(FMScore, args=('Frequency',quantiles,))
    rfmTable['m_quartile'] = rfmTable['Revenue'].apply(FMScore, args=('Revenue',quantiles,))
    # Calculate the RFM score of each customer 
    rfmTable['RFMScore'] = rfmTable.r_quartile.map(str) + \
                           rfmTable.f_quartile.map(str) +\
                           rfmTable.m_quartile.map(str)
    # Sort according to the RMF scores from the best customers
    rfmTable = rfmTable.sort_values(by = ['RFMScore'])
    
        
    # assuming rfmTable is a pandas DataFrame with columns 'RFMScore', 'Frequency', and 'Revenue'

    plot_data = [
        go.Scatter(
            x=rfmTable.query("RFMScore == '111'")['Frequency'],
            y=rfmTable.query("RFMScore == '111'")['Revenue'],
            mode='markers',
            name='Best Customers',
            marker=dict(size=7,
                        line=dict(width=1),
                        color='blue',
                        opacity=0.8
                        )
        ),
        go.Scatter(
            x=rfmTable.query("RFMScore == '311'")['Frequency'],
            y=rfmTable.query("RFMScore == '311'")['Revenue'],
            mode='markers',
            name='Almost Lost Customers',
            marker=dict(size=9,
                        line=dict(width=1),
                        color='green',
                        opacity=0.5
                        )
        ),
        go.Scatter(
            x=rfmTable.query("RFMScore == '444'")['Frequency'],
            y=rfmTable.query("RFMScore == '444'")['Revenue'],
            mode='markers',
            name='Lost Cheap Customers',
            marker=dict(size=11,
                        line=dict(width=1),
                        color='red',
                        opacity=0.9
                        )
        ),
    ]

    plot_layout = go.Layout(
        yaxis={'title': "Revenue"},
        xaxis={'title': "Frequency"},
        title='Segments'
    )

    fig = go.Figure(data=plot_data, layout=plot_layout)

    st.plotly_chart(fig, use_container_width=True)
    
    # Assuming rfmTable is a pandas DataFrame containing the necessary data
    # If not, you'll need to load the data first

    plot_data = [
        go.Scatter(
            x=rfmTable.query("RFMScore == '111'")['Frequency'],
            y=rfmTable.query("RFMScore == '111'")['Recency'],
            mode='markers',
            name='Best Customers',
            marker=dict(size=7,
                        line=dict(width=1),
                        color='blue',
                        opacity=0.8
                        )
        ),
        go.Scatter(
            x=rfmTable.query("RFMScore == '311'")['Frequency'],
            y=rfmTable.query("RFMScore == '311'")['Recency'],
            mode='markers',
            name='Almost Lost Customers',
            marker=dict(size=9,
                        line=dict(width=1),
                        color='green',
                        opacity=0.5
                        )
        ),
        go.Scatter(
            x=rfmTable.query("RFMScore == '444'")['Frequency'],
            y=rfmTable.query("RFMScore == '444'")['Recency'],
            mode='markers',
            name='Lost Cheap Customers',
            marker=dict(size=11,
                        line=dict(width=1),
                        color='red',
                        opacity=0.9
                        )
        ),
    ]

    plot_layout = go.Layout(
        yaxis={'title': "Recency"},
        xaxis={'title': "Frequency"},
        title='Segments'
    )
    fig = go.Figure(data=plot_data, layout=plot_layout)

    # Display the plot using Streamlit
    st.plotly_chart(fig)
    
    # Assuming rfmTable is a pandas DataFrame containing the necessary data
    # If not, you'll need to load the data first

    plot_data = [
        go.Scatter(
            x=rfmTable.query("RFMScore == '111'")['Recency'],
            y=rfmTable.query("RFMScore == '111'")['Revenue'],
            mode='markers',
            name='Best Customers',
            marker=dict(size=7,
                        line=dict(width=1),
                        color='blue',
                        opacity=0.8
                        )
        ),
        go.Scatter(
            x=rfmTable.query("RFMScore == '311'")['Recency'],
            y=rfmTable.query("RFMScore == '311'")['Revenue'],
            mode='markers',
            name='Almost Lost Customers',
            marker=dict(size=9,
                        line=dict(width=1),
                        color='green',
                        opacity=0.5
                        )
        ),
        go.Scatter(
            x=rfmTable.query("RFMScore == '444'")['Recency'],
            y=rfmTable.query("RFMScore == '444'")['Revenue'],
            mode='markers',
            name='Lost Cheap Customers',
            marker=dict(size=11,
                        line=dict(width=1),
                        color='red',
                        opacity=0.9
                        )
        ),
    ]

    plot_layout = go.Layout(
        yaxis={'title': "Revenue"},
        xaxis={'title': "Recency"},
        title='Segments'
    )
    fig = go.Figure(data=plot_data, layout=plot_layout)

    # Display the plot using Streamlit
    st.plotly_chart(fig)
        
    ## Convert 'orderdate' column to datetime data type
    order['orderdate'] = pd.to_datetime(order['orderdate'])
    online['dt'] = pd.to_datetime(online['dt'])

    ## Select the data for Training Features 
    train_x_order = order[order.orderdate < '2017-01-01']
    train_x_online = online[online.dt  < '2017-01-01']

    ## Select the data for Training Labels
    train_y = order[(order['orderdate']>pd.to_datetime('2016-12-31')) & (order['orderdate']< pd.to_datetime('2017-07-01'))]

    ## Select the data for Testing Features 
    test_x_order = order[(order['orderdate']>=pd.to_datetime('2016-06-30')) & (order['orderdate']< pd.to_datetime('2017-07-01'))]
    test_x_online = online[(online['dt']>=pd.to_datetime('2016-06-30')) & (online['dt']< pd.to_datetime('2017-07-01'))]

    ## Select the data for Testing Labels
    test_y = order[(order['orderdate']>=pd.to_datetime('2017-07-01')) & (order['orderdate']< pd.to_datetime('2018-01-01'))]


    # Training Labels 
    # Generate the new dataframe with the unique combination of custno and product1
    unique_user = order.custno.unique()
    unique_prodcat = order.prodcat1.unique()
    outlist = list(product(unique_user, unique_prodcat))
    df1 = pd.DataFrame(data=outlist, columns=['custno','prodcat1'])

    # Count the purchasing count by grouping by the training_y data by custno and prodcut1
    df2 = train_y.groupby(['custno', 'prodcat1']).ordno.count().reset_index().rename(columns= {'ordno' :'PurchaseCount'})

    # Join Two tables using custno and prodcat1
    train_label = pd.merge(df1,df2, how = 'left', on=['custno', 'prodcat1'])

    # Convert the non-zero purchasing count to 1 and NA value to 1 in the joined table
    train_label.PurchaseCount.fillna(0, inplace = True)
    train_label.loc[train_label.PurchaseCount >0, 'PurchaseCount'] =1
    # Change the format of data shape
    train_label = pd.pivot_table(train_label, values='PurchaseCount', index=['custno'],
                columns=['prodcat1']).reset_index()

    # Testing Labels 
    # Count the purchasing count by grouping by the testing_y data by custno and prodcut1
    df3 = test_y.groupby(['custno', 'prodcat1']).ordno.count().reset_index().rename(columns= {'ordno' :'PurchaseCount'})

    # Join Two tables using custno and prodcat1
    test_label = pd.merge(df1,df3, how = 'left', on=['custno', 'prodcat1'])

    # Convert the non-zero purchasing count to 1 and NA value to 1 in the joined table
    test_label.PurchaseCount.fillna(0, inplace = True)
    test_label.loc[test_label.PurchaseCount >0, 'PurchaseCount'] =1

    # change the format of data shape 
    test_label = pd.pivot_table(test_label, values='PurchaseCount', index=['custno'],
                columns=['prodcat1']).reset_index()

    # Training Features 

    # Number of sessions for each customers 
    online_feature_train = train_x_online.groupby('custno').session.count().reset_index().rename(columns = {'session': 'Total_sessions'})

    # Number of sessions for the categories in event1 for each customer 
    event1_brow = train_x_online.groupby(['custno', 'event1']).session.count().reset_index()
    event1_brow = pd.pivot_table(event1_brow, values='session', index=['custno'],
                columns=['event1']).reset_index().rename(columns = {1:'event1_cat1',2:'event1_cat2',3:'event1_cat3',
                                                                    4:'event1_cat4',5:'event1_cat5',6:'event1_cat6',
                                                                    7:'event1_cat7',8:'event1_cat8',9:'event1_cat9',
                                                                    10:'event1_cat10',11:'event1_cat11'})
    event1_brow = event1_brow.fillna(0) # replace the missing value with 0 

    # Number of sessions for the categories in event2 for each customer 
    event2_brow = train_x_online.groupby(['custno', 'event2']).session.count().reset_index()
    event2_brow = pd.pivot_table(event2_brow, values='session', index=['custno'],
                columns=['event2']).reset_index().rename(columns = {1:'event2_cat1',2:'event2_cat2',3:'event2_cat3',
                                                                    4:'event2_cat4',5:'event2_cat5',6:'event2_cat6',
                                                                    7:'event2_cat7',8:'event2_cat8',9:'event2_cat9',
                                                                    10:'event2_cat10'})
    event2_brow = event2_brow.fillna(0)

     # Number of sessions a customer browsering a category(procat1)
    cat_brow = train_x_online.groupby(['custno', 'category']).session.count().reset_index()
    cat_brow = pd.pivot_table(cat_brow, values='session', index=['custno'],
                columns=['category']).reset_index().rename(columns = {1:'cat1_brow',2:'cat2_brow',3:'cat3_brow'})
    cat_brow = cat_brow.fillna(0)

    # Join the tables together with the custno 
    online_feature_train = pd.merge(online_feature_train, event1_brow, on = 'custno')
    online_feature_train = pd.merge(online_feature_train, event2_brow, on = 'custno')
    online_feature_train = pd.merge(online_feature_train, cat_brow, on = 'custno')


    # Number of sessions for each customers 
    online_feature_test = test_x_online.groupby('custno').session.count().reset_index().rename(columns = {'session': 'Total_sessions'})



    # Number of sessions for the categories in event1 for each customer 
    event1_brow = test_x_online.groupby(['custno', 'event1']).session.count().reset_index()
    event1_brow = pd.pivot_table(event1_brow, values='session', index=['custno'],
                columns=['event1']).reset_index().rename(columns = {1:'event1_cat1',2:'event1_cat2',3:'event1_cat3',
                                                                    4:'event1_cat4',5:'event1_cat5',6:'event1_cat6',
                                                                    7:'event1_cat7',8:'event1_cat8',9:'event1_cat9',
                                                                    10:'event1_cat10',11:'event1_cat11'})
    event1_brow = event1_brow.fillna(0) # replace the missing value with 0 

    # Number of sessions for the categories in event2 for each customer 
    event2_brow = test_x_online.groupby(['custno', 'event2']).session.count().reset_index()
    event2_brow = pd.pivot_table(event2_brow, values='session', index=['custno'],
                columns=['event2']).reset_index().rename(columns = {1:'event2_cat1',2:'event2_cat2',3:'event2_cat3',
                                                                    4:'event2_cat4',5:'event2_cat5',6:'event2_cat6',
                                                                    7:'event2_cat7',8:'event2_cat8',9:'event2_cat9',
                                                                    10:'event2_cat10'})
    event2_brow = event2_brow.fillna(0)
    
        # Number of sessions a customer browsering a category(procat1)
    cat_brow = test_x_online.groupby(['custno', 'category']).session.count().reset_index()
    cat_brow = pd.pivot_table(cat_brow, values='session', index=['custno'],
                columns=['category']).reset_index().rename(columns = {1:'cat1_brow',2:'cat2_brow',3:'cat3_brow'})
    cat_brow = cat_brow.fillna(0)

    # Join the tables together with the custno 
    online_feature_test = pd.merge(online_feature_test, event1_brow, on = 'custno')
    online_feature_test = pd.merge(online_feature_test, event2_brow, on = 'custno')
    online_feature_test = pd.merge(online_feature_test, cat_brow, on = 'custno')


    # Training Features 

    # Get the max purchase date for each customer
    user_recency = train_x_order.groupby('custno').orderdate.max().reset_index()

    # Calculate the number of inactive numbers of each customer in the training phase
    user_recency['Recency'] = (user_recency['orderdate'].max() - user_recency['orderdate']).dt.days
    user_recency.drop(columns=['orderdate']) 

    # Calculate the total order revenue and total number of orders of each customer 
    order_feature_train = train_x_order.groupby('custno').agg({'ordno': lambda x: len(x), 
                        'revenue': lambda x: x.sum()}).reset_index().rename(
                        columns={"ordno": "Total_Order", "revenue": "Total_Revenue"})

    # Calculate the purchase frequency for each customer 
    product_freq = train_x_order.groupby(['custno','prodcat1']).size().reset_index().rename(columns= {0 :'Purchase_frequency'})
    product_freq = pd.pivot_table(product_freq, values='Purchase_frequency', index=['custno'],
                columns=['prodcat1']).reset_index().rename(columns = {1:'cat1_freq',2:'cat2_freq',3:'cat3_freq',
                                                                        4:'cat4_freq',5:'cat5_freq',7:'cat7_freq'})
    product_freq = product_freq.fillna(0)

    # Join these tables together with custno
    order_feature_train = pd.merge(order_feature_train, user_recency, on = 'custno')
    order_feature_train = pd.merge(order_feature_train, product_freq, on = 'custno')
           
     # Rescaling the Order Frequency 
    order_feature_train['cat1_freq'] = order_feature_train['cat1_freq']/order_feature_train['Total_Order']
    order_feature_train['cat2_freq'] = order_feature_train['cat2_freq']/order_feature_train['Total_Order']
    order_feature_train['cat3_freq'] = order_feature_train['cat3_freq']/order_feature_train['Total_Order']
    order_feature_train['cat4_freq'] = order_feature_train['cat4_freq']/order_feature_train['Total_Order']
    order_feature_train['cat5_freq'] = order_feature_train['cat5_freq']/order_feature_train['Total_Order']
    order_feature_train['cat7_freq'] = order_feature_train['cat7_freq']/order_feature_train['Total_Order']

    # Drop orderdate from the dataframe 
    order_feature_train.drop(['orderdate'], axis =1 ,inplace = True)


    # Testing Features


    # Get the max purchase date for each customer
    user_recency = test_x_order.groupby('custno').orderdate.max().reset_index()

    # Calculate the number of inactive numbers of each customer in the training phase
    user_recency['Recency'] = (user_recency['orderdate'].max() - user_recency['orderdate']).dt.days


    # Calculate the total order revenue and total number of orders of each customer 
    order_feature_test = test_x_order.groupby('custno').agg({'ordno': lambda x: len(x), 
                        'revenue': lambda x: x.sum()}).reset_index().rename(
                        columns={"ordno": "Total_Order", "revenue": "Total_Revenue"})

    # Calculate the purchase frequency for each customer 
    product_freq = test_x_order.groupby(['custno','prodcat1']).size().reset_index().rename(columns= {0 :'Purchase_frequency'})
    product_freq = pd.pivot_table(product_freq, values='Purchase_frequency', index=['custno'],
                columns=['prodcat1']).reset_index().rename(columns = {1:'cat1_freq',2:'cat2_freq',3:'cat3_freq',
                                                                        4:'cat4_freq',5:'cat5_freq',7:'cat7_freq'})
    product_freq = product_freq.fillna(0)

    # Join these tables together with custno
    order_feature_test = pd.merge(order_feature_test, user_recency, on = 'custno')
    order_feature_test = pd.merge(order_feature_test, product_freq, on = 'custno')

    # Rescaling the Order Frequency 
    order_feature_test['cat1_freq'] = order_feature_test['cat1_freq']/order_feature_test['Total_Order']
    order_feature_test['cat2_freq'] = order_feature_test['cat2_freq']/order_feature_test['Total_Order']
    order_feature_test['cat3_freq'] = order_feature_test['cat3_freq']/order_feature_test['Total_Order']
    order_feature_test['cat4_freq'] = order_feature_test['cat4_freq']/order_feature_test['Total_Order']
    order_feature_test['cat5_freq'] = order_feature_test['cat5_freq']/order_feature_test['Total_Order']
    order_feature_test['cat7_freq'] = order_feature_test['cat7_freq']/order_feature_test['Total_Order']

    # Drop orderdate from the dataframe 
    order_feature_test.drop(['orderdate'], axis =1 ,inplace = True)

    # Join the training and testing features together
    # Combine order Features and online Features together 
    train_features = pd.merge(online_feature_train, order_feature_train, on = 'custno')
    test_features = pd.merge(online_feature_test, order_feature_test, on = 'custno')
    # Match the features and label for each customer 
    train = pd.merge(train_features, train_label, on= 'custno')
    test = pd.merge(test_features,test_label, on ='custno')

    # Extract Features for train and test sets
    train_x = train.drop(columns = [1,2,3,4,5,7, 'custno'])
    test_x = test.drop(columns = [1,2,3,4,5,7, 'custno'])

    # Extract Labels for train and test sets 
    train_y = train[[1,2,3,4,5,7]]
    test_y = test[[1,2,3,4,5,7]]


    # Drop feature  
    train_x =  train_x.drop(columns =['Total_sessions'])
    test_x =  test_x.drop(columns =['Total_sessions'])
    rf = RandomForestClassifier()
    param_grid = {
        'max_depth': list(range(5, 10)),
        'max_leaf_nodes': list(range(8, 12)),
        'max_features': ['sqrt', 'auto', 'log2'],
        'n_estimators':[100,200]}

    rf_cv = GridSearchCV(estimator = rf,
                        param_grid = param_grid)
    rf_cv.fit(train_x, train_y)
    print(rf_cv.best_params_)


    # Fit the model based on the GridSearch Result
    rf = RandomForestClassifier(n_estimators=200, max_depth = 9, max_features = 'sqrt', max_leaf_nodes= 10).fit(train_x,train_y)
    y_pred = rf.predict(test_x)

    y_pred
    
    #save the model
    #joblib.dump(rf, 'rf.pkl')
    
    #load the model
    #rf = joblib.load('rf.pkl')
    
    #predict the model
    
    results = rf.predict_proba(test_x) # return [n_samples, n_classes ]
    df = pd.DataFrame({'custno': test.custno,
                    'procat1_1': list(results[0][:, 1]),
                    'procat1_2': list(results[1][:, 1]),
                    'procat1_3': list(results[2][:, 1]),
                    'procat1_4': list(results[3][:, 1]),
                    'procat1_5': list(results[4][:, 1]),
                    'procat1_7': list(results[5][:, 1])})
    st.subheader('Prediction results')
    st.write("This results Show the Probability of Each Customer To buy Goods from The labeled Product Category")
    st.write(df.head())
    
    # drop the custno column and calculate the mean
    data = df.drop(columns=['custno']).mean()

    # create a bar plot
    fig, ax = plt.subplots()
    ax.bar(data.index, data.values)

    # set plot title and labels
    plt.title("Prediction of Purchasing Probability")
    plt.xlabel('Procat1')
    plt.ylabel('Prediction of Purchasing Probability')

    # display plot in Streamlit app
    st.pyplot(fig)
   
   # assuming train_y is a Pandas Series object
    train_y_mean = train_y.mean()
    fig, ax = plt.subplots()
    train_y_mean.plot(kind='bar', alpha=0.7, ax=ax)
    ax.set_xlabel('Procat1')
    ax.set_ylabel('Buying Probability')
    ax.set_title('Buying Probability of each category in Procat1')

    # display the plot using Streamlit
    st.pyplot(fig)
    


 
            
else:
    st.warning("Please upload both datasets.")
