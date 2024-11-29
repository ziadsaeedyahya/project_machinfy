import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import pickle

# configure page
data2=pd.read_csv('data2.csv')
st.set_page_config(layout='wide')
data = pd.read_csv('Loan approval prediction.csv', encoding="ISO-8859-1")
#st.write(df.head())

#sidebar
option = st.sidebar.selectbox("Pick a choice:",['Home','EDA','ML'])
if option == 'Home':
    st.title("Loan approval prediction App")
    st.text('Author: @ziad')
    st.dataframe(data.head(20))
elif option == 'EDA':
    st.title("Loan Approval Prediction EDA")

    def stacked_bar_plot(data, feature, target='loan_status'):
        crosstab = pd.crosstab(data[feature], data[target], normalize='index')
        fig, ax = plt.subplots(figsize=(12, 6))
        crosstab.plot(kind='bar', stacked=True, ax=ax, cmap='coolwarm')
        ax.set_title(f'Stacked Bar Plot of {feature} vs {target}')
        ax.set_ylabel('Proportion')
        st.pyplot(fig)

    st.title("Stacked Bar Plot Example")
    stacked_bar_plot(data, 'loan_intent')
    stacked_bar_plot(data, 'person_home_ownership')
    stacked_bar_plot(data, 'loan_grade')
    stacked_bar_plot(data, 'cb_person_default_on_file')

    st.title("Pie Chart Example")
    plt.figure(figsize=(3, 3))
    plt.pie(data['loan_status'].value_counts(),
            labels=['Not Approved', 'Approved'],
            autopct='%1.0f%%',
            explode=[0.0, 0.2],
            colors=['blue', 'red'])
    plt.title('Loans Analysis', fontdict={'fontsize': 15, 'fontweight': 'bold', 'color': 'darkblue'})
    st.pyplot(plt)

    def categorize_age(age):
        if 20 <= age < 30:
            return '20s'
        elif 30 <= age < 40:
            return '30s'
        elif 40 <= age < 50:
            return '40s'
        elif 50 <= age < 60:
            return '50s'
        elif 60 <= age < 66:
            return '60s'
        else:
            return 'Others'

    data['Age_th'] = data['person_age'].apply(categorize_age)

    # Filter data for those who got loans
    data_got_loans = data[data['loan_status'] == 1]
    Age_dist = data_got_loans.groupby('Age_th')['id'].count().reset_index()

    # Bar plot for Age Distribution
    #st.title("Age Distribution of Individuals Who Got Loans")
    #fig, ax = plt.subplots(figsize=(10, 6))
    #ax.bar(Age_dist['Age_th'], Age_dist['id'], color='skyblue')
    #ax.set_title('Age Distribution of Approved Loans')
    #ax.set_xlabel('Age Group')
    #ax.set_ylabel('Number of Loans')
    #st.pyplot(fig)

    fig = px.bar(Age_dist, x='Age_th', y='id',
             title='Most Age Getting Loan in 2024',
             labels={'Age_th': 'Age Tiers', 'id': 'Distribution'},
             text='id')
    st.title('Age Distribution of Individuals Who Got Loans')
    st.plotly_chart(fig)

elif option == "ML":
    st.title("Loan approval Prediction")
    st.text("In this app, we will predict the Loan approval")
    st.text("Please enter the following values:")
# building model
    person_age = st.number_input("Enter person_age")
    person_income = st.number_input("Enter person_income")
    person_home_ownership = st.text_input("Enter person_home_ownership")
    home_ownership_mapping = {'RENT': 0,
                              'MORTGAGE': 1,
                              'OWN': 2,
                              'OTHER': 3}
# Convert string to numerical value
    person_home_ownership_encoded = home_ownership_mapping.get(person_home_ownership, 'error')
    person_home_ownership= person_home_ownership_encoded
    if person_home_ownership_encoded == 'error':
     st.error('Invalid home ownership type entered. Please enter RENT, MORTGAGE, OWN, or OTHER.')
# ------------------------------------------------------------------------------------------------
    loan_intent=st.text_input('enter loan_reason')
    Loan_int={'EDUCATION':0,
          'MEDICAL':1,
          'PERSONAL':2,
          'VENTURE':3,
          'DEBTCONSOLIDATION':4,
          'HOMEIMPROVEMENT':5}
    loan_int_encoded=Loan_int.get(loan_intent,'error')
    loan_intent = loan_int_encoded
    if loan_int_encoded == 'error':
     st.error('Invalid home ownership type entered. Please enter EDUCATION, MEDICAL,PERSONAL,VENTURE,DEBTCONSOLIDATION,HOMEIMPROVEMENT.')
#------------------------------------------------------
    person_emp_length=st.number_input("Enter person_emp_length")
#-----------------------------------------------------------------------------------------------    
    loan_grade=st.text_input("enter loan_grade")
    Loan_grade_dict={'A':0,
                'B':1,
                'C':2,
                'D':3,
                'E':4,
                'F':5,
                'G':6}
    loan_grade_encoded = Loan_grade_dict.get(loan_grade,'error')
    loan_grade = loan_grade_encoded
    if loan_grade_encoded == 'error':
     st.error('Invalid home ownership type entered. Please enter A , B , B , C , D , E , F , G .')
    #-----------------------------------------
    loan_amnt = st.number_input("Enter loan_amnt")
    loan_int_rate = st.number_input("Enter loan_int_rate")
    loan_percent_income = st.number_input("Enter loan_percent_income")
    #------------------------------------------------------------------------
    cb_person_default_on_file = st.text_input("Enter cb_person_default_on_file")
    cb_onfile_dic={"N":0,
               "Y":1}
    cb_onfile_encoded =cb_onfile_dic.get(cb_person_default_on_file,'error')
    cb_person_default_on_file = cb_onfile_encoded
    if cb_onfile_encoded == 'error':
     st.error('Invalid home ownership type entered. Please enter Y , N .')
    #---------------------------------------------
    cb_person_cred_hist_length = st.number_input("Enter cb_person_cred_hist_length")
    btn = st.button("Submit")

    X=data2.drop(['id','loan_status'],axis=1)
    y=data2['loan_status']
    from imblearn.combine import SMOTEENN       
    from collections import Counter 
    smote_enn = SMOTEENN(random_state=42)
    X_combined, y_combined = smote_enn.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.20, random_state=42)
    gb = XGBClassifier( n_estimators=300,learning_rate=0.08, gamma=0.02,subsample=0.75,colsample_bytree=1, max_depth=15)#use XGB
    gb.fit(X_train, y_train)
    #gb = pickle.load(open('my_modelp.pkl','rb'))
    #result=gb.predict(([[person_age, person_income ,person_home_ownership,person_emp_length,loan_intent,loan_grade,loan_amnt,loan_int_rate,loan_percent_income,cb_person_default_on_file,cb_person_cred_hist_length]])) #predict
# Evaluate the model
    input_data = pd.DataFrame([[
    person_age, person_income, person_home_ownership_encoded, person_emp_length, loan_int_encoded, loan_grade_encoded, loan_amnt, loan_int_rate, loan_percent_income, cb_onfile_encoded, cb_person_cred_hist_length
    ]], columns=[
    'person_age', 'person_income', 'person_home_ownership', 'person_emp_length', 'loan_intent', 'loan_grade', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_default_on_file', 'cb_person_cred_hist_length'
        ])


    if st.button("Predict"):
       prediction = gb.predict(input_data)
       if prediction[0] == 1:
           st.write("Accept The Loan")
       else:
           st.write("Reject The Loan")

