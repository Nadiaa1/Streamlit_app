import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

st.subheader(This app will predict if a client can get a loan or no)

from PIL import Image
#image = Image.open(r'CUsersnediaDownloadsLoans.jpg')
#st.image(image, caption='Loans')

@st.cache()
def get_fvalue(val)
	feature_dict = {No1,Yes2}
	for key,value in feature_dict.items()
		if val == key
			return value

def get_value(val,my_dict)
	for key,value in my_dict.items()
		if val == key
			return value 
st.sidebar.header(Informations about the client )
gender_dict = {Male1,Female2}
feature_dict = {No1,Yes2}
edu={'Graduate'1,'Not Graduate'2}
prop={'Rural'1,'Urban'2,'Semiurban'3}
Gender=st.sidebar.radio('Gender',tuple(gender_dict.keys()))
Married=st.sidebar.radio('Married',tuple(feature_dict.keys()))
Dependents=st.sidebar.selectbox('Dependents',options=['0','1' , '2' , '3+'])
Education=st.sidebar.selectbox('Education',tuple(edu.keys()))
ApplicantIncome=st.sidebar.slider('ApplicantIncome',0,10000,0,)
CoapplicantIncome=st.sidebar.slider('CoapplicantIncome',0,10000,0,)
LoanAmount=st.sidebar.slider('LoanAmount in K$',9.0,700.0,200.0)
Loan_Amount_Term=st.sidebar.selectbox('Loan_Amount_Term',(12.0,36.0,60.0,84.0,120.0,180.0,240.0,300.0,360.0)) 
Credit_History=st.sidebar.selectbox('Credit_History',(0.0,1.0))
Property_Area=st.sidebar.radio('Property_Area',tuple(prop.keys()))


class_0 , class_3 , class_1,class_2 = 0,0,0,0
if Dependents == '0'
	class_0 = 1
elif Dependents == '1'
	class_1 = 1
elif Dependents == '2' 
	class_2 = 1
else
    class_3= 1

Rural,Urban,Semiurban=0,0,0
if Property_Area == 'Urban' 
	Urban = 1
elif Property_Area == 'Semiurban' 
	Semiurban = 1
else 
    Rural=1
    
data1={
    'Gender'Gender,
    'Married'Married,
    'Dependents'[class_0,class_1,class_2,class_3], 
    'Education'Education,
    'ApplicantIncome'ApplicantIncome,
    'CoapplicantIncome'CoapplicantIncome,
    'LoanAmount'LoanAmount,
    'Loan_Amount_Term'Loan_Amount_Term,
    'Credit_History'Credit_History,
    'Property_Area'[Rural,Urban,Semiurban],
    }

feature_list=[ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History,get_value(Gender,gender_dict),get_fvalue(Married),data1['Dependents'][0],data1['Dependents'][1],data1['Dependents'][2],data1['Dependents'][3],get_value(Education,edu),data1['Property_Area'][0],data1['Property_Area'][1],data1['Property_Area'][2]]

single_sample = np.array(feature_list).reshape(1,-1)

model_choice = st.selectbox(Select Model,[Random Forest Classifier,Logistic Regression])
if st.button(Predict)
    if model_choice == Random Forest Classifier
        loaded_model = pickle.load(open(r'CUsersMSIDesktopNadia ProjectRforestClas.sav', 'rb'))
        prediction = loaded_model.predict(single_sample)
    elif model_choice == Logistic Regression
        loaded_model = pickle.load(open(r'CUsersMSIDesktopNadia Projectmodel_logreg.sav', 'rb'))
        prediction = loaded_model.predict(single_sample)
    else
        prediction=5
    st.write(prediction[0])
    if prediction[0] == '0'
        st.error(
    'According to our Calculations, you will not get the loan from Bank'
    )
    elif prediction[0] == '1' 
        st.success(
    'Congratulations!! you will get the loan from Bank'
    )
    else 
        st.write('3')












