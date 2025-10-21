import streamlit as st
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
#from sklearn.svm import SVR
lin_reg=LinearRegression()
model2 = RandomForestRegressor()

churn_frame_test = pd.read_csv('customer_churn_dataset-testing-master.csv')
object_column = ['Gender','Subscription Type', 'Contract Length']
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in object_column:
    churn_frame_test[col] = le.fit_transform(churn_frame_test[col])

churn_frame_train = pd.read_csv('customer_churn_dataset-training-master.csv')
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
gender = churn_frame_train[["Gender"]]
subscriptiontype = churn_frame_train[["Subscription Type"]]
contractlength = churn_frame_train[["Contract Length"]]
gender_encoded = ordinal_encoder.fit_transform(gender)
subscriptiontype_encoded = ordinal_encoder.fit_transform(subscriptiontype)
contractlength_encoded = ordinal_encoder.fit_transform(contractlength)
churn_frame_train["Gender"] = gender_encoded
churn_frame_train["Subscription Type"] = subscriptiontype_encoded
churn_frame_train["Contract Length"] = contractlength_encoded
churn_frame_train_clean=churn_frame_train.dropna(how='all')

x_train = churn_frame_train_clean[list(churn_frame_train.columns)[0: -1]]
y_train = churn_frame_train_clean['Churn']
x_test = churn_frame_test[list(churn_frame_test.columns)[0: -1]]
y_test = churn_frame_test['Churn']

lin_reg.fit(x_train, y_train)
pred1=lin_reg.predict(x_test)

model2.fit(x_train, y_train)
pred2 = model2.predict(x_test)

error_lin_reg = round(mean_absolute_error(y_test, pred1), 2)
error_pred2 = round(mean_absolute_error(y_test, pred2), 2)
print('MAE (Error):',error_lin_reg)
print('MAE (Error):',error_pred2)

units = st.radio("Select Machine Learning model type:",["Linear Regression","Random Forest"])
if units == "Linear Regression":
    #celsius = st.number_input("Enter temperature in Centigrade")
    #convertedtemp   =9/5 * celsius + 32
    #if st.button("Equivalent temperature in Fahrenheit?"):
    
        st.write('MAE (Error):',error_lin_reg)
else:
    #fahrenheit =  st.number_input("Enter temperature in Fahrenheit")
    #convertedtemp =(fahrenheit - 32) * 5/9
    #if st.button("Equivalent temperature in Celsius?"):
       st.write('MAE (Error):',error_pred2)