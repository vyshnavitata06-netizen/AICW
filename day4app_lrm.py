import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score
import matplotlib.pyplot as plt
#page configuration
st.set_page_config(page_title="LINEAR REGRESSION MODEL",page_icon="📈",layout="wide")
st.title("SCD-LRM: Linear Regression model")
st.write("This is a simple linear regression mode")
df=pd.read_csv("SCD.csv")
 #Display the data set
st.subheader("Dataset")
st.dataframe(df)
x=df[["Stock levels","Lead times","Order quantities"]]
y=df["Costs"]
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=LinearRegression()
lr_model=model.fit(x_train,y_train)
y_pred=lr_model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
#model information
st.subheader("Model Information")
col1,col2,col3=st.columns(3)
with col1:
    st.metric("MSE",f"{mse:.3f}")
with col2:
    st.metric("MAE",f"{mae:.3f}")
with col3:
    st.metric("R² Score",f"{r2:.3f}")

st.subheader("NEW PREDECTION")
Stock_levels = st.number_input(
    "Enter Stock Levels",
    max_value=120.0,
    min_value=0.0,
    value=5.0,
    step=0.5
)
Lead_times = st.number_input(
"EnterLead times",
    max_value=30.0,
    min_value=0.0,
    value=5.0,
    step=0.5
    
)
Order_quantities=st.number_input(
"Enter Order quantities",
    max_value=120.0,
    min_value=0.0,
    value=5.0,
    step=0.5
)
#predition button
if st.button("Predict Costs"):
    input_data=[[Stock_levels, Lead_times, Order_quantities]]
    prediction = lr_model.predict(input_data)    
    st.success(
        f"Predicted Cost: {prediction[0]:.3f}"
    )