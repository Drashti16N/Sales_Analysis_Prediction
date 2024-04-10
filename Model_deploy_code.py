import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso

# Load the dataset
data = pd.read_csv('C:/Users/DIVYA/Desktop/drashti/sales_data_update.csv')  # Replace 'your_dataset.csv' with the actual dataset path

selected_col = ['Quantity','UnitPrice','Day','Month','Hour','Minutes']
x = data[selected_col]
# Define features and target variable
# X = data.drop(columns=['Yield'])
y = data['Sales']

# Define categorical and numerical features
numerical_features = ['Quantity', 'UnitPrice', 'Day', 'Month', 'Hour','Minutes']

# Preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features)
    ])

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=48)

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline

# Assuming you have preprocessor defined elsewhere

# Create a pipeline for Gradient Boosting model
gradient_boost_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(n_estimators=1000, validation_fraction=0.1, n_iter_no_change=5, tol=0.01,random_state=48))
])
# Train the Random Forest model
gradient_boost_model.fit(X_train, y_train)

# Predictions
y_pred = gradient_boost_model.predict(X_test)

# Calculate R2 score
r2 = r2_score(y_test, y_pred)

# Print the R2 score for Random Forest model
st.write(f'R2 Score for gradient Boost Model: {r2}')

# Streamlit app for predicting crop yield
import streamlit as st
import pandas as pd

st.title('Sales Prediction App')

# Input fields for features
Quantity = st.number_input('Quantity', value=0.0)
UnitPrice = st.number_input('UnitPrice', value=0.0)
Day = st.selectbox('Day', data['Day'].unique())
Month = st.selectbox('Month', data['Month'].unique())
Hour = st.selectbox('Hour', data['Hour'].unique())
Minutes = st.number_input('Minutes', value=0.0)

# Prepare input features
input_features = pd.DataFrame({
    'Quantity': [Quantity],
    'UnitPrice': [UnitPrice],
    'Day': [Day],
    'Month': [Month],
    'Hour': [Hour],
    'Minutes': [Minutes]
})

# Predict button
if st.button('Predict'):
    # Predict using the trained model
    prediction = gradient_boost_model.predict(input_features)
    st.success(f'Predicted Sales: {prediction[0]}')