# Import necessary libraries
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import necessary functions from PyCaret
from pycaret.classification import setup as setup_clf,compare_models as compare_models_clf
from pycaret.regression import setup as setup_reg,compare_models as compare_models_reg

# Title of the app
st.title('PyCaret & Streamlit App Capstone')

# Add a sidebar
st.sidebar.title("Settings Sidebar")

# Add a file uploader to the sidebar
uploaded_file = st.sidebar.file_uploader(label="Upload your input CSV file", type=['csv'])

ColTarget =0

# Check if a file is uploaded
if uploaded_file is not None:

    # Read the uploaded file
    df = pd.read_csv(uploaded_file)

    # Show the uploaded file
    st.write(df)

    # Display the first few rows of the DataFrame
    st.write('Head of Dataset:', df.head())

    # Display the shape of the DataFrame
    st.write('Data Shape:', df.shape)

    # Display the column names
    st.write('Column Names:', df.columns.tolist())

    # Display column data types
    st.write('Column Data Types:', df.dtypes)

    # Display summary statistics
    st.write('Summary Statistics:', df.describe().transpose())

    # Ask user for the target column
    ColTarget = st.sidebar.selectbox("Select the target column", df.columns)

    if len(ColTarget) != 0:
        select_columns = st.sidebar.multiselect("select the column you want removed from the dataframe",options=df.columns)
        df.drop(select_columns, axis=1, inplace=True)
        st.write(df.sample())
        
    # Initialize LabelEncoder
    le = LabelEncoder()
    
    # Handle missing values and transform object type columns using LabelEncoder
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])  # fill missing values with mode for categorical columns
            df[col] = le.fit_transform(df[col].astype(str))  # transform with LabelEncoder
        else:
            df[col] = df[col].fillna(df[col].mean())  # fill missing values with mean for numeric columns
    
    NumericalColumns = df.select_dtypes(['int64', 'float64']).columns
    CategoricalColumns = df.select_dtypes(['object']).columns
    NumericalStrategy = st.radio("Choose Strategy for numerical value:",["mean", "median"])
    CategoricalStrategy = st.radio("Choose Strategy for categorical value:",[ "most frequent","constant" ])
    for col in NumericalColumns:
        df[col] = SimpleImputer(strategy=NumericalStrategy, missing_values=np.nan).fit_transform(df[col].values.reshape(-1, 1))
    for col in CategoricalColumns:
        if df[col].nunique() > 7 :
            df[col] = SimpleImputer(strategy=CategoricalColumns, missing_values=np.nan).fit_transform(df[col].values.reshape(-1, 1))
        else:
            df[col] = le.fit_transform(df[col])
    if (len(NumericalColumns) != 0):
        st.header("Numerical Columns")
        st.write(NumericalColumns)
    if (len(NumericalColumns) != 0):
        st.header("Categorical columns")
        st.write(CategoricalColumns)
    if (len(CategoricalColumns) != 0 or len(NumericalColumns) != 0):
        st.header("number of null value ")
        st.write(df.isna().sum())
        
    # Display correlation matrix using a heatmap
    st.write('Correlation Matrix:')
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(df.corr(), annot=True, ax=ax)
    st.pyplot(fig)

    #ColTarget = st.sidebar.selectbox('Select the target column', df.columns.tolist())
    if ColTarget is not None:
        # Check if target column values are numeric (Regression) or not (Classification)
        if pd.api.types.is_numeric_dtype(df[ColTarget]):
            # Setup PyCaret for Regression
            exp_reg = setup_reg(data = df, target = ColTarget, session_id=123, verbose=False)
            # Compare models and select the best one
            best_model = compare_models_reg()
        else:
            # Setup PyCaret for Classification
            exp_clf = setup_clf(data = df, target = ColTarget, session_id=123, verbose=False)
            # Compare models and select the best one
            best_model = compare_models_clf()

        # Display the best model
        st.write(best_model)
