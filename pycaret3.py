# Import necessary libraries
import streamlit as st
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import necessary functions from PyCaret
from pycaret.classification import setup as setup_clf, automl as automl_clf,models as clf_models
from pycaret.regression import setup as setup_reg, automl as automl_reg,models as reg_models

# Title of the app
st.title('PyCaret & Streamlit App')

# Add a sidebar
st.sidebar.title("Sidebar")

# Add a file uploader to the sidebar
uploaded_file = st.sidebar.file_uploader(label="Upload your input CSV file", type=['csv'])

# Check if a file is uploaded
if uploaded_file is not None:

    # Read the uploaded file
    df = pd.read_csv(uploaded_file)

        # Initialize LabelEncoder
    le = LabelEncoder()
    
    # Handle missing values and transform object type columns using LabelEncoder
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])  # fill missing values with mode for categorical columns
            df[col] = le.fit_transform(df[col].astype(str))  # transform with LabelEncoder
        else:
            df[col] = df[col].fillna(df[col].mean())  # fill missing values with mean for numeric columns

    st.write('Missing values are handled correctly')

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

    # Display correlation matrix using a heatmap
    st.write('Correlation Matrix:')
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(df.corr(), annot=True, ax=ax)
    st.pyplot(fig)

    # Ask user for the target column
    ColTarget = st.sidebar.selectbox('Select the target column', df.columns.tolist())
    if ColTarget is not None:
        # Check if target column values are numeric (Regression) or not (Classification)
        if pd.api.types.is_numeric_dtype(df[ColTarget]):
            # Setup PyCaret for Regression
            exp_reg = setup_reg(data = df, target = ColTarget, session_id=123, verbose=False)
            # Use AutoML to select the best model
            best_model = automl_reg(optimize = 'R2')  # optimize for R-squared for regression
            available_clf_models = clf_models(exp_reg)
            st.write('available_clf_models:', available_clf_models)
        else:
            # Setup PyCaret for Classification
            exp_clf = setup_clf(data = df, target = ColTarget, session_id=123, verbose=False)
            # Use AutoML to select the best model
            best_model = automl_clf(optimize = 'Accuracy')  # optimize for accuracy for classification
            available_reg_models = reg_models(exp_clf)
            st.write('available_clf_models:', available_reg_models)

        # Display the best model
        st.write(best_model)
