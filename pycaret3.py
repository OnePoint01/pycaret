pip install -r requirements.txt

# Title of the app
st.title('PyCaret & Streamlit App')

# Add a sidebar
st.sidebar.title("Sidebar")

# Add a file uploader to the sidebar
uploaded_file = st.sidebar.file_uploader(label="Upload your input CSV file", type=['csv'])

# Check if a file is uploaded
if uploaded_file is not None:

    # Read the uploaded file
    data = pd.read_csv(uploaded_file)

    # Show the uploaded file
    st.write(data)

    # Display the first few rows of the DataFrame
    st.write('Head of Dataset:', data.head())

    # Display the shape of the DataFrame
    st.write('Data Shape:', data.shape)

    # Display the column names
    st.write('Column Names:', data.columns.tolist())

    # Display column data types
    st.write('Column Data Types:', data.dtypes)

    # Display summary statistics
    st.write('Summary Statistics:', data.describe().transpose())

    # Display correlation matrix using a heatmap
    st.write('Correlation Matrix:')
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(data.corr(), annot=True, ax=ax)
    st.pyplot(fig)

    # Ask user for the target column
    target = st.sidebar.selectbox('Select the target column', data.columns.tolist())

    # Check if target column values are numeric (Regression) or not (Classification)
    if pd.api.types.is_numeric_dtype(data[target]):
        # Setup PyCaret for Regression
        exp_reg = setup_reg(data = data, target = target, session_id=123)
        # Use AutoML to select the best model
        best_model = automl_reg(optimize = 'R2')  # optimize for R-squared for regression
    else:
        # Setup PyCaret for Classification
        exp_clf = setup_clf(data = data, target = target, session_id=123)
        # Use AutoML to select the best model
        best_model = automl_clf(optimize = 'Accuracy')  # optimize for accuracy for classification

    # Display the best model
    st.write(best_model)