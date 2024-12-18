import streamlit as st

# I. Introduction
introduction = {
    "Overview": {
        "Dataset": "[**Bank Marketing Dataset**](https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset/data)",
        "Dataset Overview": "This dataset contains data on a bank's Term Deposit (Term Deposit) product marketing strategy.",
        "Source": "[Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, June 2014"
    },
    "Problem and Solution": {
        "Classification Problem": "Predict whether a customer will decide to enroll in this program '**Deposit**'/'**No Deposit**' (2-class classification problem) through the characteristics of that customer.",
        "Overview of the data set analysis plan in this report": [
            "Data Collection",
            "Data Manipulation: feature extraction, normalization",
            "Complementation: 'Null/None/NA' handling",
            "**Statistics:**",
            "* Statistics on data/label volume",
            "* Feature distribution, correlation, etc",
            "* Distribution charts",
            "**Cleaning:** Data pre-processing: normalization, splitting the training test set, dropping irrelevant features, etc",
            "**Feature extraction:** Encoding features into feature vectors and label vectors for computation",
            "**Analysis/forecasting using machine learning algorithms:**",
            "* Selecting some techniques",
            "* Evaluating effectiveness",
            "* Using metrics appropriate to the problem"
        ]
    },
    "Dataset Description": {
        "Description of information of some fields in the data set": [
            "* **age**: Represents the age of the customer.",
            "* **job**: Describes the person's occupation.",
            "* **marital**: Indicates the marital status of the person (e.g., married, single, divorced).",
            "* **education**: Represents the person's level of education (e.g., primary, secondary, tertiary).",
            "* **default**: Indicates whether the person has a credit card ('yes', 'no', or 'unknown').",
            "* **housing**: Indicates whether the person has a housing loan ('yes', 'no', or 'unknown').",
            "* **loan**: Indicates whether the person has a personal loan ('yes', 'no', or 'unknown').",
            "* **contact**: Describes the communication method used to contact the person (e.g., 'cellular', 'telephone').",
            "* **day**: Indicates the day of the week of the last contact.",
            "* **month**: Indicates the month of the last contact.",
            "* **duration**: Duration of the last contact in seconds.",
            "* **campaign**: The number of contacts performed during this campaign.",
            "* **pdays**: Represents the number of days since the last contact with the person, or -1 if they were never contacted before.",
            "* **previous**: The number of contacts performed before this campaign.",
            "* **poutcome**: Describes the outcome of the previous marketing campaign.",
            "* **deposit**: The target variable that indicates whether the person subscribed to a savings account ('yes' or 'no')."
        ]
    }
}

# II. Data Collecting and Manipulation
dataCollectionAndManipulation = {
    "Read Data": {
        "Dataset": "Placeholder for Dataset picture",
        "Overall Dataset": "Placeholder for Overall Dataset picture",
        "Check null and duplicated variables": "Placeholder for null and duplicated variables picture"
    },
    "Data Manipulation": {
        "Count value of 'job' attribute": "Placeholder for Count value of 'job' attribute plot",
        "Remove unknown values and special characters from the 'job' column": "Placeholder for Remove unknown values and special characters from the 'job' column picture",
        "Separating": [
            "Deposit and non-deposit customers",
            "Categorical and numerical columns"
        ]
    }
}

# Displaying the UI
st.title("Bank Marketing Dataset Report")
st.header("I. Introduction")

# Display Overview section
st.subheader("1.1. Overview")
st.write("**Dataset:**", introduction["Overview"]["Dataset"])
st.write("**Dataset Overview:**", introduction["Overview"]["Dataset Overview"])
st.write("**Source:**", introduction["Overview"]["Source"])

# Display Problem and Solution section
st.subheader("1.2. Problem and Solution")
st.write("**Classification Problem:**", introduction["Problem and Solution"]["Classification Problem"])
st.write("**Overview of the data set analysis plan in this report:**")
for item in introduction["Problem and Solution"]["Overview of the data set analysis plan in this report"]:
    if "**" in item:
        st.markdown(f"{item}")
    else:
        st.markdown(f"* {item}")

# Display Dataset Description section
st.subheader("1.3. Dataset Description")
st.write("**Description of information of some fields in the data set:**")
for item in introduction["Dataset Description"]["Description of information of some fields in the data set"]:
    st.markdown(f"{item}")

# Display Data Collecting and Manipulation section
st.header("II. Data Collecting and Manipulation")

# Display Read Data section
st.subheader("2.1. Read Data")
st.write("**Dataset:**", dataCollectionAndManipulation["Read Data"]["Dataset"])
st.write("**Overall Dataset:**", dataCollectionAndManipulation["Read Data"]["Overall Dataset"])
st.write("**Check null and duplicated variables:**", dataCollectionAndManipulation["Read Data"]["Check null and duplicated variables"])

# Display Data Manipulation section
st.subheader("2.2. Data Manipulation")
st.write("**Count value of 'job' attribute:**", dataCollectionAndManipulation["Data Manipulation"]["Count value of 'job' attribute"])
st.write("**Remove unknown values and special characters from the 'job' column:**", dataCollectionAndManipulation["Data Manipulation"]["Remove unknown values and special characters from the 'job' column"])

st.write("**Separating:**")
for item in dataCollectionAndManipulation["Data Manipulation"]["Separating"]:
    st.markdown(f"* {item}")

# Now we move to session 2
st.write("Now we move to session 2")
