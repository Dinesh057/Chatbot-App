import os
import pandas as pd
import streamlit as st
from langchain_openai import AzureChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# 🔹 Set Azure OpenAI Credentials
AZURE_ENDPOINT = "https://firstinsightaoai.openai.azure.com/"
API_KEY = '5f4cc617ad2343b794045f85185fa6ee'  # Your API key as a string
API_VERSION = "2024-02-01"
MODEL_NAME = "firstinsightdeployment"

# 🎯 Streamlit UI Setup
st.title("📊 Data Query Assistant")
st.write("Upload a file for analysis and ask queries related to it.")

# 🔹 File uploader for Excel, CSV, or JSON files
uploaded_file = st.file_uploader("📂 Upload a file (Excel, CSV, or JSON)", type=["xlsx", "csv", "json"])

# Process uploaded file
if uploaded_file:
    # Check the file extension directly (not relying on MIME type)
    file_extension = uploaded_file.name.split('.')[-1]
    df = None
    try:
        if file_extension == 'xlsx':
            df = pd.read_excel(uploaded_file)
        elif file_extension == 'csv':
            df = pd.read_csv(uploaded_file)
        elif file_extension == 'json':
            df = pd.read_json(uploaded_file)
        else:
            st.error(f"Unsupported file type: {uploaded_file.name}. Please upload an Excel, CSV, or JSON file.")
        
        if df is not None:
            st.write("🔍 **File Data Preview:**")
            st.write(df.head())  # Show the first few rows of the data
            
            # Add button to start querying after file upload
            if 'start_querying' not in st.session_state:
                st.session_state.start_querying = False

            if st.session_state.start_querying:
                if 'user_question_file' not in st.session_state:
                    st.session_state.user_question_file = ""

                user_question = st.text_input("💬 Ask a question about the file data:", st.session_state.user_question_file)

                # Add a button to submit the query for the uploaded file
                if st.button("Submit Query for File Data"):
                    if user_question:
                        # Save the question to session state
                        st.session_state.user_question_file = user_question

                        # Create the LangChain agent for analyzing the file data
                        agent = create_pandas_dataframe_agent(
                            AzureChatOpenAI(
                                temperature=0,
                                azure_endpoint=AZURE_ENDPOINT,
                                api_key=API_KEY,  # Using the API key
                                api_version=API_VERSION,
                                model=MODEL_NAME
                            ),
                            df,  # Use the dataframe as the data source
                            verbose=True,
                            allow_dangerous_code=True,  # Allow running potentially dangerous code
                            agent_type=AgentType.OPENAI_FUNCTIONS
                        )

                        # Process user query for file data
                        with st.spinner("⚡ Processing..."):
                            result = agent.invoke(user_question)

                        # Display the result
                        st.write("✅ **Answer:**")
                        st.write(result)

            else:
                # Start Querying Button
                if st.button("Start Querying"):
                    st.session_state.start_querying = True

    except Exception as e:
        st.error(f"Error processing the file: {e}")
else:
    st.write("Please upload a file to proceed with analysis.")
