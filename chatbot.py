import os
import pandas as pd
import streamlit as st
import requests
from bs4 import BeautifulSoup
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
st.write("Provide a URL to extract and analyze structured or unstructured data or upload a file for analysis.")

# 🔹 URL input for extracting data
url_input = st.text_input("🔗 Paste a URL to analyze:")

# 🔹 File uploader for Excel, CSV, or JSON files
uploaded_file = st.file_uploader("📂 Upload a file (Excel, CSV, or JSON)", type=["xlsx", "csv", "json"])

# Function to fetch and parse data from URL with User-Agent modification
def load_data_from_url(url):
    """Load structured or unstructured data directly from a webpage."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        # Request the webpage content with headers
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an error if the request was unsuccessful
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract the main text content (paragraphs)
        text_content = "\n".join([p.get_text() for p in soup.find_all('p')])

        # Extract tables and convert them to pandas DataFrames
        tables = soup.find_all('table')
        dataframes = []
        for table in tables:
            rows = table.find_all('tr')
            table_data = []
            for row in rows:
                cols = row.find_all('td')
                cols = [ele.get_text() for ele in cols]
                table_data.append(cols)
            if table_data:
                dataframes.append(pd.DataFrame(table_data))

        # Return text and tables
        return text_content, dataframes
    except Exception as e:
        st.error(f"Error fetching or parsing the URL: {e}")
        return None, None

# Process URL input
if url_input:
    # Load data from the provided URL (text + tables)
    text_data, dataframes = load_data_from_url(url_input)

    if text_data:
        # Display text preview (first 500 characters to avoid overflow)
        st.write("🔍 **Content Preview (Text):**")
        st.write(text_data[:500])  # Show first 500 characters

        # 🔹 User input field for querying the text data
        if 'user_question_text' not in st.session_state:
            st.session_state.user_question_text = ""

        user_question = st.text_input("💬 Ask a question about the content:", st.session_state.user_question_text)

        # Add a button to start the query
        if st.button("Submit Query"):
            if user_question:
                # Save the question to session state
                st.session_state.user_question_text = user_question

                # Create the LangChain agent for analyzing the text data
                agent = create_pandas_dataframe_agent(
                    AzureChatOpenAI(
                        temperature=0,
                        azure_endpoint=AZURE_ENDPOINT,
                        api_key=API_KEY,  # Using the API key
                        api_version=API_VERSION,
                        model=MODEL_NAME
                    ),
                    pd.DataFrame({"text": [text_data]}),  # Treat text as DataFrame to analyze
                    verbose=True,
                    allow_dangerous_code=True,  # Allow running potentially dangerous code
                    agent_type=AgentType.OPENAI_FUNCTIONS
                )

                # Process user query for text data
                with st.spinner("⚡ Processing..."):
                    result = agent.invoke(user_question)

                # Display the result
                st.write("✅ **Answer:**")
                st.write(result)

    # Display tables and allow querying if available
    if dataframes:
        st.write("🔍 **Tables Extracted:**")
        for idx, df in enumerate(dataframes):
            st.write(f"📊 Table {idx+1}:")
            st.write(df.head())  # Display the first few rows of each table
            if f'table_query_{idx}' not in st.session_state:
                st.session_state[f'table_query_{idx}'] = ""
            table_query = st.text_input(f"💬 Query Table {idx+1}:", st.session_state[f'table_query_{idx}'])

            # Add a button to submit the table query
            if st.button(f"Submit Query for Table {idx+1}"):
                if table_query:
                    # Save table query to session state
                    st.session_state[f'table_query_{idx}'] = table_query
                    try:
                        # Execute query on the DataFrame
                        result = df.query(table_query)
                        st.write(f"✅ **Query Result for Table {idx+1}:**")
                        st.write(result)
                    except Exception as e:
                        st.error(f"Error executing query on Table {idx+1}: {e}")

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
