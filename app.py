
import streamlit as st
import pandas as pd
import numpy as np
import os
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


def create_csv_qa_agent(csv_file, api_key, model='gpt-4o-mini'):
    """
    Create a CSV Question Answering Agent
    
    Args:
        csv_file (str): Path to the CSV file
        api_key (str): OpenAI API key
        model (str): OpenAI model to use
    
    Returns:
        CSV Agent capable of answering questions
    """
    try:
        # Initialize the OpenAI model
        llm = ChatOpenAI(
            openai_api_key=api_key, 
            model=model,
            temperature=0.0
        )
        
        # Create CSV agent with Zero Shot React Description
        agent = create_csv_agent(
            llm,
            csv_file,
            verbose=True,
            agent_type='openai-functions',
            allow_dangerous_code=True  # Add this parameter

        )
        
        return agent
    
    except Exception as e:
        st.error(f"Error creating agent: {str(e)}")
        return None


def analyze_csv_metadata(df):
    """
    Analyze CSV file for scope, risk, and estimation metrics
    """
    analysis = {
        "scope": {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "memory_usage": df.memory_usage(deep=True).sum() / 1024 / 1024,  # in MB
            "data_types": df.dtypes.value_counts().to_dict()
        },
        "risk": {
            "missing_values": df.isnull().sum().to_dict(),
            "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict(),
            "duplicate_rows": df.duplicated().sum(),
            "unusual_values": {}
        },
        "estimation": {
            "processing_time_estimate": len(df) * len(df.columns) / 1000,  # rough estimate in seconds
            "complexity_score": 0
        }
    }
    
    # Analyze numerical columns for outliers and unusual values
    for column in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR))][column].count()
        analysis["risk"]["unusual_values"][column] = outliers

    # Calculate complexity score based on data characteristics
    analysis["estimation"]["complexity_score"] = (
        (analysis["scope"]["total_columns"] * 0.3) +
        (sum(analysis["risk"]["missing_values"].values()) / len(df) * 0.4) +
        (analysis["risk"]["duplicate_rows"] / len(df) * 0.3)
    )

    return analysis


def display_selected_analysis(analysis, selected_type):
    """Display specific analysis based on button selection"""
    if selected_type == "Scope":
        st.subheader("📊 Scope Analysis")
        # Use a container instead of columns
        st.metric("Total Rows", analysis["scope"]["total_rows"])
        st.metric("Total Columns", analysis["scope"]["total_columns"])
        st.metric("Memory Usage (MB)", f"{analysis['scope']['memory_usage']:.2f}")
        
        st.write("Data Types Distribution:")
        for dtype, count in analysis["scope"]["data_types"].items():
            st.write(f"- {dtype}: {count} columns")

    elif selected_type == "Risk":
        st.subheader("⚠️ Risk Analysis")
        st.write("Missing Values Analysis:")
        missing_df = pd.DataFrame({
            'Column': list(analysis["risk"]["missing_values"].keys()),
            'Missing Count': list(analysis["risk"]["missing_values"].values()),
            'Missing Percentage': [f"{x:.2f}%" for x in analysis["risk"]["missing_percentage"].values()]
        })
        st.dataframe(missing_df)
        st.metric("Duplicate Rows", analysis["risk"]["duplicate_rows"])
        
        if analysis["risk"]["unusual_values"]:
            st.write("Outliers in Numerical Columns:")
            for col, count in analysis["risk"]["unusual_values"].items():
                st.write(f"- {col}: {count} outliers")

    elif selected_type == "Estimation":
        st.subheader("⏱️ Estimation")
        st.metric("Estimated Processing Time (seconds)", 
                 f"{analysis['estimation']['processing_time_estimate']:.2f}")
        st.metric("Complexity Score (0-1)", 
                 f"{analysis['estimation']['complexity_score']:.2f}")

def main():
    st.set_page_config(layout="wide")  # Use wide layout
    st.title("📊 CSV Analysis Assistant")
    
    # Sidebar for API configuration
    st.sidebar.header("🔑 OpenAI API Configuration")
    openai_api_key = st.sidebar.text_input(
        "Enter OpenAI API Key", 
        type="password",
        help="You can get your API key from https://platform.openai.com/account/api-keys",
        value=os.getenv("OPENAI_API_KEY", "")
    )
    
    model = st.sidebar.selectbox(
        "Select OpenAI Model",
        options=['gpt-4o-mini'],
        index=0
    )
    
    # Main content area split into two columns
    left_col, right_col = st.columns([1, 1])
    
    with left_col:
        uploaded_file = st.file_uploader(
            "Upload CSV File",
            type=['csv'],
            help="Upload a CSV file to analyze and ask questions"
        )
        
        if uploaded_file is not None:
            os.makedirs("temp", exist_ok=True)
            csv_path = os.path.join("temp", uploaded_file.name)
            with open(csv_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Read CSV and show preview
            df = pd.read_csv(csv_path)
            st.subheader("📝 File Preview")
            st.dataframe(df.head())
            
            # Question Answering Section
            st.subheader("🤔 Ask Questions About Your Data")
            question = st.text_input("Enter your question:")
            
            if st.button("Get Answer"):
                if not openai_api_key:
                    st.warning("Please enter your OpenAI API key!")
                elif not question:
                    st.warning("Please enter a question!")
                else:
                    with st.spinner("Processing your question..."):
                        try:
                            agent = create_csv_qa_agent(
                                csv_file=csv_path,
                                api_key=openai_api_key,
                                model=model
                            )
                            
                            if agent:
                                answer = agent.run(question)
                                st.success("🎉 Answer:")
                                st.write(answer)
                        
                        except Exception as e:
                            st.error(f"Error processing question: {str(e)}")
    
    with right_col:
        if uploaded_file is not None:
            # Analysis type selection buttons in a horizontal layout
            st.subheader("Select Analysis Type")
            button_cols = st.columns(3)
            
            analysis = analyze_csv_metadata(df)
            
            # Store the button states
            if 'selected_analysis' not in st.session_state:
                st.session_state.selected_analysis = None
            
            if button_cols[0].button("Scope", use_container_width=True):
                st.session_state.selected_analysis = "Scope"
            if button_cols[1].button("Risk", use_container_width=True):
                st.session_state.selected_analysis = "Risk"
            if button_cols[2].button("Estimation", use_container_width=True):
                st.session_state.selected_analysis = "Estimation"
            
            # Display the selected analysis
            if st.session_state.selected_analysis:
                display_selected_analysis(analysis, st.session_state.selected_analysis)

if __name__ == "__main__":
    main()