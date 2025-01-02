import streamlit as st
import pandas as pd
import os
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

def create_csv_qa_agent(csv_file, api_key, model='gpt-4o-mini'):
    try:
        llm = ChatOpenAI(
            openai_api_key=api_key, 
            model=model,
            temperature=0.0
        )
        
        agent = create_csv_agent(
            llm,
            csv_file,
            verbose=True,
            agent_type='openai-functions',
            prefix="You are an expert data analyst. Before answering any question, carefully analyze the CSV data to provide accurate and insightful answers. Focus on giving precise information from the data.",
            allow_dangerous_code=True
        )
        
        return agent
    
    except Exception as e:
        st.error(f"Error creating agent: {str(e)}")
        return None

def get_conversation_context(question_history):
    if not question_history:
        return ""
    
    context = "Previous conversation:\n"
    for i, (q, a) in enumerate(question_history, 1):
        context += f"Q{i}: {q}\nA{i}: {a}\n"
    context += "\nConsider the above context and CSV data to answer: "
    return context

def main():
    st.set_page_config(layout="wide")
    
    # Add custom CSS for better formatting and white text in conversation history
    st.markdown("""
        <style>
        .main > div {
            max-width: 1000px;
            margin: auto;
            padding: 0 1rem;
        }
        .stTextArea > div > div > textarea {
            min-height: 100px;
            font-size: 16px;
            line-height: 1.5;
        }
        /* White text for conversation history */
        .conversation-text textarea {
            color: white !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Center-aligned title
    st.markdown("<h1 style='text-align: center;'>📊 CSV Analysis Assistant</h1>", unsafe_allow_html=True)
    
    # Initialize session states
    if 'question_history' not in st.session_state:
        st.session_state.question_history = []
    if 'clear_input' not in st.session_state:
        st.session_state.clear_input = False
    if 'df' not in st.session_state:
        st.session_state.df = None
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🔑 OpenAI API Configuration")
        openai_api_key = st.text_input(
            "Enter OpenAI API Key", 
            type="password",
            help="Get your API key from https://platform.openai.com/account/api-keys",
            value=os.getenv("OPENAI_API_KEY", "")
        )
        
        model = st.selectbox(
            "Select OpenAI Model",
            options=['gpt-4o-mini'],
            index=0
        )
        
        if st.button("Clear Conversation History"):
            st.session_state.question_history = []
            st.success("Conversation history cleared!")

    # Main content
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
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
            
            # Load and cache DataFrame
            if st.session_state.df is None:
                st.session_state.df = pd.read_csv(csv_path)
            
            st.subheader("📝 File Preview")
            st.dataframe(st.session_state.df.head())
            
          
            # Question input using text_area
            st.subheader("🤔 Ask Questions About Your Data")
            
            if st.session_state.clear_input:
                question = st.text_area("Enter your question:", value="", height=100, key=f"question_input_{len(st.session_state.question_history)}")
                st.session_state.clear_input = False
            else:
                question = st.text_area("Enter your question:", height=100, key=f"question_input_{len(st.session_state.question_history)}")
            
            # Get Answer button
            if st.button("Get Answer", use_container_width=True):
                if not openai_api_key:
                    st.warning("Please enter your OpenAI API key!")
                elif not question:
                    st.warning("Please enter a question!")
                else:
                    with st.spinner("Analyzing data and processing your question..."):
                        try:
                            agent = create_csv_qa_agent(csv_path, openai_api_key, model)
                            if agent:
                                context = get_conversation_context(st.session_state.question_history)
                                full_question = context + question
                                answer = agent.run(full_question)
                                st.session_state.question_history.append((question, answer))
                                st.session_state.clear_input = True
                                st.rerun()
                        except Exception as e:
                            st.error(f"Error processing question: {str(e)}")
            
            # Display conversation history with white text
            if st.session_state.question_history:
                st.subheader("💭 Conversation History")
                for i, (q, a) in enumerate(reversed(st.session_state.question_history), 1):
                    st.markdown(f"**Question {i}:**")
                    st.markdown('<div class="conversation-text">', unsafe_allow_html=True)
                    st.text_area("", value=q, height=100, disabled=True, key=f"q_{i}")
                    st.markdown("**Answer:**")
                    st.text_area("", value=a, height=150, disabled=True, key=f"a_{i}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.divider()

if __name__ == "__main__":
    main()
