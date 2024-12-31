# import streamlit as st
# import pandas as pd
# import os
# from langchain_experimental.agents.agent_toolkits import create_csv_agent
# from langchain_openai import ChatOpenAI
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# def create_csv_qa_agent(csv_file, api_key, model='gpt-4o'):
#     """
#     Create a CSV Question Answering Agent
    
#     Args:
#         csv_file (str): Path to the CSV file
#         api_key (str): OpenAI API key
#         model (str): OpenAI model to use
    
#     Returns:
#         CSV Agent capable of answering questions
#     """
#     try:
#         # Initialize the OpenAI model
#         llm = ChatOpenAI(
#             openai_api_key=api_key, 
#             model=model,
#             temperature=0.0
#         )
        
#         # Create CSV agent with Zero Shot React Description
#         agent = create_csv_agent(
#             llm,
#             csv_file,
#             verbose=True,
#             agent_type='openai-functions',
#             allow_dangerous_code=True  # Add this parameter

#         )
        
#         return agent
    
#     except Exception as e:
#         st.error(f"Error creating agent: {str(e)}")
#         return None

# def main():
#     st.title("📊 CSV Agent Question Answering App")
    
#     # Sidebar for OpenAI API Key input
#     st.sidebar.header("🔑 OpenAI API Configuration")
#     openai_api_key = st.sidebar.text_input(
#         "Enter OpenAI API Key", 
#         type="password", 
#         help="You can get your API key from https://platform.openai.com/account/api-keys",
#         value=os.getenv("OPENAI_API_KEY", "")
#     )
    
#     # Model selection
#     model = st.sidebar.selectbox(
#         "Select OpenAI Model", 
#         options=['gpt-4o', 'gpt-3.5-turbo'], 
#         index=0
#     )
    
#     # CSV File Upload
#     uploaded_file = st.file_uploader(
#         "Upload CSV File", 
#         type=['csv'], 
#         help="Upload a CSV file to start asking questions"
#     )
    
#     if uploaded_file is not None:
#         # Create temp directory if it doesn't exist
#         os.makedirs("temp", exist_ok=True)
        
#         # Save uploaded file
#         csv_path = os.path.join("temp", uploaded_file.name)
#         with open(csv_path, "wb") as f:
#             f.write(uploaded_file.getbuffer())
        
#         # Read CSV
#         df = pd.read_csv(csv_path)
        
#         # Display file preview
#         st.subheader("📝 File Preview")
#         st.dataframe(df.head())
        
#         # Show column names and dataset info
#         st.subheader("📋 Dataset Information")
#         col1, col2 = st.columns(2)
#         with col1:
#             st.metric("Total Rows", len(df))
#         with col2:
#             st.metric("Total Columns", len(df.columns))
        
#         st.write("Columns:", ", ".join(df.columns))
        
#         # Question Input
#         question = st.text_input("🤔 Ask a question about your data")
        
#         # Process Question
#         if st.button("Get Answer"):
#             # Validate inputs
#             if not openai_api_key:
#                 st.warning("Please enter your OpenAI API key!")
#                 return
            
#             if not question:
#                 st.warning("Please enter a question!")
#                 return
            
#             # Spinner and processing
#             with st.spinner("Processing your question..."):
#                 try:
#                     # Create CSV agent
                    # agent = create_csv_qa_agent(
#                         csv_file=csv_path, 
#                         api_key=openai_api_key, 
#                         model=model
#                     )
                    
#                     if agent:
#                         # Run agent to get answer
#                         answer = agent.run(question)
                        
#                         # Display answer
#                         st.success("🎉 Answer:")
#                         st.write(answer)
                
#                 except Exception as e:
#                     st.error(f"Error processing question: {str(e)}")

# if __name__ == "__main__":
#     main()




from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_openai import AzureChatOpenAI
from langchain_community.chat_models.azureml_endpoint import AzureMLChatOnlineEndpoint, AzureMLEndpointApiType, CustomOpenAIChatContentFormatter
import yaml
import pandas as pd
import os




def load_config():
    """
    Load configuration from config.yaml.
    Returns:
        dict: General configuration.
    """
    with open("config.yaml", "r") as file:
        return yaml.safe_load(file)

def load_model_config():
    """
    Load model configuration from models_config.yaml.
    Returns:
        dict: Model configurations.
    """
    with open("models_config.yaml", "r") as file:
        return yaml.safe_load(file)

    
def csv_zero_shot_react_description_AzureOpenAI(api_key, azure_endpoint, api_version, deployment_name, csv_file_path, question):
    try:
        agent = create_csv_agent(
            AzureChatOpenAI(
                api_key=api_key,
                azure_endpoint=azure_endpoint,
                api_version=api_version,
                deployment_name=deployment_name
            ),
            csv_file_path,
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            allow_dangerous_code=True
        )
        answer = agent.run(question)
        return answer
    except Exception as e:
        return f"Error occurred: {str(e)}"
    
    
def csv_openai_function_AzureOpenAI(api_key, azure_endpoint, api_version, deployment_name, csv_file_path, question):
    try:
        agent = create_csv_agent(
            AzureChatOpenAI(
                api_key=api_key,
                azure_endpoint=azure_endpoint,
                api_version=api_version,
                deployment_name=deployment_name
            ),
            csv_file_path,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            allow_dangerous_code=True
        )
        answer = agent.run(question)
        return answer
    except Exception as e:
        return f"Error occurred: {str(e)}"
    

def csv_llama_mistral_AzureOpenAI(endpoint_url, endpoint_api_key, csv_file_path, question):
    try:
        agent = create_csv_agent(
            AzureMLChatOnlineEndpoint(
                endpoint_url=endpoint_url,
                endpoint_api_type=AzureMLEndpointApiType.serverless,
                endpoint_api_key=endpoint_api_key,
                content_formatter=CustomOpenAIChatContentFormatter(),
            ),
            csv_file_path,
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            allow_dangerous_code=True,
            handle_parsing_errors=True
        )
        answer = agent.run(question)
        return answer
    except Exception as e:
        return f"Error occurred: {str(e)}"
      
    



def initialize_model(general_config, models_config):

    selected_model = general_config.get('model')
    if not selected_model:
        raise ValueError("No model specified in config.yaml.")
    

    model_specific_config = models_config['Models'].get(selected_model)
    if not model_specific_config:
        raise ValueError(f"Configuration for model '{selected_model}' not found in models_config.yaml.")
    

    file_path = general_config.get('directory') + '/' + general_config.get('input_file')
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    csv_file_path = os.path.join(general_config.get('directory'), f"{base_name}.csv")

    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == '.xlsx' or file_extension == '.xls':  # If it's an Excel file
        df = pd.read_excel(file_path)
        df.to_csv(csv_file_path, index=False, encoding='utf-8')
    elif file_extension == '.csv':  
        csv_file_path = file_path  
    else:
        raise ValueError(f"Unsupported file format: {file_extension}. Only .xlsx, .xls, or .csv are supported.")
    
    return {
        "selected_model": selected_model,
        "model_specific_config": model_specific_config,
        "csv_file_path": csv_file_path
    }



def process_question_with_model(selected_model, model_specific_config, csv_file_path, question):
    azure_models = ['GPT-4o', 'GPT-4o-mini', 'GPT-4', 'GPT-3.5-Turbo']
    llama_mistral_models = ['Llama-3-1-8b-instruct', 'Mistral-Small']
    config = load_config()
    agent_type = config['agent_type']
    if selected_model in azure_models:
        if agent_type == 1:
            return csv_openai_function_AzureOpenAI(
                api_key=model_specific_config['AZURE_OPENAI_API_KEY'],
                azure_endpoint=model_specific_config['AZURE_OPENAI_ENDPOINT'],
                api_version=model_specific_config['AZURE_OPENAI_API_VERSION'],
                deployment_name=model_specific_config['AZURE_OPENAI_MODEL'],
                csv_file_path=csv_file_path,
                question=question
            )
        elif agent_type == 2:
            return csv_zero_shot_react_description_AzureOpenAI(
                api_key=model_specific_config['AZURE_OPENAI_API_KEY'],
                azure_endpoint=model_specific_config['AZURE_OPENAI_ENDPOINT'],
                api_version=model_specific_config['AZURE_OPENAI_API_VERSION'],
                deployment_name=model_specific_config['AZURE_OPENAI_MODEL'],
                csv_file_path=csv_file_path,
                question=question
            )
                
    elif selected_model in llama_mistral_models:
        return csv_llama_mistral_AzureOpenAI(
            endpoint_url=model_specific_config['AZURE_OPENAI_ENDPOINT'],
            endpoint_api_key=model_specific_config['AZURE_OPENAI_API_KEY'],
            csv_file_path=csv_file_path,
            question=question
        )
    else:
        raise ValueError(f"Model '{selected_model}' is not supported.")
        
        
        
def print_column_names():
    """Reads a CSV or Excel file and prints the column names in a beautiful format."""
    config = load_config()
    directory = config["directory"] 
    input_file = config["input_file"]
    csv_file_path = str(directory+'/'+input_file)
    file_path= csv_file_path
    # Determine the file extension
    file_extension = file_path.split('.')[-1].lower()

    # Read the file based on its extension
    if file_extension == 'csv':
        df = pd.read_csv(file_path)
    elif file_extension in ['xls', 'xlsx']:
        df = pd.read_excel(file_path)
    else:
        print("Unsupported file format. Please provide a CSV or Excel file.")
        return
    
    # Get column names
    columns = df.columns.tolist()
    
    # Print column names in a beautiful format
    print("=" * 80)
    print(f"{'Column Names':^80}")  # Center-align the title
    print("=" * 80)
    
    # Display columns in a clean, readable format
    for index, column in enumerate(columns, 1):
        print(f"{index:>2}. {column}")
    
    print("=" * 80)