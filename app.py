from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import os
import logging
import time
from functools import lru_cache
from typing import Sequence, Dict, Any, List, Optional
from typing_extensions import Annotated, TypedDict

# LangChain imports
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.callbacks import CallbackManager, StdOutCallbackHandler
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages

count=0

# Configure robust logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("chatbot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Environment configuration (consider using dotenv for production)
os.environ["GROQ_API_KEY"] = "YOUR_GROQ_API_KEY"

# Constants
CSV_FILES = {
    "steps": "Advertiser Actions - Steps.csv",
    "advertiser": "data/Advertiser Actions - Advertiser.csv"
}

# CSV Data Loading with error handling and caching
@lru_cache(maxsize=4)
def load_csv_data() -> Dict[str, Any]:
    """Load and cache CSV data with robust error handling"""
    csv_data = {}
    
    for key, filepath in CSV_FILES.items():
        try:
            df = pd.read_csv(filepath)
            csv_data[f"df_{key}"] = df
            csv_data[f"summary_{key}"] = df.head(10).to_string(index=False)
            logger.info(f"Successfully loaded CSV: {filepath}")
        except FileNotFoundError:
            logger.error(f"CSV file not found: {filepath}")
            csv_data[f"summary_{key}"] = f"Error: CSV file '{filepath}' not found."
        except pd.errors.EmptyDataError:
            logger.error(f"Empty CSV file: {filepath}")
            csv_data[f"summary_{key}"] = f"Error: CSV file '{filepath}' is empty."
        except pd.errors.ParserError:
            logger.error(f"CSV parsing error: {filepath}")
            csv_data[f"summary_{key}"] = f"Error: Could not parse '{filepath}'. Invalid CSV format."
        except Exception as e:
            logger.error(f"Unexpected error loading {filepath}: {str(e)}")
            csv_data[f"summary_{key}"] = f"Error loading '{filepath}': {str(e)}"
    
    return csv_data

# Initialize CSV data
csv_data = load_csv_data()

# Init LLaMA-3 with Groq and proper callback handling
callback_manager = CallbackManager([StdOutCallbackHandler()])
try:
    llm = init_chat_model(
        "llama3-8b-8192", 
        model_provider="groq",
        callback_manager=callback_manager
    )
    logger.info("Successfully initialized LLaMA-3 model with Groq")
except Exception as e:
    logger.error(f"Failed to initialize LLM: {e}")
    # Fallback option could be added here

# Create system prompt with CSV data
def get_system_prompt() -> str:
    """Generate the system prompt with CSV data summaries"""
    csv_info = "\n\n".join([
        f"{key.replace('summary_', '').upper()} DATA:\n{value}" 
        for key, value in csv_data.items() 
        if key.startswith("summary_")
    ])
    
    return f"""You are a sophisticated platform guide that excels at providing precise information to users based on documentation.

You have access to the following data from CSV files:

{csv_info}

INSTRUCTIONS:
1. Answer user questions based on both the CSV data and prior conversation history
2. Use clear, strategic, and business-relevant language
3. Provide specific examples from the data when relevant
4. If the data doesn't contain the answer, be honest about the limitations
5. Keep responses concise and focused on the user's question

Respond directly without phrases like "according to the data" or "the document shows" - just provide the information as if you inherently know it."""

# Prompt Template with better context
prompt_template = ChatPromptTemplate.from_messages([
    ("system", get_system_prompt()),
    MessagesPlaceholder(variable_name="messages")
])

# State Schema
class State(TypedDict):
    messages: Annotated[Sequence[HumanMessage], add_messages]
    metadata: Dict[str, Any]

# Build Graph with enhanced error handling and performance monitoring
workflow = StateGraph(state_schema=State)

def call_model(state: State) -> Dict:
    """Process user input and generate response with the LLM"""
    start_time = time.time()

    try:
        # Get chat history
        messages = state["messages"]
        metadata = state.get("metadata", {})

        # Inject history into prompt
        prompt = prompt_template.invoke({"messages": messages})

        # Get model response
        response = llm.invoke(prompt)

        # Log performance metrics
        processing_time = time.time() - start_time
        logger.info(f"Model response generated in {processing_time:.2f} seconds")

        # Update metadata
        metadata["last_processing_time"] = processing_time

        # Append model response to the messages
        updated_messages = messages + [response]

        # Keep only the last 4 messages (2 user + 2 AI)
        trimmed_messages = updated_messages[-4:]

        return {
            "messages": trimmed_messages,
            "metadata": metadata
        }

    except Exception as e:
        logger.error(f"Model error: {str(e)}", exc_info=True)
        error_message = "I apologize, but I'm having trouble processing your request. Please try again in a moment."

        fallback_messages = state["messages"][-3:]  # Get the last few in case of crash
        return {
            "messages": fallback_messages + [AIMessage(content=error_message)],
            "metadata": state.get("metadata", {})
        }

# Add node and define workflow
workflow.add_node("model", call_model)
workflow.add_edge(START, "model")

# Set up memory persistence
memory = MemorySaver()
langgraph_app = workflow.compile(checkpointer=memory)

# Enhanced chat function with session management
def generate_bot_response(user_input: str, session_id: str = "default") -> str:
    """Generate a response to user input with proper session tracking"""
    thread_id = f"csv-thread-{session_id}"
    
    try:
        # Initialize or continue conversation
        messages = [HumanMessage(content=user_input)]
        
        state = {
            "messages": messages,
            "metadata": {"session_id": session_id}
        }
        
        # Invoke the graph with the session's thread_id
        response = langgraph_app.invoke(
            state, 
            config={"configurable": {"thread_id": thread_id}}
        )
        
        return response['messages'][-1].content
    
    except Exception as e:
        logger.error(f"Response generation error: {str(e)}", exc_info=True)
        return "I'm sorry, something went wrong with my processing. Please try again or rephrase your question."

# Flask App with improved structure
app = Flask(__name__, static_folder='../frontend/build', static_url_path='/')

# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({
        'status': 'ok',
        'timestamp': time.time(),
        'version': '1.0.0'
    })

# Serve static files
@app.route('/')
def index():
    """Serve the frontend application"""
    return app.send_static_file('index.html')

# API endpoint for chat
@app.route('/api/messages', methods=['POST'])
def handle_message():
    """Handle incoming chat messages"""
    try:
        data = request.json
        
        # Validate input
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        user_input = data.get('message')
        if not user_input or not isinstance(user_input, str):
            return jsonify({'error': 'Invalid or missing message'}), 400
            
        # Get session ID or use default
        session_id = data.get('session_id', 'default')
        
        # Generate response
        response = generate_bot_response(user_input, session_id)
        
        return jsonify({
            'user_message': user_input,
            'bot_response': response,
            'timestamp': time.time(),
            'session_id': session_id
        })
        
    except Exception as e:
        logger.error(f"API error: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Internal server error',
            'message': 'The server encountered an unexpected condition'
        }), 500

# Catch-all route for SPA
@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors by serving the frontend app"""
    return app.send_static_file('index.html')

# CSV data refreshing endpoint (admin only - would need auth in production)
@app.route('/api/refresh-data', methods=['POST'])
def refresh_data():
    """Force refresh of cached CSV data"""
    try:
        # Clear cache and reload data
        load_csv_data.cache_clear()
        global csv_data
        csv_data = load_csv_data()
        
        return jsonify({
            'status': 'success',
            'message': 'CSV data refreshed successfully'
        })
    except Exception as e:
        logger.error(f"Data refresh error: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Failed to refresh data',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    # Check data is loaded before starting server
    if not any('df_' in key for key in csv_data):
        logger.warning("No CSV data was successfully loaded. Check file paths and formats.")
    
    # Start server with production-ready settings
    app.run(
        host='0.0.0.0',  # Listen on all interfaces
        port=int(os.environ.get('PORT', 5000)),
        debug=os.environ.get('FLASK_ENV', 'production') == 'development'
    )
