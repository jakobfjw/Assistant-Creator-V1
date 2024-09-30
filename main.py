import streamlit as st
import os
from openai import OpenAI
import tempfile
from dotenv import load_dotenv
import logging
from typing import List, Tuple, Optional, IO, Union
from assistant_instructions import get_assistant_instructions
from messages import MESSAGES, PHASE_NAMES, PITCH_DECK_MESSAGES, PITCH_DECK_PHASE_NAMES  # Import MESSAGES and PHASE_NAMES
from streamlit_modal import Modal

# Load environment variables and set up logging
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client with API key from .env
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Streamlit app configuration
st.set_page_config(page_title="OpenAI Assistant Manager", layout="wide")

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state['vector_store'] = None
if 'assistants' not in st.session_state:
    st.session_state['assistants'] = []
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'refresh_files' not in st.session_state:
    st.session_state['refresh_files'] = False
if 'assistant' not in st.session_state:
    st.session_state['assistant'] = None
if 'message_index' not in st.session_state:
    st.session_state['message_index'] = 0
if 'conversation_history' not in st.session_state:
    st.session_state['conversation_history'] = []
if 'assistant_chat_histories' not in st.session_state:
    st.session_state['assistant_chat_histories'] = {}

# Update this constant to use relative paths
GRANT_ASSISTANT_REQUIRED_FILES = [
    "required_files/COMPLETED IN PHASE 1 - SOLINTEGRA AS.pdf",
    "required_files/COMPLETED IN PHASE 1 - ERKE TEKNOLOGI AS.pdf"
]

# At the top of the file, update this constant
PITCH_DECK_CREATOR_REQUIRED_FILES = [
    "required_files/Pitchdeck Questions.pdf",
    "required_files/The Overall Pitch Deck Guide.pdf",
    "required_files/VB AI pitchdeck.pdf"
]

def create_vector_store(name: str) -> Optional[dict]:
    """Create a new vector store"""
    try:
        vector_store = client.beta.vector_stores.create(name=name)
        logger.info(f"Vector store '{name}' created successfully")
        return vector_store
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        st.error(f"Error creating vector store: {str(e)}")
        return None

def upload_files_to_vector_store(vector_store_id: str, files: List[Union[IO, Tuple[str, Tuple[str, bytes]]]]) -> Optional[List[dict]]:
    """Upload files to the vector store"""
    try:
        uploaded_files = []
        with tempfile.TemporaryDirectory() as temp_dir:
            for file in files:
                if isinstance(file, tuple):  # For required files
                    _, (filename, content) = file
                    file_path = os.path.join(temp_dir, filename)
                    with open(file_path, "wb") as f:
                        f.write(content)
                else:  # For uploaded files
                    file_path = os.path.join(temp_dir, file.name)
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())

                with open(file_path, "rb") as f:
                    file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
                        vector_store_id=vector_store_id, files=[f]
                    )
                uploaded_files.append(file_batch)
                logger.info(f"File {os.path.basename(file_path)} uploaded successfully to vector store {vector_store_id}")

        return uploaded_files
    except Exception as e:
        logger.error(f"Error uploading files: {str(e)}")
        st.error(f"Error uploading files: {str(e)}")
        return None

def create_assistant(name: str, assistant_type: str, vector_store_id: str) -> Optional[dict]:
    """Create a new OpenAI assistant"""
    try:
        instructions = get_assistant_instructions(assistant_type)
        assistant = client.beta.assistants.create(
            name=name,
            instructions=instructions,
            model="gpt-4o",
            tools=[{"type": "file_search"}],
            metadata={"type": assistant_type}
        )
        assistant = client.beta.assistants.update(
            assistant_id=assistant.id,
            tool_resources={"file_search": {"vector_store_ids": [vector_store_id]}},
        )
        st.session_state['assistants'].append(assistant)
        logger.info(f"Assistant '{name}' created successfully")
        return assistant
    except Exception as e:
        logger.error(f"Error creating assistant: {str(e)}")
        st.error(f"Error creating assistant: {str(e)}")
        return None

def chat_with_assistant(assistant_id: str, user_message: str) -> Optional[str]:
    """Chat with the selected assistant"""
    try:
        if 'thread_id' not in st.session_state:
            thread = client.beta.threads.create()
            st.session_state['thread_id'] = thread.id
        
        client.beta.threads.messages.create(
            thread_id=st.session_state['thread_id'],
            role="user",
            content=user_message
        )
        run = client.beta.threads.runs.create(
            thread_id=st.session_state['thread_id'],
            assistant_id=assistant_id
        )
        
        # Wait for the run to complete
        while run.status != "completed":
            run = client.beta.threads.runs.retrieve(thread_id=st.session_state['thread_id'], run_id=run.id)
        
        # Retrieve messages
        messages = client.beta.threads.messages.list(thread_id=st.session_state['thread_id'])
        return messages.data[0].content[0].text.value
    except Exception as e:
        logger.error(f"Error chatting with assistant: {str(e)}")
        st.error(f"Error chatting with assistant: {str(e)}")
        return None

def list_vector_store_files(vector_store_id: str) -> Optional[List[Tuple[str, str]]]:
    """List files in the vector store"""
    try:
        vector_store_files = client.beta.vector_stores.files.list(vector_store_id=vector_store_id)
        return [(file.id, str(file)) for file in vector_store_files.data] if vector_store_files else None
    except Exception as e:
        logger.error(f'Error listing vector store files: {str(e)}')
        st.error(f'Error listing vector store files: {str(e)}')
        return None

def delete_vector_store_file(vector_store_id: str, file_id: str) -> Optional[dict]:
    """Delete a file from the vector store"""
    try:
        deleted_file = client.beta.vector_stores.files.delete(
            vector_store_id=vector_store_id,
            file_id=file_id
        )
        if deleted_file:
            st.session_state['refresh_files'] = True
            logger.info(f"File {file_id} deleted successfully from vector store {vector_store_id}")
        return deleted_file
    except Exception as e:
        logger.error(f'Error deleting file: {str(e)}')
        st.error(f'Error deleting file: {str(e)}')
        return None

def display_tabs(conversation_history):
    if conversation_history:
        tab_titles = [title for title, _ in conversation_history]
        tabs = st.tabs(tab_titles)
        
        for i, (phase_name, content) in enumerate(conversation_history):
            with tabs[i]:
                st.markdown(content)
                
                if phase_name.startswith("Information Sufficiency Analysis"):
                    st.write("---")
                    if st.button("Analyze Additional Files", key=f"analyze_additional_{i}"):
                        message = "I have uploaded additional files. Please analyze them, absorb their information and regenerate your information sufficiency analysis."
                        with st.spinner("Assistant is analyzing additional files..."):
                            response = chat_with_assistant(st.session_state['assistant'].id, message)
                        if response:
                            conversation_history[i] = (phase_name, response)
                            st.rerun()
                
                if phase_name.startswith("Draft Answer Generation"):
                    st.write("---")
                    st.write("Provide feedback or ask for modifications:")
                    user_message = st.text_input("Your message:", key=f"user_message_draft_{i}")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Send", key=f"send_message_draft_{i}"):
                            if user_message:
                                prefixed_message = f"Please regenerate your draft answer with the following modification: {user_message}"
                                with st.spinner("Assistant is processing your feedback..."):
                                    response = chat_with_assistant(st.session_state['assistant'].id, prefixed_message)
                                
                                if response:
                                    conversation_history[i] = (phase_name, response)
                                    st.rerun()
                            else:
                                st.warning("Please enter a message before sending.")
                    
                    with col2:
                        if st.button("Next Question", key=f"next_question_{i}"):
                            next_message = "Next question please"
                            with st.spinner("Assistant is preparing the next draft answer..."):
                                response = chat_with_assistant(st.session_state['assistant'].id, next_message)
                            if response:
                                new_draft_answer_count = len([p for p in conversation_history if p[0].startswith("Draft Answer Generation")]) + 1
                                new_phase_name = f"Draft Answer Generation {new_draft_answer_count}"
                                conversation_history.append((new_phase_name, response))
                                st.rerun()
                    
                    if st.button("Show me all the questions", key=f"show_all_questions_{i}"):
                        show_questions_message = "Amazing! Please show me all the draft answers that have been generated so far."
                        with st.spinner("Assistant is retrieving all questions..."):
                            response = chat_with_assistant(st.session_state['assistant'].id, show_questions_message)
                        if response:
                            new_phase_name = "All Questions Summary"
                            conversation_history.append((new_phase_name, response))
                            st.rerun()

def chat_section():
    if st.session_state['assistant']:
        assistant_type = st.session_state['assistant'].metadata.get('type', 'Unknown')
        st.header(f"Chat with {st.session_state['assistant'].name} - {assistant_type}")
    else:
        st.header("Chat with Assistant")

    if st.session_state['assistants']:
        if st.session_state['assistant']:
            assistant_id = st.session_state['assistant'].id
            if assistant_id not in st.session_state['assistant_chat_histories']:
                st.session_state['assistant_chat_histories'][assistant_id] = {
                    'chat_history': [],
                    'conversation_history': [],
                    'message_index': 0,
                    'company_name': '',
                    'selected_slides': ''
                }
            
            current_chat = st.session_state['assistant_chat_histories'][assistant_id]

            if st.session_state['assistant'].metadata.get('type') in ["Grant Assistant", "Pitch Deck Creator"]:
                # Determine which set of messages and phase names to use
                if st.session_state['assistant'].metadata.get('type') == "Grant Assistant":
                    messages = MESSAGES
                    phase_names = PHASE_NAMES
                else:  # Pitch Deck Creator
                    messages = PITCH_DECK_MESSAGES
                    phase_names = PITCH_DECK_PHASE_NAMES

                    # Add company name input for Pitch Deck Creator
                    if not current_chat.get('company_name'):
                        company_name = st.text_input("Enter the name of the company you want to create a pitch deck for:")
                        if st.button("Submit Company Name"):
                            if company_name:
                                current_chat['company_name'] = company_name
                                st.success(f"Company name set to: {company_name}")
                                
                                # Send only the first message to the assistant
                                initial_message = messages[0].format(company_name=company_name)
                                with st.spinner("Assistant is analyzing the company information..."):
                                    response = chat_with_assistant(st.session_state['assistant'].id, initial_message)
                                if response:
                                    current_chat['conversation_history'].append((phase_names[0], response))
                                    current_chat['message_index'] = 1
                                    st.rerun()
                            else:
                                st.warning("Please enter a company name.")
                        return  # Exit the function to wait for company name input

                # Create tabs for each phase
                tabs = st.tabs(phase_names[:current_chat['message_index']+1])

                for i, (phase, content) in enumerate(current_chat['conversation_history']):
                    with tabs[i]:
                        st.markdown(content)

                        # Add phase-specific interactions
                        if phase == "1. Business Report Generation" and st.session_state['assistant'].metadata.get('type') == "Pitch Deck Creator":
                            if i == len(current_chat['conversation_history']) - 1:  # If this is the last phase in the history
                                if st.button("Next Phase", key=f"next_phase_{i}"):
                                    if current_chat['message_index'] < len(messages):
                                        message = messages[current_chat['message_index']]
                                        with st.spinner("Assistant is preparing the next phase..."):
                                            response = chat_with_assistant(st.session_state['assistant'].id, message)
                                        if response:
                                            phase_name = phase_names[current_chat['message_index']]
                                            current_chat['conversation_history'].append((phase_name, response))
                                            current_chat['message_index'] += 1
                                            st.rerun()
                                    else:
                                        st.error("All phases have been completed.")

                        elif phase == "2. Slide Suggestion" and st.session_state['assistant'].metadata.get('type') == "Pitch Deck Creator":
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("Select All Slides"):
                                    current_chat['selected_slides'] = "All slides"
                            with col2:
                                slide_numbers = st.text_input("Or enter specific slide numbers:")
                                if slide_numbers:
                                    current_chat['selected_slides'] = slide_numbers

                            # Add text box for suggesting additional slides
                            additional_slides = st.text_input("Suggest additional slides:")
                            if st.button("Submit Additional Slides"):
                                if additional_slides:
                                    message = f"Please regenerate your list of suggested slides now including these additional slide ideas: {additional_slides}"
                                    with st.spinner("Assistant is updating slide suggestions..."):
                                        response = chat_with_assistant(st.session_state['assistant'].id, message)
                                    if response:
                                        current_chat['conversation_history'][i] = (phase, response)
                                        st.rerun()
                                else:
                                    st.warning("Please enter additional slide suggestions before submitting.")

                            if current_chat.get('selected_slides'):
                                st.success(f"Selected slides: {current_chat['selected_slides']}")
                                if st.button("Proceed with Selected Slides"):
                                    message = messages[current_chat['message_index']].format(selected_slides=current_chat['selected_slides'])
                                    with st.spinner("Assistant is analyzing the selected slides..."):
                                        response = chat_with_assistant(st.session_state['assistant'].id, message)
                                    if response:
                                        current_chat['conversation_history'].append((phase_names[current_chat['message_index']], response))
                                        current_chat['message_index'] += 1
                                        st.rerun()

                        elif phase == "3. Information Evaluation" and st.session_state['assistant'].metadata.get('type') == "Pitch Deck Creator":
                            st.write("---")  # Add a separator for clarity
                            st.subheader("Additional Actions")
                            if st.button("I have uploaded additional information", key=f"upload_info_{i}"):
                                message = "I have just uploaded additional files, please analyze these files and regenerate your formal information gap analysis report, now reflecting the additional information you've extracted from the uploaded files."
                                with st.spinner("Assistant is analyzing new files and updating the report..."):
                                    response = chat_with_assistant(st.session_state['assistant'].id, message)
                                if response:
                                    current_chat['conversation_history'][i] = (phase, response)
                                    st.rerun()

                            # Add text box for user to submit answers to assistant's questions
                            user_answers = st.text_area("Your answers to the assistant's questions:", height=200)
                            if st.button("Submit Answers"):
                                if user_answers:
                                    message = f"Here are the answers to the questions you've asked: {user_answers}. Please regenerate your formal information gap analysis report, now reflecting the additional information you've extracted from the answers I've provided."
                                    with st.spinner("Assistant is updating the information gap analysis..."):
                                        updated_report = chat_with_assistant(st.session_state['assistant'].id, message)
                                    if updated_report:
                                        current_chat['conversation_history'][i] = (phase, updated_report)
                                        st.rerun()
                                else:
                                    st.warning("Please provide answers before submitting.")
                            
                            # Add "Draft Pitchdeck Slides" button only in the Information Evaluation phase
                            if st.button("Draft Pitchdeck Slides", key="draft_slides_button"):
                                if current_chat['message_index'] < len(messages):
                                    message = messages[current_chat['message_index']]
                                    with st.spinner("Assistant is drafting pitchdeck slides..."):
                                        response = chat_with_assistant(st.session_state['assistant'].id, message)
                                    if response:
                                        phase_name = phase_names[min(current_chat['message_index'], len(phase_names) - 1)]
                                        current_chat['conversation_history'].append((phase_name, response))
                                        current_chat['message_index'] += 1
                                        st.rerun()
                                    else:
                                        st.error(f"Failed to get a response from the assistant. Please try again.")
                                else:
                                    st.error("All phases have been completed. No more messages to send.")

                        elif phase == "4. Draft Slide Generation" and st.session_state['assistant'].metadata.get('type') == "Pitch Deck Creator":
                            st.write("---")  # Add a separator for clarity
                            st.subheader("Provide Additional Information")
                            additional_info = st.text_area("Enter additional information for the pitch deck:", height=200)
                            if st.button("Submit Additional Information"):
                                if additional_info:
                                    message = f"Here is the additional information you've required, please extract and absorb this information and regenerate the entire content for the pitchdeck, as before, reflecting this new information: {additional_info}"
                                    with st.spinner("Assistant is updating the pitch deck content..."):
                                        updated_content = chat_with_assistant(st.session_state['assistant'].id, message)
                                    if updated_content:
                                        current_chat['conversation_history'][i] = (phase, updated_content)
                                        st.rerun()
                                else:
                                    st.warning("Please provide additional information before submitting.")
                            
                            st.write("---")  # Add another separator for clarity
                            st.info("If you have additional files to upload, please use the 'Upload Additional Files' section in the sidebar.")
                            if st.button("Analyze New Files and Regenerate Pitch Deck"):
                                message = "I have just uploaded additional files, please analyze these files and regenerate your Pitch Deck Content, now reflecting the additional information you've extracted from the uploaded files."
                                with st.spinner("Assistant is analyzing new files and updating the pitch deck..."):
                                    updated_content = chat_with_assistant(st.session_state['assistant'].id, message)
                                if updated_content:
                                    current_chat['conversation_history'][i] = (phase, updated_content)
                                    st.rerun()

                # Remove the separate "Draft Pitchdeck Slides" button outside the tabs
                # Remove the "Next Phase" button for Pitch Deck Creator

            else:
                # Original chat functionality for other assistant types
                for message in current_chat['chat_history']:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                
                user_message = st.chat_input("Enter your message")
                if user_message:
                    current_chat['chat_history'].append({"role": "user", "content": user_message})
                    with st.chat_message("user"):
                        st.markdown(user_message)
                    
                    with st.chat_message("assistant"):
                        message_placeholder = st.empty()
                        response = chat_with_assistant(st.session_state['assistant'].id, user_message)
                        if response:
                            message_placeholder.markdown(response)
                            current_chat['chat_history'].append({"role": "assistant", "content": response})
        else:
            st.warning("Please select an Assistant to chat with.")
    else:
        st.info("Please create an Assistant first.")

def reset_grant_assistant_conversation(assistant_id: str):
    """Reset the Grant Assistant conversation"""
    if assistant_id in st.session_state['assistant_chat_histories']:
        st.session_state['assistant_chat_histories'][assistant_id] = {
            'chat_history': [],
            'conversation_history': [],
            'message_index': 0
        }
    if 'thread_id' in st.session_state:
        del st.session_state['thread_id']
    st.success("Conversation has been reset. You can start over from phase 1.")

def reset_chat():
    """Reset the chat history and thread for the current assistant"""
    if st.session_state['assistant']:
        assistant_id = st.session_state['assistant'].id
        if assistant_id in st.session_state['assistant_chat_histories']:
            st.session_state['assistant_chat_histories'][assistant_id] = {
                'chat_history': [],
                'conversation_history': [],
                'message_index': 0
            }
    if 'thread_id' in st.session_state:
        del st.session_state['thread_id']
    st.success("Chat history and thread have been reset for the current assistant.")

def upload_additional_files(vector_store_id: str):
    """Upload additional files to the vector store"""
    st.sidebar.subheader("Upload Additional Files")
    uploaded_files = st.sidebar.file_uploader("Choose files to upload", accept_multiple_files=True, type=['pdf', 'txt'], key="additional_files")
    if st.sidebar.button("Upload Additional Files"):
        if uploaded_files:
            file_batch = upload_files_to_vector_store(vector_store_id, uploaded_files)
            if file_batch:
                st.sidebar.success("Additional files uploaded successfully!")
                st.sidebar.write("File batch status:", file_batch.status)
                st.sidebar.write("File counts:", file_batch.file_counts)
                st.session_state['refresh_files'] = True
                st.rerun()
        else:
            st.sidebar.warning("Please select files to upload.")

def display_current_ids():
    """Display the current Assistant ID and Thread ID in the sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("Current Session Info")
    
    if 'assistant' in st.session_state and st.session_state['assistant']:
        st.sidebar.markdown(f"**Assistant ID:**")
        st.sidebar.code(st.session_state['assistant'].id)
    else:
        st.sidebar.markdown("**Assistant ID:** Not selected")
    
    if 'thread_id' in st.session_state:
        st.sidebar.markdown(f"**Thread ID:**")
        st.sidebar.code(st.session_state['thread_id'])
    else:
        st.sidebar.markdown("**Thread ID:** Not created")

def create_assistant_popup():
    st.write(f"Current working directory: {os.getcwd()}")  # Debug line
    
    modal = Modal(key="create_assistant_modal", title="Create Assistant and Upload Files")
    
    open_modal = st.sidebar.button("Create New Assistant", key="open_create_assistant_modal")
    
    if open_modal:
        modal.open()

    if modal.is_open():
        with modal.container():
            # Step 1: Upload Files
            uploaded_files = st.file_uploader("Choose files to upload", accept_multiple_files=True, type=['pdf', 'txt'], key="create_assistant_file_uploader")
            
            # Step 2: Enter Assistant Name and Select Type
            assistant_name = st.text_input("Enter Assistant Name (Hint: Name of the company)", key="create_assistant_name")
            assistant_types = list(get_assistant_instructions("").keys())
            assistant_type = st.selectbox("Select Assistant Type", assistant_types, key="create_assistant_type")
            
            # Step 3: Create Vector Store, Upload Files, and Create Assistant
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Create Assistant", key="create_assistant_button"):
                    if assistant_name and assistant_type:
                        with st.spinner("Creating assistant... This may take a few moments."):
                            # Create Vector Store
                            vector_store = create_vector_store(assistant_name)
                            if vector_store:
                                st.session_state['vector_store'] = vector_store
                                st.success(f"Vector Store '{assistant_name}' created successfully!")
                                
                                # Prepare files for upload
                                files_to_upload = []
                                for uploaded_file in uploaded_files:
                                    files_to_upload.append(uploaded_file)
                                
                                # Handle required files based on assistant type
                                required_files = []
                                if assistant_type == "Grant Assistant":
                                    required_files = GRANT_ASSISTANT_REQUIRED_FILES
                                elif assistant_type == "Pitch Deck Creator":
                                    required_files = PITCH_DECK_CREATOR_REQUIRED_FILES

                                required_files_found = True
                                for required_file in required_files:
                                    full_path = os.path.join(os.getcwd(), required_file)
                                    st.write(f"Searching for file: {full_path}")  # Debug line
                                    if os.path.exists(full_path):
                                        with open(full_path, "rb") as f:
                                            file_content = f.read()
                                            files_to_upload.append(("file", (os.path.basename(full_path), file_content)))
                                        st.success(f"Required file '{os.path.basename(full_path)}' found and added for upload.")
                                    else:
                                        st.error(f"Required file '{full_path}' not found.")
                                        required_files_found = False
                                
                                if not required_files_found:
                                    st.error("Some required files are missing. Please ensure all required files are in the correct directory.")
                                    return  # Exit the function if required files are missing

                                # Upload Files
                                if files_to_upload:
                                    file_batches = upload_files_to_vector_store(vector_store.id, files_to_upload)
                                    if file_batches:
                                        st.success("Files uploaded successfully!")
                                        for i, file_batch in enumerate(file_batches):
                                            st.write(f"File {i+1} batch status:", file_batch.status)
                                            st.write(f"File {i+1} counts:", file_batch.file_counts)
                                        
                                        # Create Assistant
                                        assistant = create_assistant(assistant_name, assistant_type, vector_store.id)
                                        if assistant:
                                            st.session_state['assistant'] = assistant
                                            st.success(f"Assistant '{assistant_name}' created successfully!")
                                            modal.close()
                                            st.rerun()
                                    else:
                                        st.error("Failed to upload files. Assistant creation aborted.")
                                else:
                                    st.error("No files to upload. Assistant creation aborted.")
                            else:
                                st.error("Failed to create Vector Store. Assistant creation aborted.")
                    else:
                        if not assistant_name:
                            st.warning("Please enter a name for the Assistant.")
                        if not assistant_type:
                            st.warning("Please select a type for the Assistant.")
            
            with col2:
                if st.button("Close", key="close_create_assistant_modal"):
                    modal.close()

def get_assistant_vector_store_id(assistant):
    """Get the vector store ID associated with the assistant"""
    if assistant and hasattr(assistant, 'tool_resources'):
        file_search = assistant.tool_resources.file_search
        if file_search and file_search.vector_store_ids:
            return file_search.vector_store_ids[0]
    return None

def main():
    if 'refresh_files' not in st.session_state:
        st.session_state['refresh_files'] = False

    # Add VB AI logo to the top of the sidebar
    st.sidebar.markdown("""
        <div style='text-align: center; margin-bottom: 20px;'>
            <h1 style='font-size: 3em; font-weight: bold; color: #1E90FF;'>VB AI</h1>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar title
    st.sidebar.title("Assistant Manager")

    # Create Assistant button
    create_assistant_popup()

    # Move "Select an Assistant" to sidebar
    if st.session_state['assistants']:
        st.sidebar.subheader("Select an Assistant")
        assistant_options = [f"{assistant.metadata.get('type', 'Unknown')} - {assistant.name}" for assistant in st.session_state['assistants']]
        selected_assistant_option = st.sidebar.selectbox("Choose an assistant", assistant_options, key="assistant_select")
        
        selected_assistant_name = selected_assistant_option.split(" - ", 1)[1]
        selected_assistant = next((a for a in st.session_state['assistants'] if a.name == selected_assistant_name), None)
        
        if selected_assistant:
            st.session_state['assistant'] = selected_assistant
            
            # Get the vector store ID for the selected assistant
            vector_store_id = get_assistant_vector_store_id(selected_assistant)
            
            if vector_store_id:
                st.session_state['vector_store'] = client.beta.vector_stores.retrieve(vector_store_id)

                # Display Vector Store Files in a dropdown
                with st.sidebar.expander("Vector Store Files", expanded=False):
                    files = list_vector_store_files(vector_store_id)
                    if files:
                        for file_id, file_info in files:
                            col1, col2 = st.columns([3, 1])
                            col1.text(file_info)
                            if col2.button('Delete', key=f'delete_{file_id}'):
                                if delete_vector_store_file(vector_store_id, file_id):
                                    st.success(f'File deleted successfully!')
                                    st.rerun()
                    else:
                        st.text('No files in the vector store')

                # Add the upload additional files function to the sidebar
                upload_additional_files(vector_store_id)

                # Display current Assistant ID and Thread ID
                display_current_ids()
            else:
                st.sidebar.warning("No vector store associated with this assistant.")
        else:
            st.sidebar.warning("Please select an Assistant.")
    else:
        st.sidebar.info("No assistants available. Please create an assistant.")

    # Add Reset Chat button to the bottom of the sidebar
    st.sidebar.markdown("---")
    if st.sidebar.button("Reset Conversation"):
        reset_chat()

    # Main content
    chat_section()

if __name__ == "__main__":
    main()