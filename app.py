import os
import streamlit as st
from langchain.llms import HuggingFaceHub
from llm import similarity
from file_manipulation import make_directory_if_not_exists

from models import return_models, return_text2text_generation_models, return_task_name, return_text_generation_models
class LLM_Langchain():
    def __init__(self):
        dummy_parent = "google"
        self.models_count = return_text2text_generation_models(dummy_parent, True) + return_text_generation_models(dummy_parent, True)
        st.warning("Warning: Some models may not work and some models may require GPU to run")
        st.text(f"As of now there are {self.models_count} model available")
        st.text("Made with Langchain, StreamLit, Hugging Face and 💖")
        st.header('🦜🔗 One stop for Open Source Models')

        self.API_KEY = st.sidebar.text_input(
            'API Key',
            type='password',
            help="Type in your HuggingFace API key to use this app")

        self.task_name = st.sidebar.selectbox(
            label = "Choose the task you want to perform",
            options = return_task_name(),
            help="Choose your open source LLM to get started"
            )
        if self.task_name is None:
            model_parent_visibility = True
        else:
            model_parent_visibility = False
            
        model_parent_options = return_models(self.task_name)
        model_parent = st.sidebar.selectbox(
            label = "Choose your Source",
            options = model_parent_options,
            help="Choose your source of models",
            disabled=model_parent_visibility
            )

        if model_parent is None:
            model_name_visibility = True
        else:
            model_name_visibility = False
        if self.task_name == "text2text-generation":
            options = return_text2text_generation_models(model_parent)
        else:
            options = return_text_generation_models(model_parent)
        self.model_name = st.sidebar.selectbox(
            label = "Choose your Models",
            options = options,
            help="Choose your open source LLM to get started",
            disabled=model_name_visibility
            )

        self.temperature = st.sidebar.slider(
            label="Temperature",
            min_value=0.1,
            max_value=1.0,
            step=0.1,
            value=0.9,
            help="Set the temperature to get accurate results"
            )

        self.max_token_length = st.sidebar.slider(
            label="Token Length",
            min_value=32,
            max_value=1024,
            step=32,
            value=1024,
            help="Set the max tokens to get accurate results"
            )
        

        self.model_kwargs = {
            "temperature": self.temperature,
            "max_length": self.max_token_length
        }

        os.environ['HUGGINGFACEHUB_API_TOKEN'] = self.API_KEY


    def generate_response(self, input_text):
        
        
        llm = HuggingFaceHub(
            repo_id = self.model_name,
            model_kwargs = self.model_kwargs
        )

        return llm(input_text)
    

    def radio_button(self):
        options = ['FineTune', 'Inference']
        selected_option = st.radio(
            label="Choose your options",
            options=options
            )
        return selected_option
    

    def pdf_uploader(self):
        if self.selected_option == "Inference":
            self.uploader_visibility = True
        else:
            self.uploader_visibility = False

        self.file_upload_status = st.file_uploader(
            label="Upload PDF file",
            disabled=self.uploader_visibility
        )
        make_directory_if_not_exists('assets/')

        if self.file_upload_status is not None:
            self.pdf_file_path = f"assets/{self.file_upload_status.name}"
        
            with open(self.pdf_file_path, "wb") as f:
                f.write(self.file_upload_status.getbuffer())
            st.write("File Uploaded Successfully")

    def form_data(self):
        # with st.form('my_form'):
            try:
                if not self.API_KEY.startswith('hf_'):
                    st.warning('Please enter your API key!', icon='⚠')
                
                self.selected_option = self.radio_button()
                self.pdf_uploader()

                if self.selected_option == "FineTune":
                    if self.file_upload_status is None:
                        text_input_visibility = True
                    else:
                        text_input_visibility = False
                else:
                    text_input_visibility = False
                
                if "messages" not in st.session_state:
                    st.session_state.messages = []

                st.write(f"You are using {self.model_name} model")

                for message in st.session_state.messages:
                    with st.chat_message(message.get('role')):
                        st.write(message.get("content"))
                text = st.chat_input(disabled=text_input_visibility)
                
                if text:
                    st.session_state.messages.append(
                        {
                            "role":"user",
                            "content": text
                        }
                    )
                    with st.chat_message("user"):
                        st.write(text)
                    
                    if text.lower() == "clear":
                        del st.session_state.messages
                        return
                    if self.selected_option == 'FineTune':
                        result = similarity(self.pdf_file_path, self.model_name, self.model_kwargs, text)
                    else:
                        result = self.generate_response(text)
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": result
                        }
                    )
                    with st.chat_message('assistant'):
                        st.markdown(result)
                
            except Exception as e:
                st.error(e, icon="🚨")

model = LLM_Langchain()
model.form_data()