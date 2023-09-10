import os
import streamlit as st
from langchain.llms import HuggingFaceHub


from models import return_models, return_text2text_generation_models, return_task_name, return_text_generation_models
dummy_parent = "google"
models_count = return_text2text_generation_models(dummy_parent, True) + return_text_generation_models(dummy_parent, True)
st.warning("Warning: Some models may not work and some models may require GPU to run")
st.text(f"As of now there are {models_count} model available")
st.text("Made with Langchain, StreamLit, Hugging Face and 💖")
st.header('🦜🔗 One stop for Open Source Models')

API_KEY = st.sidebar.text_input(
    'API Key',
    type='password',
    help="Type in your HuggingFace API key to use this app")

task_name = st.sidebar.selectbox(
    label = "Choose the task you want to perform",
    options = return_task_name(),
    help="Choose your open source LLM to get started"
    )
if task_name is None:
    model_parent_visibility = True
else:
    model_parent_visibility = False
    
model_parent_options = return_models(task_name)
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
if task_name == "text2text-generation":
    options = return_text2text_generation_models(model_parent)
else:
    options = return_text_generation_models(model_parent)
model_name = st.sidebar.selectbox(
    label = "Choose your Models",
    options = options,
    help="Choose your open source LLM to get started",
    disabled=model_name_visibility
    )

temperature = st.sidebar.slider(
    label="Temperature",
    min_value=0.1,
    max_value=1.0,
    step=0.1,
    value=0.9,
    help="Set the temperature to get accurate results"
    )

max_token_length = st.sidebar.slider(
    label="Token Length",
    min_value=32,
    max_value=1024,
    step=32,
    value=1024,
    help="Set the max tokens to get accurate results"
    )


os.environ['HUGGINGFACEHUB_API_TOKEN'] = API_KEY
def generate_response(input_text):
    
    model_kwargs = {
        "temperature": temperature,
        "max_length": max_token_length
    }
    llm = HuggingFaceHub(
        repo_id = model_name,
        model_kwargs = model_kwargs
    )

    st.info(llm(input_text))


with st.form('my_form'):
    try:
        text = st.text_area('Enter Your Prompt', 'What are the three key pieces of advice for learning how to code?')
        submitted = st.form_submit_button('Submit')
        if not API_KEY.startswith('hf_'):
            st.warning('Please enter your API key!', icon='⚠')
        if submitted and API_KEY.startswith('hf_'):
            with st.spinner("Running...."):
                generate_response(text)
    except Exception as e:
        st.error(e, icon="🚨")