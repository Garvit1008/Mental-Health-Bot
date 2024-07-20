import streamlit as st
import google.generativeai as palm
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatGooglePalm
from langchain.embeddings import GooglePalmEmbeddings
import os

os.environ['PINECONE_API_KEY'] = '7a13cf1a-9206-426b-a502-0499d8928c2b'

GOOGLE_API_KEY = "AIzaSyDwrgGw2CW6Mcmwk6iS-_5mX6VPnKkuoeQ"
palm.configure(api_key=GOOGLE_API_KEY)
index_name = "mental-health-chatbot"
embeddings = GooglePalmEmbeddings(google_api_key=GOOGLE_API_KEY)
vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
llm = ChatGooglePalm(google_api_key=GOOGLE_API_KEY, temperature=0.7)
QA = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())

st.set_page_config(layout="wide")
col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.image(r"../streamlit/image.png", caption=None, use_column_width=True)

with col2:
    st.title("Mindful.AI")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    prompt = st.chat_input("How are you feeling today?")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display the user's message immediately
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get the response from Google PaLM
        try:
            # Make the API call to Google PaLM
            response = QA({'query': prompt})

            # Extract the response text
            assistant_response = response['result']

            # Add assistant message to chat history
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})

            # Display the assistant's response immediately
            with st.chat_message("assistant"):
                st.markdown(assistant_response)

        except Exception as e:
            st.error(f"An error occurred: {e}")
