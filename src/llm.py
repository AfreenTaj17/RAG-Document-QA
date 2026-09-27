import os

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


# Load variables from .env
load_dotenv()


def get_openai_api_key():

    # Try Streamlit Cloud secrets first
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass

    # Fall back to local .env
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not configured. "
            "Add it to your local .env file or Streamlit Cloud Secrets."
        )

    return api_key


def create_llm():

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=get_openai_api_key()
    )

    return llm
