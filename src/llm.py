import os

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv()


def get_openai_api_key():

    # Streamlit Cloud
    if "OPENAI_API_KEY" in st.secrets:
        return st.secrets["OPENAI_API_KEY"]

    # Local development
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not configured."
        )

    return api_key


def create_llm():

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=get_openai_api_key()
    )

    return llm


if __name__ == "__main__":

    llm = create_llm()

    response = llm.invoke(
        "What is machine learning? "
        "Answer in one sentence."
    )

    print(response.content)
