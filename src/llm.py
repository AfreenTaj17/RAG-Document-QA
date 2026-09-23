import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv()


def create_llm():
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    return llm


if __name__ == "__main__":
    llm = create_llm()

    response = llm.invoke(
        "What is machine learning? Answer in one sentence."
    )

    print(response.content)