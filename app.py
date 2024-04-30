from flask import Flask, render_template, jsonify, request
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Store the chunks in vector store
from langchain_community.vectorstores import Chroma
from langchain.memory import ChatMessageHistory
# from openai import OpenAI
import streamlit as st
from langchain_cohere import ChatCohere
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_tool_calling_agent
from langchain_cohere.react_multi_hop.agent import create_cohere_react_agent
from langchain.agents import AgentExecutor
from langchain_cohere import CohereEmbeddings
from langchain_core.tools import Tool
import os

import asyncio

def set_event_loop_for_thread():
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError as e:
        if "There is no current event loop in thread" in str(e):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    return loop

app = Flask(__name__)

### Read the Gemini Key
f = open("keys/.cohere_api_key.txt")
cohere_api_key = f.read()

## Create the Model
chat_model = ChatCohere(cohere_api_key=cohere_api_key,model="command-r")


wrapper = DuckDuckGoSearchAPIWrapper(max_results=2)
duck_tool = DuckDuckGoSearchResults(api_wrapper=wrapper,source="site:webmd.com")

embedding_model=CohereEmbeddings(cohere_api_key=cohere_api_key,model="embed-english-light-v3.0")

# Setting a Connection with the ChromaDB
db_connection = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)

# Converting CHROMA db_connection to Retriever Object
retriever = db_connection.as_retriever()


medical_tool=create_retriever_tool(retriever,"medical_book",
                     "Search for information about Medical Treatment.")

## Define the Chat template
chat_template = ChatPromptTemplate.from_messages([
    # System Message Prompt Template
    SystemMessage(content="""You are a Helpful Medical AI Assistant. 
    You take the question from user. Your answer should be based on the specific question."""),
    
    MessagesPlaceholder(variable_name="chat_history"),
    # Human Message Prompt Template
    HumanMessagePromptTemplate.from_template(""" Use the following pieces of information to answer the user's question and provide the response in simple English language.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

     Question: {question}
     
     Only return the helpful answer below and nothing else.

    Answer: """),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
    
])


tools = [medical_tool,duck_tool]


agent = create_cohere_react_agent(llm=chat_model,tools=tools,prompt=chat_template)

agent_executor=AgentExecutor(agent=agent,tools=tools,verbose=True)


@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    # Set the event loop for the current thread
    set_event_loop_for_thread()
    msg = request.form["msg"]
    input = msg
    print(input)
    result=agent_executor.invoke({"chat_history":[],"question": input})
    print("Response : ", result.get('output'))
    return str(result.get('output'))

if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)