from flask import Flask, render_template, jsonify, request
from langchain_openai import OpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
import os

app = Flask(__name__)

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Set API keys
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Download embeddings model
embeddings = download_hugging_face_embeddings()

# Define Pinecone index
index_name = "medicalbots"

# Load existing index
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings 
)

# Define retriever and chains
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
llm = OpenAI(temperature=0.4, max_tokens=500)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form.get("msg") or request.args.get("msg")
    if not msg:
        return jsonify({"error": "No message received"}), 400

    print(f"User Input: {msg}")
    response = rag_chain.invoke({"input": msg})
    
    print("Response:", response.get("answer", "No response generated"))
    return jsonify({"answer": response.get("answer", "Error generating response")})

if __name__ == "__main__":
    app.run(debug=True)
