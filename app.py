import re
import os
from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv

from langchain_community.llms import HuggingFaceHub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from langchain_pinecone import PineconeVectorStore
from src.prompt import system_prompt
from src.helper import download_hugging_face_embeddings

# Initialize Flask app
app = Flask(__name__)

# Load environment variables
load_dotenv()

# Fetch API keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

# Set API key for Pinecone
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Initialize Hugging Face model
hf_model = HuggingFaceHub(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    huggingfacehub_api_token=HF_API_TOKEN,
    model_kwargs={"temperature": 0.4, "max_new_tokens": 500}
)

# Initialize embedding model
embeddings = download_hugging_face_embeddings()

# Load Pinecone index
index_name = "medicalbots"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Define retriever
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Define prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", "Context:\n{context}\n\nQuestion:\n{input}")
])

# Define LangChain chains
question_answer_chain = create_stuff_documents_chain(hf_model, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Wrapper for direct model generation
def hf_generate_response(prompt_text):
    return hf_model.invoke(prompt_text)

# Flask routes
@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form.get("msg") or request.args.get("msg")
    if not msg:
        return jsonify({"error": "No message received"}), 400

    print("User Input:", msg)

    # Retrieve relevant documents
    retrieved_docs = retriever.invoke(msg)
    context = " ".join([doc.page_content for doc in retrieved_docs])

    # Construct final prompt
    if msg.lower() in ["explain more", "tell me more", "continue", "go on"]:
        full_prompt = f"{context}\nAssistant: Continue explaining in more detail."
    else:
        full_prompt = f"{context}\nUser: {msg}\nAssistant:"

    # Generate response
    response_text = hf_generate_response(full_prompt)

    # Clean up output
    if "Assistant:" in response_text:
        response_text = response_text.split("Assistant:", 1)[-1].strip()
    else:
        response_text = response_text.strip()

    print("Response:", response_text)

    return jsonify({"answer": response_text})


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
