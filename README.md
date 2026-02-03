How to make Basic Rag project?
- cd ~/projects
mkdir rag-chatbot
cd rag-chatbot
python3 -m venv venv
source venv/bin/activate
python --version
pip --version
pip install --upgrade pip
pip install fastapi uvicorn python-dotenv
pip install langchain langchain-openai openai \
            chromadb tiktoken \
            pypdf sentence-transformers
touch main.py .env .gitignore requirements.txt
mkdir data
echo -e "venv/\n.env\n__pycache__/" > .gitignore
------------
main.py
------------
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "RAG API is running 🚀"}

@app.post("/ask")
def ask(question: str):
    return {"answer": f"You asked: {question}"}
--------------------------------
Test: python -c "from main import app; print('OK')"
------------
.env
------------
https://platform.openai.com/settings/organization/api-keys
https://app.pinecone.io/organizations/-OV8XOXAX9CEk5DIMqgn/projects/a5c07b41-889c-47cb-b3c0-8a858d836e3d/keys
OPENAI_API_KEY=
PINECONE_API_KEY=
PINECONE_ENV=
------------
rag.py
------------
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
import pinecone
import os

pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENV")
)

llm = OpenAI()
embeddings = OpenAIEmbeddings()
-----------------------------
python -m uvicorn main:app --reload --port 8001
Run 8001 link and see your page.

####################################################################################
How to Rerun Rag project?
- Run inside your project folder.
-- which python
-- source venv/bin/activate
-- which pip
-- pip install fastapi uvicorn
-- pip install langchain openai chromadb tiktoken python-dotenv
-- pip list | grep fastapi
-- python -c "import fastapi; print(fastapi.__version__)"
-- python -m uvicorn main:app --reload --port 8001
You should see the URL on this command.

####################################################################################
Main commands:
source venv/bin/activate
pip install fastapi uvicorn
python -m uvicorn main:app --reload --port 8001

####################################################################################
http://127.0.0.1:8001
http://127.0.0.1:8001/docs
