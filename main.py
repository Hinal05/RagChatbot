from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "RAG API is running 🚀"}

@app.post("/ask")
def ask(question: str):
    return {"answer": f"You asked: {question}"}
