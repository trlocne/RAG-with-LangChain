import os
import wget
import uuid
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langserve import add_routes

from src.base.llm_model import get_hf_llm
from src.rag.file_loader import Loader
from src.rag.vectorstore import VectorDB
from src.rag.offline_rag import Offline_RAG

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class InputQA(BaseModel):
    question: str = Field(..., title="Question to ask the model")

class OutputQA(BaseModel):
    answer: str = Field(..., title="Answer from the model")

vetorDB = VectorDB()

llm = get_hf_llm(
    model_name="meta-llama/Llama-3.1-8B",
    temperature=0.9,
    rope_scaling={
        "type": "linear",
        "factor": 4.0
    }
)

def init_data(data_dir, data_type):
    doc_loaded = Loader(file_type=data_type).load_dir(data_dir, workers=2)
    vetorDB.update_db(doc_loaded)

def gen_ai_chain():
    retriever = vetorDB.get_retriever()
    genai_chain = Offline_RAG(llm = llm).get_chain(retriever)
    return genai_chain

genai_docs = "./data_source/generate_ai"
init_data(data_dir=genai_docs , data_type ="pdf")

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API server using Langchain's Runnable interfaces",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

@app.get("/check")
async def check():
    return {"status": "ok"}

@app.post("/generative_ai", response_model=OutputQA)
async def generative_ai(inputs: InputQA):
    answer = gen_ai_chain().invoke(inputs.question)
    return {"answer": answer}

@app.post("/update_db/upload")
async def upload_pdf(file: UploadFile = File(...)):
    temp_dir = "./temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)

    file_path = os.path.join(temp_dir, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    doc_loaded = Loader(file_type="pdf").load(file_path, workers=2)
    vetorDB.update_db(doc_loaded)
    return {"status": f"Successfully updated the database with {len(doc_loaded)} page."}

@app.post("/update_db/url")
async def update_from_url(url: str = Form(...)):
    temp_dir = "./temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)

    random_filename = str(uuid.uuid4())
    file_path = f"{temp_dir}/{random_filename}.pdf"
    wget.download(url, out=file_path)

    doc_loaded = Loader(file_type="pdf").load(file_path, workers=2)
    vetorDB.update_db(doc_loaded)
    return {"status": f"Successfully updated the database with {len(doc_loaded)} page from URL."}

add_routes(app, gen_ai_chain(), playground_type="default", path="/generative_ai")