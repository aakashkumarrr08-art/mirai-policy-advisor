
import os
import shutil
import tempfile

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

app = FastAPI(title="MirAI Student Policy Advisor")

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("GOOGLE_API_KEY missing. Check your .env file.")

DATA_DIR = "data"
DB_DIR = "chroma_db"
COLLECTION_NAME = "mirai_policy"

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=API_KEY,
)

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=API_KEY,
)

vector_store = None
retriever = None


class ChatRequest(BaseModel):
    question: str


@app.get("/")
async def home():
    return {"message": "MirAI Student Policy Advisor API is running"}


@app.post("/ingest")
async def ingest_pdf(file: UploadFile = File(...)):
    global vector_store, retriever

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
            temp_path = temp.name
            shutil.copyfileobj(file.file, temp)

        documents = PyPDFLoader(temp_path).load()

        if not documents:
            raise HTTPException(status_code=400, detail="The PDF contains no readable pages.")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunks = splitter.split_documents(documents)

        if not chunks:
            raise HTTPException(status_code=400, detail="No readable text found in the PDF.")

        if os.path.exists(DB_DIR):
            shutil.rmtree(DB_DIR)

        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=DB_DIR,
        )

        retriever = MultiQueryRetriever.from_llm(
            retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
            llm=llm,
        )

        return {
            "message": "PDF processed and indexed successfully.",
            "pages": len(documents),
            "chunks": len(chunks),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        await file.close()


@app.post("/chat")
async def chat(request: ChatRequest):
    if retriever is None:
        raise HTTPException(
            status_code=400,
            detail="Please upload and ingest the policy PDF first using /ingest.",
        )

    try:
        relevant_docs = await retriever.ainvoke(request.question)

        if not relevant_docs:
            return {
                "answer": "Sorry, I could not find this information in the MirAI Student Policy Handbook."
            }

        context = "\n\n".join(doc.page_content for doc in relevant_docs)

        prompt = ChatPromptTemplate.from_template(
            """
You are the MirAI Student Policy Advisor.

Answer the student's question using ONLY the policy context below.

Rules:
- Do not use outside knowledge or make up policy details.
- If the context does not contain the answer, politely say that the handbook
  does not provide enough information to answer.
- Give a clear, student-friendly answer.
- Mention a page number only if supported by the provided context metadata.

Policy context:
{context}

Student question:
{question}

Answer:
"""
        )

        chain = prompt | llm | StrOutputParser()

        answer = await chain.ainvoke(
            {"context": context, "question": request.question}
        )

        return {"answer": answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
