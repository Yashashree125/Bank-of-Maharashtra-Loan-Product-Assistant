from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import FAISS_DIR, MODEL_NAME, GROQ_API_KEY, LLM_MODEL
from .rag_pipeline import search_faiss
from .utils.faiss_utils import load_faiss_index
from .utils.llm_utils import generate_answer
from sentence_transformers import SentenceTransformer
import markdown2  

app = FastAPI()

# Static + templates
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# Load model + FAISS index 
index, texts, metadata = load_faiss_index(FAISS_DIR)
model = SentenceTransformer(MODEL_NAME)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ask", response_class=HTMLResponse)
async def ask_question(request: Request, query: str = Form(...)):
    top_matches = search_faiss(query, index, texts, metadata, model, top_k=5)
    raw_answer = generate_answer(query, top_matches, GROQ_API_KEY, LLM_MODEL)

    answer_html = markdown2.markdown(raw_answer)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "query": query,
        "answer": answer_html,   
        "matches": top_matches
    })
