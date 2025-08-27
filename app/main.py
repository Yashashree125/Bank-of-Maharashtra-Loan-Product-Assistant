from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import FAISS_DIR, MODEL_NAME, GROQ_API_KEY, LLM_MODEL
from .rag_pipeline import create_embeddings, search_faiss
from .utils.faiss_utils import load_faiss_index
from .utils.llm_utils import generate_answer
import markdown2  

app = FastAPI()

# Static + templates
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# Init embeddings + FAISS index
embeddings = create_embeddings(MODEL_NAME)
vectordb = load_faiss_index(
    FAISS_DIR,
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ask", response_class=HTMLResponse)
async def ask_question(request: Request, query: str = Form(...)):
    top_matches = search_faiss(query, vectordb, top_k=5)
    raw_answer = generate_answer(query, top_matches, GROQ_API_KEY, LLM_MODEL)

    answer_html = markdown2.markdown(raw_answer)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "query": query,
        "answer": answer_html,
        "matches": [doc.page_content for doc in top_matches]
    })
