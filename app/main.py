import logging
import pickle

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

from .embedding import Embedding
from .models import SearchRequest

app = FastAPI(title="semlib-web-app")
templates = Jinja2Templates(directory="templates")

try:
    logging.info("loading embeddings")
    with open("./embeddings.pickle", "rb") as f:
        embeddings = pickle.load(f)
    logging.info("embeddings not found!")
except FileNotFoundError:
    logging.error("loading bookmarks")
    with open("./bookmarks.pickle", "rb") as f:
        bookmarks = pickle.load(f)

    logging.error("embedding bookmarks")
    embeddings = Embedding(bookmarks, key_column="title")
    embeddings.embed()
    logging.error("done embedding bookmarks")
    logging.error("pickling embeddings")
    with open("./embeddings.pickle", "wb") as f:
        pickle.dump(embeddings, f)
    logging.error("done pickling embeddings")


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.post("/search", response_class=HTMLResponse)
def search(request: Request, sr: SearchRequest):
    items = embeddings.nn_items(sr.query).iloc[:10].to_dict("records")
    return templates.TemplateResponse(
        "table_rows.html", context={"request": request, "items": items}
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
