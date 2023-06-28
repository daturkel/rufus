from pydantic import BaseModel


class SearchRequest(BaseModel):
    query: str


class SearchResultItem(BaseModel):
    name: str
    score: float
    id: int | str
