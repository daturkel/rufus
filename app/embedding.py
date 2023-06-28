import numpy as np
import pandas as pd
from tqdm import tqdm

from .nn import get_sentence_embeddings, get_tokenizer_and_model


class EmbeddedError(Exception):
    pass


class NotEmbeddedError(Exception):
    pass


class Embedding:
    def __init__(
        self,
        items: pd.DataFrame,
        key_column: str,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.items = items
        self.key_column = key_column
        self.model = model
        self.tokenizer, self.model = get_tokenizer_and_model(model)
        self._embeddings: np.ndarray | None = None

    def _raise_if_not_embedded(self):
        if self._embeddings is None:
            raise NotEmbeddedError("No embeddings available. Run embed() first.")

    def embed(self, batch_size: int = 1):
        if self._embeddings is not None:
            raise EmbeddedError("Already embedded!")

        embeddings = []
        for i in tqdm(range(0, len(self.items), batch_size)):
            these_sentences = list(
                self.items.loc[i : i + batch_size - 1, self.key_column]
            )
            these_embeddings = get_sentence_embeddings(
                these_sentences, self.tokenizer, self.model
            )
            embeddings.append(these_embeddings)
        self._embeddings = np.concatenate(embeddings)

    @property
    def embeddings(self):
        self._raise_if_not_embedded()
        return self._embeddings

    def nn_index(self, query: str) -> tuple[np.ndarray, np.ndarray]:
        self._raise_if_not_embedded()
        query_embedding = get_sentence_embeddings([query], self.tokenizer, self.model)[
            0
        ]
        scores = self.embeddings @ query_embedding
        index = np.argsort(-scores)
        return index, scores

    def nn_items(
        self,
        query: str,
        item_cols_to_return: list[str] | None = None,
        return_score_col: bool = True,
        score_col_name: str = "score",
    ) -> pd.DataFrame:
        index, scores = self.nn_index(query)
        columns = (
            list(self.items.columns)
            if item_cols_to_return is None
            else item_cols_to_return
        )

        if return_score_col:
            columns += [score_col_name]
            return self.items.assign(**{score_col_name: scores}).loc[index, columns]

        return self.items.loc[index, columns]
