import numpy as np
import numpy.typing as npt

from .. import ResultSet


class ModelWrapper:
    """A base class for wrapping embedding models. The class instance is callable."""

    def __init__(self, name: str):
        """Create a new `ModelWrapper`.

        Args:
            name: A string name for the wrapper.
        """
        self.name = name
        """A string representing the wrapper."""

    def __call__(self, inputs: list[str], batch_size: int = 1) -> npt.NDArray:
        """Embed a list of string inputs using the model.

        Args:
            inputs: A list of strings to be embedded.
            batch_size: How many strings to embed at once. High batch size is faster but requires more memory. Defaults to 1.

        Raises:
            NotImplementedError: Not implemented in base class.

        Returns:
            A numpy array of embedded inputs.
        """
        raise NotImplementedError()


class CrossEncoderWrapper:
    """A base class for wrapping cross-encoder models. The class instance is callable and supports a `predict_one_to_many` method."""

    def __init__(self, name: str):
        """Create a new `CrossEncoderWrapper`.

        Args:
            name: A string name for the wrapper.
        """
        self.name = name

    def __call__(self, inputs: list[list[str]], batch_size: int = 1) -> npt.NDArray:
        """Score pairs of strings with the cross-encoder.

        Args:
            inputs: A list of lists of strings to be scored.
            batch_size: How many strings to score at once. High batch size is faster but requires more memory. Defaults to 1.

        Raises:
            NotImplementedError: Not implemented in the base class.

        Returns:
            A numpy array of embeddings.
        """
        raise NotImplementedError()

    def predict_one_to_many(
        self, query: str, items: list[str], batch_size: int = 1
    ) -> npt.ArrayLike:
        """A convenience method to score a single query document against multiple reference documents.

        Args:
            query: A string that will be scored against all the strings in `items`.
            items: A list of strings.
            batch_size: How many pairs of strings to score at once. Defaults to 1.

        Returns:
            A numpy array of scores for query-document pairs.
        """
        return self([[query, item] for item in items], batch_size=batch_size)

    def predict_one_to_many_results(
        self,
        query: str,
        items: list[str],
        ids: npt.NDArray[np.integer],
        batch_size: int = 1,
    ) -> ResultSet:
        """A convenience method to score a single query document against multiple reference documents.

        Args:
            query: A string that will be scored against all the strings in `items`.
            items: A list of tuples of ID/string pairs.
            batch_size: How many pairs of strings to score at once. Defaults to 1.

        Returns:
            A numpy array of scores for query-document pairs.
        """
        scores = self([[query, item] for item in items], batch_size=batch_size)
        return ResultSet(indices=ids, scores=scores)
