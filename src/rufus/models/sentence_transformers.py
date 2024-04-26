import numpy as np
import numpy.typing as npt
from sentence_transformers import CrossEncoder, SentenceTransformer

from ._base import CrossEncoderWrapper, ModelWrapper


class SentenceTransformerModelWrapper(ModelWrapper):
    """A wrapper around models from the `sentence-transformers` library."""

    def __init__(self, name: str):
        """Create a new `SentenceTransformerWrapper`.

        Args:
            name: The name of the model to load. Will be passed to a `SentenceTransformer` object as `model_name_or_path`.
        """
        super().__init__(name)
        self.name = name  # repeating this line allows us to document it for pdoc
        """The name of the (`sentence-transformers`) model to use."""
        self.model = SentenceTransformer(name)
        """A `SentenceTransformer` object used to embed items."""

    def __call__(
        self,
        inputs: list[str],
        batch_size: int = 1,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> npt.NDArray:
        """Embed a list of string inputs using the model.

        Args:
            inputs: A list of strings to be embedded.
            batch_size: How many strings to embed at once. High batch size is faster but requires more memory. Defaults to 1.

        Returns:
            A numpy array of embedded inputs.
        """
        outputs = self.model.encode(
            inputs,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=show_progress_bar,
        )
        return outputs


class SentenceTransformerCrossEncoderWrapper(CrossEncoderWrapper):
    def __init__(self, name: str, apply_sigmoid: bool = False):
        super().__init__(name)
        self.name: str = name
        self.model = CrossEncoder(name)
        self.apply_sigmoid = apply_sigmoid

    def __call__(
        self,
        inputs: list[list[str]],
        batch_size: int = 1,
        show_progress_bar: bool = False,
    ) -> npt.NDArray:
        scores = self.model.predict(
            inputs,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=show_progress_bar,
        )
        if self.apply_sigmoid:
            scores = 1 / (1 + np.exp(-scores))  # type: ignore
        return scores
