import os

import numpy as np
import numpy.typing as npt
import tiktoken
from openai import OpenAI

from ._base import ModelWrapper


class OpenAIEmbeddingsWrapper(ModelWrapper):
    """A wrapper around the OpenAI embeddings endpoint."""

    def __init__(self, name: str, api_key: str | None = None):
        """Create a new `OpenAIEmbeddingsWrapper`.

        Args:
            name: The name of the model to use.
            api_key: An OpenAI API key. If `None`, an attempt will be made to read a key from an `OPENAI_API_KEY` environment variable. Defaults to None.

        Raises:
            ValueError: No API key provided or found.
        """
        self.name = name  # repeating this line allows us to document it for pdoc
        """The name of the OpenAI embeddings model to use."""
        self.api_key: str
        """The OpenAI API key used for requests."""
        if api_key is not None:
            self.api_key = api_key
        else:
            try:
                self.api_key = os.environ["OPENAI_API_KEY"]
            except KeyError:
                raise ValueError(
                    "Either supply an api_key or set environment variable OPENAI_API_KEY"
                )
        self._client = OpenAI(api_key=self.api_key)

    def __call__(
        self, inputs: list[str], batch_size: int = 1, max_tokens: int | None = 8191
    ) -> npt.NDArray:
        """Embed a list of string inputs using the model.

        Args:
            inputs: A list of strings to be embedded.
            batch_size: Ignored for `OpenAIEmbeddingsWrapper`.
            max_tokens: Maximum number of tokens to send per input. If `None`, inputs will not be truncated, but `embed` may return an error if inputs are too long. Defaults to 8191.

        Returns:
            A numpy array of embedded inputs.
        """
        openai_inputs: list[str] | list[list[int]]
        if max_tokens is not None:
            encoding = tiktoken.encoding_for_model(self.name)
            # if limiting tokens, we must tokenize the input in advance and send a list[list[int]]
            encoded_inputs = encoding.encode_batch(inputs)
            openai_inputs = [input_[:max_tokens] for input_ in encoded_inputs]
        else:
            # send a list[str] and let OpenAI do tokenization
            openai_inputs = inputs

        response = self._client.embeddings.create(
            model=self.name,
            input=openai_inputs,
        )
        embeddings = np.array([obj["embedding"] for obj in response["data"]])  # type: ignore
        return embeddings
