import numpy.typing as npt
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel, AutoTokenizer

from ._base import ModelWrapper

# Drawn from sample code in model cards such as [this one](https://huggingface.co/thenlper/gte-large)


def _average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


class HFAveragePooled(ModelWrapper):
    def __init__(self, name: str, max_length: int = 512, normalize: bool = True):
        super().__init__(name)
        self.name = name  # repeating this line allows us to document it for pdoc
        """The name of the HuggingFace model to use."""
        self.model = AutoModel.from_pretrained(name)
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.max_length = max_length
        self.normalize = normalize

    def __call__(self, inputs: list[str], batch_size: int = 1) -> npt.NDArray:
        batch_dict = self.tokenizer(
            inputs,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        outputs = self.model(**batch_dict)
        embeddings = _average_pool(
            outputs.last_hidden_state,
            batch_dict["attention_mask"],  # type: ignore
        )

        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.detach().numpy()
