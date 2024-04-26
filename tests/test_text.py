from typing import Any, Callable

from rufus.text import (
    TextPipeline,
    lowercase,
    remove_characters,
    remove_punctuation,
    split_on_whitespace,
    substitute_characters,
)


class FunctionSpy:
    def __init__(self, fn: Callable):
        self.fn = fn
        self.history: list[Any] = []

    def __call__(self, *args, **kwargs):
        self.history.append((args, kwargs))
        return self.fn(*args, **kwargs)


def test_pipeline():
    f1 = FunctionSpy(lambda x: x.lower())
    f2 = FunctionSpy(lambda x: x + " bar")
    f3 = FunctionSpy(lambda x: x.split(" "))

    pipeline = TextPipeline([f1, f2, f3])

    result = pipeline("Sample Text")
    assert result == ["sample", "text", "bar"]
    assert f1.history[0] == (("Sample Text",), {})
    assert f2.history[0] == (("sample text",), {})
    assert f3.history[0] == (("sample text bar",), {})


def test_pipeline_batch():
    f1 = FunctionSpy(lambda x: x.lower())
    f2 = FunctionSpy(lambda x: x + " bar")
    f3 = FunctionSpy(lambda x: x.split(" "))

    pipeline = TextPipeline([f1, f2, f3])
    inputs = ["Sample 1", "Sample 2", "Sample 3"]
    results = pipeline.batch(inputs)
    assert results == [
        ["sample", "1", "bar"],
        ["sample", "2", "bar"],
        ["sample", "3", "bar"],
    ]
    assert len(f1.history) == 3
    assert len(f2.history) == 3
    assert len(f3.history) == 3


def test_pipeline_remove_words():
    def reverse_text(x: str) -> str:
        return x[::-1]

    def split(x: str) -> list[str]:
        return x.split(" ")

    pipeline = TextPipeline(
        [reverse_text, split], words_to_remove=["foo", "bar", "ding bat"]
    )
    assert pipeline("alice foo bar bob ding bat") == ["bob", "ecila"]


def test_lowercase():
    assert lowercase("THIS ISN'T A TEst") == "this isn't a test"


def test_remove_punctuation():
    assert remove_punctuation("I'm an 123 example?—") == "Im an 123 example"


def test_remove_characters():
    assert (
        remove_characters("I'm a different—example?", "'—") == "Im a differentexample?"
    )


def test_subtitute_characters():
    assert (
        substitute_characters("This—is a sentence?", "—?", " !")
        == "This is a sentence!"
    )


def test_split_on_whitespace():
    assert split_on_whitespace("This    is    a test?ok") == [
        "This",
        "is",
        "a",
        "test?ok",
    ]
