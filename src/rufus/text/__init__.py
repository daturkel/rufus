import re
from typing import Any, Callable


class TextPipeline:
    def __init__(self, steps: list[Callable], words_to_remove: list[str] | None = None):
        """A pipeline that runs functions in sequence on an input string. Optionally, a pipeline can
        take a list of words and remove them from the output.

        Args:
            steps: A list of functions that will be run on the input of the pipeline. One step's output is the next step's input.
            words_to_remove: Words that will be removed from the output. These words will be passed through the pipeline and
                the output will be removed from the pipeline's output. Assumes that the last step of the pipeline outputs a list of strings.
                Defaults to None.
        """
        self.steps = steps
        self.words_to_remove = words_to_remove if words_to_remove is not None else []

        # inspired by https://stackoverflow.com/a/2158522
        def flatten(nested: list) -> list:
            if isinstance(nested, str):
                return [nested]
            else:
                return [innermost for inner in nested for innermost in flatten(inner)]

        self.tokens_to_remove = set(
            flatten(
                [self._call_without_word_removal(word) for word in self.words_to_remove]
            )
        )

    def _call_without_word_removal(self, text: str) -> str | list[str]:
        result = self.steps[0](text)
        for step in self.steps[1:]:
            result = step(result)
        return result

    def __call__(self, text: str) -> str | list[str]:
        """Run the pipeline on an input text.

        Args:
            text: A single string of input text.

        Returns:
            The text after it has been passed through all steps and optionally had `words_to_remove` removed.
        """
        result = self._call_without_word_removal(text)

        if self.tokens_to_remove:
            result = [token for token in result if token not in self.tokens_to_remove]

        return result

    def batch(self, iterable) -> Any:
        """Run the pipeline on an iterable of text inputs and return a list of the outputs.

        Args:
            iterable: An iterable of input texts.

        Returns:
            A list of pipeline outputs.
        """
        return [self(item) for item in iterable]


def lowercase(text: str) -> str:
    """Convert text to lowercase.

    Args:
        text: A single string of input text.

    Returns:
        The input string in lowercase.
    """
    return text.lower()


def remove_punctuation(text: str) -> str:
    """Remove punctuation from text using a regular expression.

    Args:
        text: A single string of input text.

    Returns:
        The input string after removing all characters non-word characters and non-whitespace characters.
    """
    pattern = re.compile(r"[^\w\s]")
    return pattern.sub("", text)


def remove_characters(text: str, characters: str) -> str:
    """Remove a set of characters from text.

    Args:
        text: A single string of input text.
        characters: A string containing all of the characters to be removed, e.g. `".,;"`.

    Returns:
        The input string after removing all instances of the characters in `characters `.
    """
    table = str.maketrans(dict.fromkeys(characters))
    return text.translate(table)


def substitute_characters(text: str, characters_in: str, characters_out: str) -> str:
    """Substitute characters from the input text with a different set of characters.

    Args:
        text: A single string of input text.
        characters_in: A string containing the characters in the input string to be replaced by other characters.
        characters_out: A string containing characters that will replace the characters in `characters_in`.

    Returns:
        The input string where each character in `characters_in` has been replaced with the corresponding character
        in `characters_out`.
    """
    table = str.maketrans(characters_in, characters_out)
    return text.translate(table)


def split_on_whitespace(text: str) -> list[str]:
    """Split the input text into a list of strings that were originally separated by whitespace.

    Args:
        text: A single string of input text.

    Returns:
        A list of strings from the input text that were previously separated by whitespace.
    """
    pattern = re.compile(r"\s+")
    return pattern.split(text)
