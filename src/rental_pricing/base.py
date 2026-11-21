"""The shared interface of every pipeline stage."""

from abc import ABC, abstractmethod


class Stage(ABC):
    """One step of the pipeline (preprocess, features, train, promote, score, ...).

    A stage receives everything it needs in ``__init__`` and does its work in ``run()``.
    The CLI scripts in ``scripts/`` only parse arguments, build a stage and call ``run()``.
    """

    name: str = "stage"

    @abstractmethod
    def run(self):
        """Execute the stage and return its main result."""
