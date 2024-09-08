from abc import abstractmethod
from typing import Protocol, Self

from PIL import Image


class StimulusSet(Protocol):
    identifier: str

    @abstractmethod
    def __getitem__(self: Self, stimulus: int | str) -> Image.Image:
        pass

    @abstractmethod
    def __len__(self: Self) -> int:
        pass
