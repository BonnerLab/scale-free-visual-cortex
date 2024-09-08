from abc import abstractmethod
from typing import Protocol, Self

import pandas as pd
from PIL import Image


class StimulusSet(Protocol):
    identifier: str
    metadata: pd.DataFrame

    @abstractmethod
    def __getitem__(self: Self, stimulus: int | str) -> Image.Image:
        pass
