from .background_remover import (
    ImageProcessor,
    SimpleRemovalStrategy,
    U2NETRemovalStrategy,
)

# Optionally, you could also include the ColorDictionary if you want it to be part of the public API
from .color_finder import ColorDictionary, ColorFinder

__all__ = [
    "ColorFinder",
    "SimpleRemovalStrategy",
    "U2NETRemovalStrategy",
    "ImageProcessor",
    "ColorDictionary",
]
