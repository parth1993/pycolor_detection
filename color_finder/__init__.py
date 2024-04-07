from .color_finder import ColorFinder
from .background_remover import SimpleRemovalStrategy, U2NETRemovalStrategy, ImageProcessor

# Optionally, you could also include the ColorDictionary if you want it to be part of the public API
from .color_finder import ColorDictionary

__all__ = [
    "ColorFinder",
    "SimpleRemovalStrategy",
    "U2NETRemovalStrategy",
    "ImageProcessor",
    "ColorDictionary",
]
