from .base_processor import BaseProcessor
from utils.registry import PROCESSOR_FACTORY

from .processor import Processor
from .processor_clip import  PromptLearnProcessor

__all__ = [
    "BaseProcessor",
    "Processor",
    "PromptLearnProcessor"
]
