from .base_processor import BaseProcessor
from utils.registry import PROCESSOR_FACTORY

from .processor import Processor
from .processor_clip import  PromptLearnProcessor
from .processor_clip_caption import PromptLearnCaptionProcessor

__all__ = [
    "BaseProcessor",
    "Processor",
    "PromptLearnProcessor",
    "PromptLearnCaptionProcessor"
]
