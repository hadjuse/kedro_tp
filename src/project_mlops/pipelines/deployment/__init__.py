"""
This is a boilerplate pipeline 'deployment'
generated using Kedro 1.2.0
"""

from .pipeline import create_pipeline
from dotenv import load_dotenv

__all__ = ["create_pipeline"]

__version__ = "0.1"
load_dotenv()
