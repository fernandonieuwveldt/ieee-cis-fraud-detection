"""
Fraud detector info file
"""

from os import path
from src import classifier
from src import detector
from src import embedding_classifier
from src import feature_extractor
from src import pre_process
from src import transformers

__all__ = ['classifier', 'detector', 'embedding_classifier', 'feature_extractor', 'pre_process', 'transformers']

INPUT_DIR = path.join(path.dirname(path.realpath(__file__)), '../data/input_data/ieee-fraud-detection/')
OUTPUT_DIR = path.join(path.dirname(path.realpath(__file__)), '../data/output_data/')

