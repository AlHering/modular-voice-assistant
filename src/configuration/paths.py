# -*- coding: utf-8 -*-
"""
****************************************************
*          Basic Language Model Backend            *
*            (c) 2023 Alexander Hering             *
****************************************************
"""
import os


"""
Base 
"""
PACKAGE_PATH = os.path.abspath(os.path.dirname(os.path.dirname(
    os.path.dirname(__file__))))
SOURCE_PATH = os.path.join(PACKAGE_PATH, "src")
DOCS_PATH = os.path.join(PACKAGE_PATH, "docs")
DATA_PATH = os.path.join(PACKAGE_PATH, "data")
FILE_PATH = os.path.join(DATA_PATH, "files")
TEST_PATH = os.path.join(DATA_PATH, "testing")
PLUGIN_PATH = os.path.join(SOURCE_PATH, "plugins")
DUMP_PATH = os.path.join(DATA_PATH, "dumps")


"""
Machine Learning Models
"""
MODEL_PATH = os.path.join(
    PACKAGE_PATH, "machine_learning_models")
TEXT_GENERATION_MODEL_PATH = os.path.join(
    MODEL_PATH, "text_generation")
IMAGE_GENERATION_MODEL_PATH = os.path.join(
    MODEL_PATH, "image_generation")
SOUND_GENERATION_MODEL_PATH = os.path.join(
    MODEL_PATH, "sound_generation")


"""
Backends
"""
BACKEND_PATH = os.path.join(DATA_PATH, "backend")


"""
Frontends
"""
FRONTEND_PATH = os.path.join(DATA_PATH, "frontend")
FRONTEND_ASSETS = os.path.join(
    FRONTEND_PATH, "assets")
FRONTEND_CACHE= os.path.join(
    FRONTEND_PATH, "cache.json")
FRONTEND__DEFAULT_CACHE= os.path.join(
    FRONTEND_PATH, "default_cache.json")
