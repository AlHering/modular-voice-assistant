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
DUMP_PATH = os.path.join(DATA_PATH, "dumps")


"""
Further data
"""
MODEL_PATH = "/mnt/Workspaces/Resources/machine_learning" #os.path.join(PACKAGE_PATH, "models")
CONFIG_PATH = os.path.join(PACKAGE_PATH, "configs")
BACKEND_PATH = os.path.join(DATA_PATH, "backend")
FRONTEND_PATH = os.path.join(DATA_PATH, "frontend")
FRONTEND_CACHE = os.path.join(FRONTEND_PATH, "cache.json")
