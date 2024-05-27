# -*- coding: utf-8 -*-
"""
****************************************************
*          Basic Language Model Backend            *
*            (c) 2023 Alexander Hering             *
****************************************************
"""
from src.configuration import configuration as cfg
from src.interfaces import text_generation_interface


if __name__ == "__main__":
    text_generation_interface.run_backend(cfg.BACKEND_HOST, cfg.BACKEND_PORT)
