# -*- coding: utf-8 -*-
"""
****************************************************
*          Basic Language Model Backend            *
*            (c) 2023 Alexander Hering             *
****************************************************
"""
from src.configuration import configuration as cfg
from src.interfaces import backend_interface


if __name__ == "__main__":
    backend_interface.run_backend(cfg.BACKEND_HOST, cfg.BACKEND_PORT)
