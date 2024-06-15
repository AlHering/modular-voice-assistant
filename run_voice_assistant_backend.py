# -*- coding: utf-8 -*-
"""
****************************************************
*          Basic Language Model Backend            *
*            (c) 2023 Alexander Hering             *
****************************************************
"""
from src.interfaces import voice_assistant_interface


if __name__ == "__main__":
    voice_assistant_interface.run_backend(reload=False)
