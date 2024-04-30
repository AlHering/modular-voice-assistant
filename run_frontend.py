# -*- coding: utf-8 -*-
"""
****************************************************
*          Basic Language Model Backend            *
*            (c) 2023 Alexander Hering             *
****************************************************
"""
from src.configuration import configuration as cfg, flask_frontend_config as flask_cfg
from src.utility.bronze import json_utility
from src.utility.silver.file_system_utility import safely_create_path
from src.control.flask_frontend_controller import FlaskFrontendController
from src.view.streamlit_frontends import llm_tutor_app as app


def run_streamlit() -> None:
    """
    Function for running streamlit frontend.
    """
    app.run_app_with_st_pages()


def run_flask() -> None:
    """
    Function for running flask frontend.
    """
    app = FlaskFrontendController(
        flask_cfg.global_config
    )
    app.run_app()


if __name__ == "__main__":
    run_flask()
