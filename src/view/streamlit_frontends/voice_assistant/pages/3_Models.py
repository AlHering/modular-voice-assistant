# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2024 Alexander Hering             *
****************************************************
"""
import os
import streamlit as st
from typing import List, Any, Optional
from requests.exceptions import ConnectionError
import json
import pandas as pd
from time import sleep
from src.utility.silver.file_system_utility import safely_create_path
from src.utility.bronze.streamlit_utility import render_json_input
from src.configuration import configuration as cfg
from src.view.streamlit_frontends.voice_assistant.utility.state_cache_handling import wait_for_setup, clear_tab_config
from src.view.streamlit_frontends.voice_assistant.utility.frontend_rendering import render_sidebar
from src.view.streamlit_frontends.voice_assistant.utility import backend_interaction


###################
# Main page functionality
###################
DATAFRAMES = {
    "transcriber": pd.read_csv(os.path.join(cfg.PATHS.DATA_PATH, "frontend", "transcriber_models.csv"))
}



def render_selection_dataframe(key: str, dataframe: pd.DataFrame) -> Any:
    """
    Function for rendering selection dataframe:
    :param dataframe: Dataframe.
    :return: Selection.
    """
    event = st.dataframe(
        dataframe,
        key=key,
        on_select="rerun",
        column_config={"Info": st.column_config.LinkColumn()},
        use_container_width=True
    )
    return event.selection


def download_model(backend: str, model_id: str, target_folder: str) -> None:
    """
    Function for downloading model.
    :param backend: Backend.
    :param model_id: Model ID.
    :param target_folder: Target folder.
    """
    print(backend)
    print(model_id)
    print(target_folder)
    

def render_model_page(object_type: str) -> None:
    """
    Function for rendering model page.
    :param object_type: Object type.
    """
    st.header("Defaults")
    
    if object_type in DATAFRAMES:
        selection = render_selection_dataframe(
            key=f"{object_type}_model_select",
            dataframe=DATAFRAMES[object_type]).get("rows")

        DATAFRAMES["transcriber"].to_csv(os.path.join(cfg.PATHS.DATA_PATH, "frontend", "transcriber_models.csv"), index=False)

        download_path_col, download_button_col, _ = st.columns([.8, .1, .1])
        download_path = download_path_col.text_input(
            key=f"{object_type}_model_download_path", 
            label="Download Folder")
        download_button_col.write("")
        if download_button_col.button(
            "Download selected"
        ):
            for row_index in selection:
                backend = DATAFRAMES[object_type]["Backend"].iloc[row_index]
                model_id = DATAFRAMES[object_type]["Model"].iloc[row_index]
                download_model(backend=backend,model_id=model_id, target_folder=download_path)




###################
# Entrypoint
###################


if __name__ == "__main__":
    # Basic metadata
    st.set_page_config(
        page_title="Voice Assistant",
        page_icon=":ocean:",
        layout="wide"
    )

    # Wait for backend and dependencies
    wait_for_setup()
        
    # Page content
    st.title("Models")
    
    tabs = ["transcriber", "synthesizer", "chat"]
    for index, tab in enumerate(st.tabs([" ".join(elem.split("_")).title()+"s" for elem in tabs])):
        with tab:
            render_model_page(tabs[index])
            
    render_sidebar()