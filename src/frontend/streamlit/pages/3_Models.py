# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2024 Alexander Hering             *
****************************************************
"""
import os
import streamlit as st
from typing import Any
import traceback
import pandas as pd
from src.configuration import configuration as cfg
from src.frontend.streamlit.utility.frontend_rendering import render_sidebar
from src.utility.whisper_utility import download_whisper_model
from src.utility.faster_whisper_utility import download_faster_whisper_model
from src.utility.coqui_tts_utility import download_coqui_tts_model


###################
# Main page functionality
###################
DATAFRAMES = {
    "transcriber": pd.DataFrame(
        columns=["Model", "Backend", "Info", "Size (downloaded)", "Path (downloaded)"],
        data=[
            ["", "coqui-tts", "", "", ""],
            ["", "coqui-tts", "", "", ""],
            ["", "coqui-tts", "", "", ""],
            ["", "coqui-tts", "", "", ""],
            ["", "coqui-tts", "", "", ""],
            ["", "coqui-tts", "", "", ""],
            ["", "coqui-tts", "", "", ""],
            ["", "coqui-tts", "", "", ""],
            ["", "coqui-tts", "", "", ""],
            ["", "coqui-tts", "", "", ""],
        ]
    )
}



def render_selection_dataframe(key: str, dataframe: pd.DataFrame) -> Any:
    """
    Renders selection dataframe:
    :param dataframe: Dataframe.
    :return: Selection.
    """
    event = st.dataframe(
        dataframe,
        key=key,
        on_select="rerun",
        column_config={"Info": st.column_config.LinkColumn()},
        column_order=["Model", "Backend", "Info", "Size (downloaded)", "Path (downloaded)"],
        use_container_width=True
    )
    return event.selection


def check_downloaded_models(dataframe: pd.DataFrame) -> None:
    """
    Checks and adjusts downloaded model data.
    :param dataframe: Dataframe.
    """
    pass

def download_model(backend: str, model_id: str, target_folder: str) -> bool:
    """
    Downloads model.
    :param backend: Backend.
    :param model_id: Model ID.
    :param target_folder: Target folder.
    :return: True if process was successful.
    """
    try:
        with st.spinner(f"Downloading {backend}-{model_id} to '{target_folder}'..."):
            {
                "whisper": download_whisper_model,
                "faster-whisper": download_faster_whisper_model,
                "coqui-tts": download_coqui_tts_model
            }[backend](model_id, target_folder)
        return True
    except Exception:
        traceback.print_exc()
        return False
    
def render_model_page(object_type: str) -> None:
    """
    Renders model page.
    :param object_type: Object type.
    """
    if f"{object_type}_model_download_flair" not in st.session_state:
        flairs = []
    else: 
        flairs = st.session_state[f"{object_type}_model_download_flair"]
    for flair in flairs:
        {
            "success": st.success,
            "warning": st.warning
        }[flair[0]](flair[1])
    st.session_state[f"{object_type}_model_download_flair"] = []

    st.header("Defaults")
   
    changes = False
    if object_type in DATAFRAMES:
        check_downloaded_models(DATAFRAMES[object_type])
        selection = render_selection_dataframe(
            key=f"{object_type}_model_select",
            dataframe=DATAFRAMES[object_type]).get("rows")

        download_path_col, download_button_col = st.columns([.76, .14])
        download_path = download_path_col.text_input(
            key=f"{object_type}_model_download_path", 
            label="Download Folder",
            value=st.session_state.get(f"{object_type}_model_download_path"))
        download_button_col.write("")
        if download_button_col.button(
            "Download selected ..."
        ):
            for row_index in selection:
                backend = DATAFRAMES[object_type]["Backend"].iloc[row_index]
                model_id = DATAFRAMES[object_type]["Model"].iloc[row_index]
                if download_model(backend=backend,model_id=model_id, target_folder=download_path):
                    changes = True
                    st.session_state[f"{object_type}_model_download_flair"].append(("success", f"Downloading {backend}-{model_id} to '{download_path}' successful!"))
                    if DATAFRAMES[object_type]["Path (downloaded)"].iloc[row_index] != download_path:
                        DATAFRAMES[object_type]["Path (downloaded)"].iloc[row_index] = download_path
                        DATAFRAMES[object_type].to_csv(os.path.join(cfg.PATHS.DATA_PATH, "frontend", f"{object_type}_models.csv"), index=False)
                else:
                    st.session_state[f"{object_type}_model_download_flair"].append(("warning", f"Downloading {backend}-{model_id} to '{download_path}' failed!"))
    if changes:
        st.session_state.pop("transcriber_model_select")
        st.rerun()


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

    # Page content
    st.title("Models")
    st.error("Currently not implemented!")
    
    # Wait for backend and dependencies
    if "SETUP" not in st.session_state or not st.session_state["SETUP"]:
        st.info("System inactive. Please enter a correct backend server API in the sidebar (Local example: 'http://127.0.0.1:7861/api/v1').")
    else:
        tabs = ["transcriber", "synthesizer", "chat"]
        for index, tab in enumerate(st.tabs([" ".join(elem.split("_")).title()+"s" for elem in tabs])):
            with tab:
                render_model_page(tabs[index])
            
    render_sidebar()