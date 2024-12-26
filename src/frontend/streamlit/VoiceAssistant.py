# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2024 Alexander Hering             *
****************************************************
"""
import streamlit as st

from requests.exceptions import ConnectionError
from time import sleep
from src.frontend.streamlit.utility.frontend_rendering import render_sidebar, render_pipeline_node_plane
from src.frontend.streamlit.utility.backend_interaction import get_components, SpeechRecorderModule, BasicHandlerModule
from src.frontend.streamlit.utility.state_cache_handling import wait_for_setup


###################
# Main page functionality
###################


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
    st.title("Voice Assistant")

    """module_set = BaseModuleSet()
    available = get_components()
    header_columns = st.columns([.25, .25, .25, .10, .10, .5])
    header_columns[0].selectbox(
        key=f"va_input_type", 
        label="Input Type", 
        options=["Text", "Speech"])
    header_columns[1].selectbox(
        key=f"va_speech_recorder", 
        label="Recorder", 
        options=available["speech_recorder"],
        disabled=st.session_state["va_input_type"] == "Speech")
    header_columns[2].selectbox(
        key=f"va_transcriber", 
        label="Transcriber", 
        options=available["transcriber"],
        disabled=st.session_state["va_input_type"] == "Speech")
    if header_columns[3].button("Load",
                            help="Loads the input components."):
        if st.session_state["va_input_type"] == "Speech":
            module_set.input_modules.append(
                SpeechRecorderModule(speech_recorder=, 
                                logger=forward_logging,
                                name="SpeechRecorder"))
        self.module_set.input_modules.append(
            BasicHandlerModule(handler_method=self.transcriber.transcribe, 
                              input_queue=self.module_set.input_modules[-1].output_queue, 
                              logger=forward_logging,
                              name="Transcriber"))
        self.module_set.worker_modules.append(
            BasicHandlerModule(handler_method=self.chat_model.chat_stream if stream else self.chat_model.chat,
                            input_queue=self.module_set.input_modules[-1].output_queue,
                            logger=forward_logging,
                            name="Chat")
        )
        self.module_set.output_modules.append(
            BasicHandlerModule(handler_method=clean_worker_output,
                               input_queue=self.module_set.worker_modules[-1].output_queue,
                               logger=forward_logging,
                               name="Cleaner")
        )
        self.module_set.output_modules.append(
            BasicHandlerModule(handler_method=self.synthesizer.synthesize,
                              input_queue=self.module_set.worker_modules[-1].output_queue if len(
                                  self.module_set.output_modules) == 0 else self.module_set.output_modules[-1].output_queue,
                              logger=forward_logging,
                              name="Synthesizer")
        )
        self.module_set.output_modules.append(
            WaveOutputModule(input_queue=self.module_set.output_modules[-1].output_queue, 
                             logger=forward_logging,
                             name="WaveOutput")
        )"""
        

    render_pipeline_node_plane(parent_widget=st.container(),
                               block_entries=get_components(),
                               session_state_key="pipeline_schema")
    
    render_sidebar()