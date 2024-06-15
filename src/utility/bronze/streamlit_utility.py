# -*- coding: utf-8 -*-
"""
****************************************************
*                     Utility                      *
*            (c) 2023 Alexander Hering             *
****************************************************
"""
from typing import Any, List
from time import sleep
import streamlit as st
import json
try:
    from code_editor import code_editor
except ImportError:
    pass
from datetime import datetime as dt


def wait_for_state_variable(streamlit_context: Any, variable_name: str, waiting_message: str, timeout: float = -1.0) -> None:
    """
    Function for waiting for state variable to be available.
    :param streamlit_context: Streamlit context.
    :param variable_name: Variable name of state variable to wait for.
    :param waiting_message: Waiting message to display with a spinner while waiting.
    :param timeout: Time to wait in seconds before raising a timeout error.
        Defaults to -1 in which case no timeout error is raised.
    :raises: TimeoutError if timeout is larger than -1 and is exceeded.
    """
    with streamlit_context.spinner(waiting_message):
        start = dt.now()
        while variable_name not in streamlit_context.session_state:
            sleep(0.1)
            if timeout >= 0 and dt.now() - start >= timeout:
                raise TimeoutError(
                    f"Timeout while waiting for '{variable_name}' to be available under Streamlit context.")


def update_bound_state_dictionary(field: str, update: dict) -> None:
    """
    Function for updating a bound state managed dictionary.
    :param field: State field of dictionary.
    :param update: Streamlit update dictionary.
    """
    for entry in update["added_rows"]:
        if len(entry.items()) > 1:
            st.session_state[field].loc[entry["_index"],
                                        ["value"]] = entry["value"]
            update["added_rows"].remove(entry)

    done = []
    for entry_key in update["edited_rows"]:
        if len(update["edited_rows"][entry_key].items()) > 1:
            entry = update["edited_rows"][entry_key]
            st.session_state[field].loc[entry["_index"],
                                        ["value"]] = entry["value"]
            done.append(entry_key)
    for to_remove in done:
        update["edited_rows"].pop(to_remove)
    for entry_index in update["deleted_rows"]:
        st.session_state[field].drop(
            st.session_state[field].index[entry_index], inplace=True)
    update["deleted_rows"] = []


def get_json_editor_buttons() -> List[dict]:
    """
    Function for acquiring json payload code editor buttons.
    Commands can be found at https://github.com/ajaxorg/ace/blob/v1.2.6/lib/ace/commands/default_commands.js.
    :return: Buttons as list of dictionaries.
    """
    return [
        {
            "name": "save",
            "feather": "Save",
            "hasText": True,
            "alwaysOn": True,
            "commands": [
                    "save-state",
                    [
                        "response",
                        "saved"
                    ]
            ],
            "response": "saved",
            "style": {"top": "0rem", "right": "9.6rem"}
        },
        {
            "name": "copy",
            "feather": "Copy",
            "hasText": True,
            "alwaysOn": True,
            "commands": ["copyAll"],
            "style": {"top": "0rem", "right": "5rem"}
        },
        {
            "name": "clear",
            "feather": "X",
            "hasText": True,
            "alwaysOn": True,
            "commands": ["selectall", "del", ["insertstring", "{\n\t\n}"], "save-state",
                         ["response", "saved"]],
            "style": {"top": "0rem", "right": "0.4rem"}
        },
    ]


def render_json_input(parent_widget: Any, key: str, label: str = None, default_data: dict = None) -> None:
    """
    Function for rendering JSON input.
    :param parent_widget: Parent widget.
    :param key: Widget key.
    :param label: Optional label.
    :param default_data: Default data.
    """
    if label is not None:
        parent_widget.write(label)
    with parent_widget.empty():
        widget = content = st.session_state["CACHE"].get(key)
        if widget is not None:
            content = widget["text"]
        else:
            content = json.dumps(
                default_data, 
                indent=4, 
                ensure_ascii=False
            )
        content = "{\n\t\n}" if content == "{}" else content
        code_editor(
            content,
            key=key,
            lang="json",
            allow_reset=True,
            options={"wrap": True},
            buttons=get_json_editor_buttons(),
            response_mode="debounce"
        )

        