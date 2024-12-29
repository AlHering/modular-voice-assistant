# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2024 Alexander Hering             *
****************************************************
"""
from enum import Enum
from typing import List, Generator
import json
import requests
from src.interface import VoiceAssistantInterface
from src.configuration import configuration as cfg


API_BASE = f"http://{cfg.BACKEND_HOST}:{cfg.BACKEND_PORT}{cfg.BACKEND_ENDPOINT_BASE}"
class Endpoints(str, Enum):
    """
    Endpoints config.
    """
    check_connection = API_BASE + "/check"
    add_configs = API_BASE + "/configs/add"
    patch_configs = API_BASE + "/configs/patch"
    get_configs = API_BASE + "/configs/get"
    load_modules = API_BASE + "/modules/load"
    unload_modules = API_BASE + "/modules/unload"
    setup_assistant = API_BASE + "/assistant/setup"
    reset_assistant = API_BASE + "/assistant/reset"
    stop_assistant = API_BASE + "/assistant/stop"
    inject_prompt = API_BASE + "/assistant/inject-prompt"
    interaction = API_BASE + "/assistant/interaction"
    conversation = API_BASE + "/assistant/conversation"
    terminal_conversation = API_BASE + "/assistant/terminal-conversation"
    transcribe = API_BASE + "/services/transcribe"
    synthesize = API_BASE + "/services/synthesize"
    chat = API_BASE + "/services/chat"
    chat_stream = API_BASE + "/services/chat-stream"

    def __str__(self) -> str:
        """
        Returns string representation.
        """
        return str(self.value)


class RemoteVoiceAssistantClient(object):
    """
    Remote voice assistant client.
    """
    def __init__(self) -> None:
        """
        Initiation method.
        :param working_directory: Working directory.
        """
        pass

    def check_connection(self) -> bool:
        """
        Checks connection to backend.
        :return: True, if available, else False.
        """
        try:
            resp = requests.get(Endpoints.check_connection).json()
            return True
        except Exception as ex:
            return False

    """
    Config handling
    """

    def add_config(self,
                   module_type: str,
                   config: dict) -> dict | None:
        """
        Adds a config to the database.
        :param module_type: Target module type.
        :param config: Config.
        :return: Response.
        """
        return requests.post(Endpoints.add_configs, json={
            "module_type": module_type,
            "config": config
        }).json().get("result")

    def overwrite_config(self,
                   module_type: str,
                   config: dict) -> dict:
        """
        Overwrites a config in the database.
        :param module_type: Target module type.
        :param config: Config.
        :return: Response.
        """
        return requests.post(Endpoints.patch_configs, json={
            "module_type": module_type,
            "config": config
        }).json().get("result")
    
    def get_configs(self,
                    module_type: str = None) -> List[dict] | None:
        """
        Adds a config to the database.
        :param module_type: Target module type.
            Defaults to None in which case all configs are returned.
        :return: Response.
        """
        return requests.post(
            Endpoints.get_configs, json={"module_type": module_type}).json().get("result")

    """
    Module handling
    """

    def load_module(self,
                    module_type: str,
                    config_uuid: str) -> dict:
        """
        Loads a module from the given config UUID.
        :param module_type: Target module type.
        :param config_uuid: Config UUID.
        :return: Response.
        """
        try:
            return requests.post(Endpoints.load_modules, json={
                "module_type": module_type,
                "config_uuid": config_uuid
            }).json()
        except:
            pass
            
    def unload_module(self,
                      module_type: str,
                      config_uuid: str) -> dict:
        """
        Unloads a module from the given config UUID.
        :param module_type: Target module type.
        :param config_uuid: Config UUID.
        :return: Response.
        """
        try:
            return requests.post(Endpoints.unload_modules, json={
                "module_type": module_type,
                "config_uuid": config_uuid
            }).json()
        except:
            pass

    """
    Assistant handling
    """

    def setup_assistant(self,
                        speech_recorder_uuid: str,
                        transcriber_uuid: str,
                        worker_uuid: str,
                        synthesizer_uuid: str,
                        wave_output_uuid: str,
                        stream: bool = True,
                        forward_logging: bool = False,
                        report: bool = False) -> dict:
        """
        Sets up a voice assistant from currently configured modules.
        :param speech_recorder_uuid: Speech Recorder config UUID.
        :param transcriber_uuid: Transcriber  config UUID.
        :param worker_uuid: Worker config UUID, e.g. for LocalChatModule or RemoteChatModule.
        :param synthesizer_uuid: Synthesizer config UUID.
        :param wave_output_uuid: Wave output config UUID.
        :param stream: Declares, whether chat model should stream its response.
        :param forward_logging: Flag for forwarding logger to modules.
        :param report: Flag for running report thread.
        """
        try:
            return requests.post(Endpoints.setup_assistant, json={
                "speech_recorder_uuid": speech_recorder_uuid,
                "transcriber_uuid": transcriber_uuid,
                "worker_uuid": worker_uuid,
                "synthesizer_uuid": synthesizer_uuid,
                "wave_output_uuid": wave_output_uuid,
                "stream": stream,
                "forward_logging": forward_logging,
                "report": report
            }).json()
        except:
            pass

    """
    Direct module access
    """

    def transcribe(self, 
                   audio_input: List[int | float] | str, 
                   dtype: str | None = None, 
                   transcription_parameters: dict | None = None) -> dict:
        """
        Transcribes audio to text.
        :param audio_input: Audio data or wave file path.
        :param dtype: Dtype in case of audio data input.
        :param transcription_parameters: Transcription parameters as dictionary.
            Defaults to None.
        :return: Transcript and metadata if successful, else error report.
        """
        try:
            return requests.post(Endpoints.transcribe, json={
                "audio_input": audio_input,
                "dtype": dtype,
                "transcription_parameters": transcription_parameters,
            }).json()
        except:
            pass

    def synthesize(self, text: str, synthesis_parameters: dict | None = None) -> dict:
        """
        Synthesizes audio from input text.
        :param text: Text to synthesize to audio.
        :param synthesis_parameters: Synthesis parameters as dictionary.
            Defaults to None.
        :return: Audio data, dtype and metadata if successful, else error report.
        """
        try:
            return requests.post(Endpoints.synthesize, json={
                "text": text,
                "synthesis_parameters": synthesis_parameters,
            }).json()
        except:
            pass
        
    def chat(self, 
             prompt: str, 
             chat_parameters: dict | None = None,
             local: bool = True) -> dict:
        """
        Generates a chat response.
        :param prompt: User input.
        :param chat_parameters: Kwargs for chatting in the chatting process as dictionary.
            Defaults to None in which case an empty dictionary is created.
        :return: Generated response and metadata if successful, else error report.
        """
        try:
            return requests.post(Endpoints.chat, json={
                "prompt": prompt,
                "chat_parameters": chat_parameters,
                "local": local,
            }).json()
        except:
            pass
        
    def chat_stream(self, 
             prompt: str, 
             chat_parameters: dict | None = None,
             local: bool = True) -> Generator[dict, None, None] | None:
        """
        Generates a streamed chat response.
        :param prompt: User input.
        :param chat_parameters: Kwargs for chatting in the chatting process as dictionary.
            Defaults to None in which case an empty dictionary is created.
        :return: Generated response and metadata if successful, else error report.
        """
        try:
            with requests.stream("POST", Endpoints.chat_stream, json={
                "prompt": prompt,
                "chat_parameters": chat_parameters,
                "local": local
            }) as response:
                for chunk in response.iter_lines():
                    if not chunk:
                        break
                    yield json.loads(chunk.decode("utf-8"))
        except:
            pass

class LocalVoiceAssistantClient(object):
    """
    Local voice assistant client.
    """
    def __init__(self) -> None:
        """
        Initiation method.
        """
        self.interface = VoiceAssistantInterface()

    def check_connection(self) -> bool:
        """
        Checks connection to backend.
        :return: True, if available, else False.
        """
        try:
            resp = self.interface.check_connection()
            return True
        except Exception as ex:
            return False

    """
    Config handling
    """

    def add_config(self,
                   module_type: str,
                   config: dict) -> dict | None:
        """
        Adds a config to the database.
        :param module_type: Target module type.
        :param config: Config.
        :return: Response.
        """
        return self.interface.add_config(payload={
            "module_type": module_type,
            "config": config
        }).get("result")

    def overwrite_config(self,
                   module_type: str,
                   config: dict) -> dict:
        """
        Overwrites a config in the database.
        :param module_type: Target module type.
        :param config: Config.
        :return: Response.
        """
        return self.interface.overwrite_config(payload={
            "module_type": module_type,
            "config": config
        }).get("result")
    
    def get_configs(self,
                    module_type: str = None) -> List[dict] | None:
        """
        Adds a config to the database.
        :param module_type: Target module type.
            Defaults to None in which case all configs are returned.
        :return: Response.
        """
        return self.interface.get_configs(payload={"module_type": module_type}).get("result")

    """
    Module handling
    """

    def load_module(self,
                    module_type: str,
                    config_uuid: str) -> dict:
        """
        Loads a module from the given config UUID.
        :param module_type: Target module type.
        :param config_uuid: Config UUID.
        :return: Response.
        """
        return self.interface.load_module(payload={
            "module_type": module_type,
            "config_uuid": config_uuid
        })
            
    def unload_module(self,
                      module_type: str,
                      config_uuid: str) -> dict:
        """
        Unloads a module from the given config UUID.
        :param module_type: Target module type.
        :param config_uuid: Config UUID.
        :return: Response.
        """
        self.interface.unload_module(payload={
            "module_type": module_type,
            "config_uuid": config_uuid
        })

    """
    Assistant handling
    """

    def setup_assistant(self,
                        speech_recorder_uuid: str,
                        transcriber_uuid: str,
                        worker_uuid: str,
                        synthesizer_uuid: str,
                        wave_output_uuid: str,
                        stream: bool = True,
                        forward_logging: bool = False,
                        report: bool = False) -> dict:
        """
        Sets up a voice assistant from currently configured modules.
        :param speech_recorder_uuid: Speech Recorder config UUID.
        :param transcriber_uuid: Transcriber  config UUID.
        :param worker_uuid: Worker config UUID, e.g. for LocalChatModule or RemoteChatModule.
        :param synthesizer_uuid: Synthesizer config UUID.
        :param wave_output_uuid: Wave output config UUID.
        :param stream: Declares, whether chat model should stream its response.
        :param forward_logging: Flag for forwarding logger to modules.
        :param report: Flag for running report thread.
        """
        return self.interface.setup_assistant(payload={
            "speech_recorder_uuid": speech_recorder_uuid,
            "transcriber_uuid": transcriber_uuid,
            "worker_uuid": worker_uuid,
            "synthesizer_uuid": synthesizer_uuid,
            "wave_output_uuid": wave_output_uuid,
            "stream": stream,
            "forward_logging": forward_logging,
            "report": report
        })

    """
    Direct module access
    """

    def transcribe(self, 
                   audio_input: List[int | float] | str, 
                   dtype: str | None = None, 
                   transcription_parameters: dict | None = None) -> dict:
        """
        Transcribes audio to text.
        :param audio_input: Audio data or wave file path.
        :param dtype: Dtype in case of audio data input.
        :param transcription_parameters: Transcription parameters as dictionary.
            Defaults to None.
        :return: Transcript and metadata if successful, else error report.
        """
        return self.interface.transcribe(**{
            "audio_input": audio_input,
            "dtype": dtype,
            "transcription_parameters": transcription_parameters,
        })

    def synthesize(self, text: str, synthesis_parameters: dict | None = None) -> dict:
        """
        Synthesizes audio from input text.
        :param text: Text to synthesize to audio.
        :param synthesis_parameters: Synthesis parameters as dictionary.
            Defaults to None.
        :return: Audio data, dtype and metadata if successful, else error report.
        """
        return self.interface.synthesize(**{
            "text": text,
            "synthesis_parameters": synthesis_parameters,
        })
        
    def chat(self, 
             prompt: str, 
             chat_parameters: dict | None = None,
             local: bool = True) -> dict:
        """
        Generates a chat response.
        :param prompt: User input.
        :param chat_parameters: Kwargs for chatting in the chatting process as dictionary.
            Defaults to None in which case an empty dictionary is created.
        :return: Generated response and metadata if successful, else error report.
        """
        return self.interface.chat(**{
            "prompt": prompt,
            "chat_parameters": chat_parameters,
            "local": local,
        })
        
    def chat_stream(self, 
             prompt: str, 
             chat_parameters: dict | None = None,
             local: bool = True) -> Generator[dict, None, None] | None:
        """
        Generates a streamed chat response.
        :param prompt: User input.
        :param chat_parameters: Kwargs for chatting in the chatting process as dictionary.
            Defaults to None in which case an empty dictionary is created.
        :return: Generated response and metadata if successful, else error report.
        """
        for response in self.interface._wrapped_streamed_chat(**{
            "prompt": prompt,
            "chat_parameters": chat_parameters,
            "local": local
        }):
            yield response
        