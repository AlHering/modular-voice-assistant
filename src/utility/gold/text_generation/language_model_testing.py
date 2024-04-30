# -*- coding: utf-8 -*-
"""
****************************************************
*          Basic Language Model Backend            *
*            (c) 2023 Alexander Hering             *
****************************************************
"""
import os
import traceback
from typing import List
from .language_model_abstractions import LanguageModelInstance
from src.configuration import configuration as cfg


"""
Evaluation and experimentation
"""


TESTING_CONFIGS = {
    #########################
    # llamacpp
    #########################
    "llamacpp_openhermes-2.5-mistral-7b_v2": {
        "instance_parameters": {
            "model_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                       "TheBloke_OpenHermes-2.5-Mistral-7B-GGUF"),
            "backend": "llamacpp",
            "model_file": "openhermes-2.5-mistral-7b.Q4_K_M.gguf",
            "model_parameters": {"n_ctx": 4096},
            "tokenizer_path": "/mnt/Workspaces/Resources/machine_learning_models/text_generation/MODELS/OpenHermes-2.5-Mistral-7B",
            "default_system_prompt": "You are 'Hermes 2', a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia."
        },
        "generation_parameters": {
            "prompt": "Your task is to create a Python script for scraping the first 10 google hits for a given search query. Explain your solution afterwards.",
            "history_merger": lambda history: "\n".join(f"<|im_start|>{entry[0]}\n{entry[1]}<|im_end|>" for entry in history) + "\n",
            "generating_parameters": {"max_tokens": 1024}
        }
    },
    "llamacpp_openhermes-2.5-mistral-7b": {
        "instance_parameters": {
            "model_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                       "TheBloke_OpenHermes-2.5-Mistral-7B-GGUF"),
            "backend": "llamacpp",
            "model_file": "openhermes-2.5-mistral-7b.Q4_K_M.gguf",
            "model_parameters": {"n_ctx": 4096},
            "tokenizer_path": "/mnt/Workspaces/Resources/machine_learning_models/text_generation/MODELS/OpenHermes-2.5-Mistral-7B",
            "default_system_prompt": "You are 'Hermes 2', a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia."
        },
        "generation_parameters": {
            "prompt": "Create a Python script for scraping the first 10 google hits for a search query.",
            "history_merger": lambda history: "\n".join(f"<|im_start|>{entry[0]}\n{entry[1]}<|im_end|>" for entry in history) + "\n",
            "generating_parameters": {"max_tokens": 1024}
        }
    },
    "llamacpp_openhermes-2.5-mistral-7b-16k_v2": {
        "instance_parameters": {
            "model_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                       "TheBloke_OpenHermes-2.5-Mistral-7B-16k-GGUF"),
            "backend": "llamacpp",
            "model_file": "openhermes-2.5-mistral-7b-16k.Q4_K_M.gguf",
            "model_parameters": {"n_ctx": 4096},
            "tokenizer_path": "/mnt/Workspaces/Resources/machine_learning_models/text_generation/MODELS/OpenHermes-2.5-Mistral-7B",
            "default_system_prompt": "You are 'Hermes 2', a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia."
        },
        "generation_parameters": {
            "prompt": "Your task is to create a Python script for scraping the first 10 google hits for a given search query. Explain your solution afterwards.",
            "history_merger": lambda history: "\n".join(f"<|im_start|>{entry[0]}\n{entry[1]}<|im_end|>" for entry in history) + "\n",
            "generating_parameters": {"max_tokens": 1024}
        }
    },
    "llamacpp_openhermes-2.5-mistral-7b-16k": {
        "instance_parameters": {
            "model_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                       "TheBloke_OpenHermes-2.5-Mistral-7B-16k-GGUF"),
            "backend": "llamacpp",
            "model_file": "openhermes-2.5-mistral-7b-16k.Q4_K_M.gguf",
            "model_parameters": {"n_ctx": 16384},
            "tokenizer_path": "/mnt/Workspaces/Resources/machine_learning_models/text_generation/MODELS/OpenHermes-2.5-Mistral-7B",
            "default_system_prompt": "You are 'Hermes 2', a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia."
        },
        "generation_parameters": {
            "prompt": "Create a Python script for scraping the first 10 google hits for a search query.",
            "history_merger": lambda history: "\n".join(f"<|im_start|>{entry[0]}\n{entry[1]}<|im_end|>" for entry in history) + "\n",
            "generating_parameters": {"max_tokens": 2048}
        }
    },
    #########################
    # ctransformers
    #########################
    "ctransformers_openhermes-2.5-mistral-7b": {
        "instance_parameters": {
            "model_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                       "TheBloke_OpenHermes-2.5-Mistral-7B-GGUF"),
            "backend": "ctransformers",
            "model_file": "openhermes-2.5-mistral-7b.Q4_K_M.gguf",
            "model_parameters": {"context_length": 4096, "max_new_tokens": 1024},
            "tokenizer_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                           "TheBloke_OpenHermes-2.5-Mistral-7B-GGUF"),
            "default_system_prompt": "You are 'Hermes 2', a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia."
        },
        "generation_parameters": {
            "prompt": "Create a Python script for scraping the first 10 google hits for a search query.",
            "history_merger": lambda history: "\n".join(f"<|im_start|>{entry[0]}\n{entry[1]}<|im_end|>" for entry in history) + "\n",
            "generating_parameters": {"max_new_tokens": 1024}
        }
    },
    "ctransformers_openhermes-2.5-mistral-7b-16k": {
        "instance_parameters": {
            "model_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                       "TheBloke_OpenHermes-2.5-Mistral-7B-16k-GGUF"),
            "backend": "ctransformers",
            "model_file": "openhermes-2.5-mistral-7b-16k.Q4_K_M.gguf",
            "model_parameters": {"context_length": 4096, "max_new_tokens": 2048},
            "tokenizer_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                           "TheBloke_OpenHermes-2.5-Mistral-7B-16k-GGUF"),
            "default_system_prompt": "You are 'Hermes 2', a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia."
        },
        "generation_parameters": {
            "prompt": "Create a Python script for scraping the first 10 google hits for a search query.",
            "history_merger": lambda history: "\n".join(f"<|im_start|>{entry[0]}\n{entry[1]}<|im_end|>" for entry in history) + "\n",
            "generating_parameters": {"max_new_tokens": 2048}
        }
    },
    #########################
    # langchain_llamacpp
    #########################
    "langchain_llamacpp_openhermes-2.5-mistral-7b": {
        "instance_parameters": {
            "model_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                       "TheBloke_OpenHermes-2.5-Mistral-7B-GGUF"),
            "backend": "langchain_llamacpp",
            "model_file": "openhermes-2.5-mistral-7b.Q4_K_M.gguf",
            "model_parameters": {"n_ctx": 4096},
            "tokenizer_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                           "TheBloke_OpenHermes-2.5-Mistral-7B-GGUF"),
            "default_system_prompt": "You are 'Hermes 2', a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia."
        },
        "generation_parameters": {
            "prompt": "Create a Python script for scraping the first 10 google hits for a search query.",
            "history_merger": lambda history: "\n".join(f"<|im_start|>{entry[0]}\n{entry[1]}<|im_end|>" for entry in history) + "\n",
            "generating_parameters": {"max_tokens": 1024}
        }
    },
    "langchain_llamacpp_openhermes-2.5-mistral-7b-16k": {
        "instance_parameters": {
            "model_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                       "TheBloke_OpenHermes-2.5-Mistral-7B-16k-GGUF"),
            "backend": "langchain_llamacpp",
            "model_file": "openhermes-2.5-mistral-7b-16k.Q4_K_M.gguf",
            "model_parameters": {"n_ctx": 4096},
            "tokenizer_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                           "TheBloke_OpenHermes-2.5-Mistral-7B-16k-GGUF"),
            "default_system_prompt": "You are 'Hermes 2', a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia."
        },
        "generation_parameters": {
            "prompt": "Create a Python script for scraping the first 10 google hits for a search query.",
            "history_merger": lambda history: "\n".join(f"<|im_start|>{entry[0]}\n{entry[1]}<|im_end|>" for entry in history) + "\n",
            "generating_parameters": {"max_tokens": 2048}
        }
    },
    #########################
    # autoqptq
    #########################
    "autogptq_openhermes-2.5-mistral-7b": {
        "instance_parameters": {
            "model_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                       "TheBloke_OpenHermes-2.5-Mistral-7B-GPTQ"),
            "backend": "autogptq",
            "model_file": "model.safetensors",
            "model_parameters": {"device": "cuda:0", "local_files_only": True},
            "tokenizer_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                           "TheBloke_OpenHermes-2.5-Mistral-7B-GPTQ"),
            "default_system_prompt": "You are 'Hermes 2', a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia."
        },
        "generation_parameters": {
            "prompt": "Create a Python script for scraping the first 10 google hits for a search query.",
            "history_merger": lambda history: "\n".join(f"<|im_start|>{entry[0]}\n{entry[1]}<|im_end|>" for entry in history) + "\n",
            "tokenizing_parameters": {"return_tensors": "pt"},
            "generating_parameters": {"max_new_tokens": 1024},
            "decoding_parameters": {"skip_special_tokens": True}
        }
    },
    "autogptq_openhermes-2.5-mistral-7b-16k": {
        "instance_parameters": {
            "model_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                       "TheBloke_OpenHermes-2.5-Mistral-7B-16k-GPTQ"),
            "backend": "autogptq",
            "model_file": "model.safetensors",
            "model_parameters": {"device": "cuda:0", "local_files_only": True},
            "tokenizer_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                           "TheBloke_OpenHermes-2.5-Mistral-7B-16k-GPTQ"),
            "default_system_prompt": "You are 'Hermes 2', a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia."
        },
        "generation_parameters": {
            "prompt": "Create a Python script for scraping the first 10 google hits for a search query.",
            "history_merger": lambda history: "\n".join(f"<|im_start|>{entry[0]}\n{entry[1]}<|im_end|>" for entry in history) + "\n",
            "tokenizing_parameters": {"return_tensors": "pt"},
            "generating_parameters": {"max_new_tokens": 2048},
            "decoding_parameters": {"skip_special_tokens": True}
        }
    },
    "autogptq_openchat_3.5": {
        "instance_parameters": {
            "model_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                       "TheBloke_openchat_3.5-GPTQ"),
            "backend": "autogptq",
            "model_file": "model.safetensors",
            "model_parameters": {"device": "cuda:0", "local_files_only": True},
            "tokenizer_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                           "TheBloke_openchat_3.5-GPTQ"),
            "default_system_prompt": "You are 'Hermes 2', a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia."
        },
        "generation_parameters": {
            "prompt": "Create a Python script for scraping the first 10 google hits for a search query.",
            "tokenizing_parameters": {"return_tensors": "pt"},
            "generating_parameters": {"max_new_tokens": 1024},
            "decoding_parameters": {"skip_special_tokens": True}
        }
    },
}

CURRENTLY_NOT_WORKING = {
    #########################
    # autoqptq
    #########################
    "autogptq_rocket-3b": {
        "instance_parameters": {
            "model_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                       "TheBloke_rocket-3B-GPTQ"),
            "backend": "autogptq",
            "model_file": "model.safetensors",
            "model_parameters": {"device_map": "auto", "use_triton": True, "local_files_only": True, "trust_remote_code": True},
            "tokenizer_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                           "TheBloke_rocket-3B-GPTQ"),
            "default_system_prompt": "You are 'Hermes 2', a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia."
        },
        "generation_parameters": {
            "prompt": "Create a Python script for scraping the first 10 google hits for a search query.",
            "history_merger": lambda history: "\n".join(f"<|im_start|>{entry[0]}\n{entry[1]}<|im_end|>" for entry in history) + "\n",
            "tokenizing_parameters": {"return_tensors": "pt"},
            "generation_parameters": {"max_new_tokens": 128},
            "decoding_parameters": {"skip_special_tokens": True}
        }
    },
    "autogptq_stablecode-instruct-alpha-3b": {
        "instance_parameters": {
            "model_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                       "TheBloke_stablecode-instruct-alpha-3b-GPTQ"),
            "backend": "autogptq",
            "model_file": "model.safetensors",
            "model_parameters": {"device": "cuda:0", "local_files_only": True},
            "tokenizer_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                           "TheBloke_stablecode-instruct-alpha-3b-GPTQ"),
            "default_system_prompt": "You are 'Hermes 2', a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia."
        },
        "generation_parameters": {
            "prompt": "Create a Python script for scraping the first 10 google hits for a search query.",
            "history_merger": lambda history: "\n".join(f"<s>{entry[0].replace('user', '###Instruction:').replace('assistant', '###Response:')}\n{entry[1]}<|im_end|>" for entry in history if entry[0] != "system") + "\n",
            "tokenizing_parameters": {"return_tensors": "pt"},
            "generation_parameters": {"max_new_tokens": 128},
            "decoding_parameters": {"skip_special_tokens": True, "return_token_type_ids": False}
        }
    },
    #########################
    # exllamav2
    #########################
    "exllamav2_openhermes-2.5-mistral-7b": {
        "instance_parameters": {
            "model_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                       "TheBloke_OpenHermes-2.5-Mistral-7B-GPTQ"),
            "backend": "exllamav2",
            "model_file": "model.safetensors",
            "model_parameters": {},
            "tokenizer_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                           "TheBloke_OpenHermes-2.5-Mistral-7B-GPTQ"),
            "default_system_prompt": "You are 'Hermes 2', a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia."
        },
        "generation_parameters": {
            "prompt": "Create a Python script for scraping the first 10 google hits for a search query.",
            "history_merger": lambda history: "\n".join(f"<|im_start|>{entry[0]}\n{entry[1]}<|im_end|>" for entry in history) + "\n",
            "generation_parameters": {"max_tokens": 1024}
        }
    },
    "exllamav2_openhermes-2.5-mistral-7b-16k": {
        "instance_parameters": {
            "model_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                       "TheBloke_OpenHermes-2.5-Mistral-7B-16k-GPTQ"),
            "backend": "exllamav2",
            "model_file": "model.safetensors",
            "model_parameters": {},
            "tokenizer_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                           "TheBloke_OpenHermes-2.5-Mistral-7B-16k-GPTQ"),
            "default_system_prompt": "You are 'Hermes 2', a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia."
        },
        "generation_parameters": {
            "prompt": "Create a Python script for scraping the first 10 google hits for a search query.",
            "history_merger": lambda history: "\n".join(f"<|im_start|>{entry[0]}\n{entry[1]}<|im_end|>" for entry in history) + "\n",
            "generation_parameters": {"max_tokens": 2048}
        }
    },
    "exllamav2_rocket-3b": {
        "instance_parameters": {
            "model_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                       "TheBloke_rocket-3B-GPTQ"),
            "backend": "exllamav2",
            "model_file": "model.safetensors",
            "model_parameters": {"device_map": "auto", "local_files_only": True, "trust_remote_code": True},
            "tokenizer_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                           "TheBloke_rocket-3B-GPTQ"),
            "default_system_prompt": "You are 'Hermes 2', a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia."
        },
        "generation_parameters": {
            "prompt": "Create a Python script for scraping the first 10 google hits for a search query.",
            "history_merger": lambda history: "\n".join(f"<|im_start|>{entry[0]}\n{entry[1]}<|im_end|>" for entry in history) + "\n",
            "generation_parameters": {"max_tokens": 128}
        }
    },
}


class Colors:
    std = "\x1b[0;37;40m"
    fat = "\x1b[1;37;40m"
    blk = "\x1b[6;30;42m"
    wrn = "\x1b[31m"
    end = "\x1b[0m"


def run_model_test(configs: List[str] = None) -> dict:
    """
    Function for running model tests based off of configurations.
    :param configs: List of names of configurations to run.
    :return: Answers.
    """
    if configs is None or not all(config in TESTING_CONFIGS for config in configs):
        configs = list(TESTING_CONFIGS.keys())
    answers = {}

    for config in configs:
        try:
            coder = LanguageModelInstance(
                **TESTING_CONFIGS[config]["instance_parameters"]
            )

            answer, metadata = coder.generate(
                **TESTING_CONFIGS[config]["generation_parameters"]
            )
            answers[config] = ("success", answer, metadata)
        except Exception as ex:
            answers[config] = ("failure", ex, traceback.format_exc())
        print(*answers[config])

    for config in configs:
        print("="*100)
        print(
            f"{Colors.blk}Config:{Colors.end}{Colors.fat}{config}{Colors.end}\n")
        if answers[config][0] == "success":
            print(
                f"{Colors.fat}Answer:{Colors.std}\n {answers[config][1]}{Colors.end}\n")
            print(
                f"{Colors.fat}Metadata:{Colors.std}\n {answers[config][2]}{Colors.end}\n")
        else:
            print(
                f"{Colors.wrn}Exception:\n {answers[config][1]}{Colors.end}\n")
            print(
                f"{Colors.wrn}Stacktrace:\n {answers[config][2]}{Colors.end}\n")
        print("="*100)
    return answers
