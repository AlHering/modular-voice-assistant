
# -*- coding: utf-8 -*-
"""

WARNING: LEGACY CODE - just for reference

****************************************************
*          Basic Language Model Backend            *
*            (c) 2023 Alexander Hering             *
****************************************************
"""
from typing import Tuple

# TODO: Plan out and implement common utility.
"""
Model backend overview
------------------------------------------
llama-cpp-python - GGML/GGUF run on CPU, offload layers to GPU, CUBLAS support (CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python)
- CPU: llama-cpp-python==0.2.18
- GPU: https://github.com/jllllll/llama-cpp-python-cuBLAS-wheels/releases/download/textgen-webui/llama_cpp_python_cuda-0.2.18+cu117-cp310-cp310-manylinux_2_31_x86_64.whl ; platform_system == "Linux" and platform_machine == "x86_64"

exllamav2 - 4-bit GPTQ weights, GPU inference (tested on newer GPUs > Pascal)
- CPU: exllamav2==0.0.9
- GPU: https://github.com/jllllll/llama-cpp-python-cuBLAS-wheels/releases/download/textgen-webui/llama_cpp_python_cuda-0.2.18+cu117-cp310-cp310-manylinux_2_31_x86_64.whl; platform_system == "Linux" and platform_machine == "x86_64"

auto-gptq - 4-bit GPTQ weights, GPU inference, can be used with Triton (auto-gptq[triton])
- CPU: auto-gptq==0.5.1
- GPU: https://github.com/jllllll/AutoGPTQ/releases/download/v0.5.1/auto_gptq-0.5.1+cu117-cp310-cp310-linux_x86_64.whl; platform_system == "Linux" and platform_machine == "x86_64"

gptq-for-llama - 4-bit GPTQ weights, GPU inference -> practically replaced by auto-gptq !
- CPU: gptq-for-llama==0.1.0
- GPU: https://github.com/jllllll/GPTQ-for-LLaMa-CUDA/releases/download/0.1.0/gptq_for_llama-0.1.0+cu117-cp310-cp310-linux_x86_64.whl; platform_system == "Linux" and platform_machine == "x86_64"

transformers - support common model architectures, CUDA support (e.g. via PyTorch)
- CPU: transformers==4.35.2
- GPU: use PyTorch e.g.:
    --extra-index-url https://download.pytorch.org/whl/cu117
    torch==2.0.1+cu117
    torchaudio==2.0.2+cu117
    torchvision==0.15.2+cu117

ctransformers - transformers C bindings, Cuda support (ctransformers[cuda])
- CPU: ctransformers==0.2.27
- GPU: ctransformers[cuda]==0.2.27 or https://github.com/jllllll/ctransformers-cuBLAS-wheels/releases/download/AVX2/ctransformers-0.2.27+cu117-py3-none-any.whl
"""


"""
Model instantiation functions
"""


def load_transformers_model(model_path: str,
                            model_file: str | None = None,
                            model_parameters: dict = {},
                            tokenizer_path: str | None = None,
                            tokenizer_parameters: dict = {},
                            embeddings_path: str | None = None,
                            embeddings_parameters: dict = {},
                            config_path: str | None = None,
                            config_parameters: dict = {}) -> Tuple:
    """
    Function for loading transformers based model objects.
    :param model_path: Path to model files.
    :param model_file: Model file to load.
        Defaults to None.
    :param model_parameters: Model loading kwargs as dictionary.
        Defaults to empty dictionary.
    :param tokenizer_path: Tokenizer path.
        Defaults to None.
    :param tokenizer_parameters: Tokenizer loading kwargs as dictionary.
        Defaults to empty dictionary.
    :param embeddings_path: Embeddings path.
        Defaults to None.
    :param embeddings_parameters: Embeddings loading kwargs as dictionary.
        Defaults to empty dictionary.
    :param config_path: Config path.
        Defaults to None.
    :param config_parameters: Config loading kwargs as dictionary.
        Defaults to empty dictionary.
    :return: Tuple of config, tokenizer, embeddings, model and generator object.
        Note, that all objects that do not belong to the backend will be None.
    """
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    config = None
    tokenizer = None
    embeddings = None
    model = None
    generator = None

    if config_path:
        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=config_path)
    if config_parameters is not None:
        for key in config_parameters:
            setattr(config, key, config_parameters[key])

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=tokenizer_path, **tokenizer_parameters) if tokenizer_path is not None else None
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_path, config=config, **model_parameters)

    return (config, tokenizer, embeddings, model, generator)