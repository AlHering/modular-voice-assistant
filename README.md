# Modular Voice Assistant
Under construction.

## Code overview
### Porject source code
`src/` contains the main project source code, split in
- `backend/`
- `frontend/`
- common `configuration/`
- common `utility/` 

`language-model-server/` contains a decoupled LlamaCPP server, which can be used as is or as docker container, serving language models via an OpenAI-compatible API. It can be utilized for other projects, either using the endpoints or the abstraction `RemoteChatModelInstance` found in `src/services/abstractions/language_model_abstractions.py`.

### Top level files and folders
- `cpu.Dockerfile` can be used to build a CPU-based docker container (currently untested)
- `gpu.Dockerfile` can be used to build a GPU-based docker container for CUDA-compatible systems (set up for CUDA 12.1)
- `install.sh` is used by the docker files, to install the necessary requirements into a virtual Python environment
- `requirements_cpu.txt` contain the requirements for CPU-based systems
- `requirements_gpu.txt` contain the requirements for GPU-based (CUDA 12.1) systems
- `run_backend.py` is the main runner file for the Python backend
- `run_frontend_.py` is the main runner file for the Python (Streamlit) frontend
- `run.sh` is the main runner file for the project (starts backend and frontend) and entrypoint for docker containers

- `.streamlit` contains a minimalistic streamlit configuration
- `configs` can be used to mount or link config files for modular voice assistants (and its components)
- `data` will be used to store productive data for different operations
- `docs` contains a basic documentation (not yet)
- `language-model-server` contains the code for a decoupled (standalone) LlamaCPP based language model server
- `models` can be used to mount or link machine learning model files for modular voice assistants (and its components)
- `src` contains the main project source code


