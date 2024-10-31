#!/bin/bash
docker run  \
    -it --net=host --gpus all \
    --mount type=bind,source="$1",target=/llama-cpp-server/models \
    --mount type=bind,source="$2",target=/llama-cpp-server/configs \
    "llama-cpp-server:v0.2" /bin/bash
