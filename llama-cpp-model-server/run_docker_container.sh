#!/bin/bash
docker run  \
    -it --net=host --gpus all \
    --mount type=bind,source="$2",target=/llama-cpp-model-server/models \
    --mount type=bind,source="$3",target=/llama-cpp-model-server/configs \
    "$1" /bin/bash
