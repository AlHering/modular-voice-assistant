#!/bin/bash
docker run  \
    -it --net=host --gpus all \
    --mount type=bind,source="$1",target=/service/models \
    --mount type=bind,source="$2",target=/service/configs \
    "audio-model-server:v0.1" /bin/bash
