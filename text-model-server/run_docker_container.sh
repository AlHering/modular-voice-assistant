#!/bin/bash
docker run  \
    -it --net=host --gpus all \
    --mount type=bind,source="$1",target=/service/models \
    --mount type=bind,source="$2",target=/service/configs \
    "text-model-server:v0.3" /bin/bash
