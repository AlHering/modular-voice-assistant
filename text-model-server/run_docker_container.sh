#!/bin/bash
docker run  \
    -it --net=host --gpus all \
    --mount type=bind,source="$2",target=/service/models \
    --mount type=bind,source="$3",target=/service/configs \
    "$1" /bin/bash
