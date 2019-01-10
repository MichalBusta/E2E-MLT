PRETRAINED_MODEL_PATH=~/pretrained_models

docker run --rm --shm-size 8G -it \
    -v ${PRETRAINED_MODEL_PATH}:/pretrained_models \
    -v `pwd`/../:/workspace \
    --name e2e-mlt e2e-mlt:latest \
    bash
