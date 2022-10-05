WORKSPACE=/home/yusuff/Desktop/Image_Class

docker run -it --rm \
	--gpus all \
	--net host \
	--shm-size 8G \
    -w $WORKSPACE \
	-v $WORKSPACE:$WORKSPACE \
	tensorflow_docker