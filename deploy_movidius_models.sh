#!/bin/bash

ROOT_FOLDER="$PWD"

function check_ncsk() {
    type mvNCCompile >/dev/null 2>&1 || { echo >&2 "Check NCSDK for Movidius ................. FAIL"; exit 1; }
    echo "Check NCSDK for Movidius ................. OK"
}

function compile_app_zoo() {
    echo "Compiling caffe and tensorflow from ncappzoo"

	cd $ROOT_FOLDER/ncappzoo/caffe
	make all

	cd $ROOT_FOLDER/ncappzoo/tensorflow
	make all

	cd $ROOT_FOLDER
}


function copy_movidius_graphs() {

    echo "Copy movidus models from caffe and tensorflow"
    
    if [ ! -d "$ROOT_FOLDER/MovidiusModels" ]; then
        # Control will enter here if DIRECTORY doesn't exist.
        mkdir $ROOT_FOLDER/MovidiusModels
    fi

    # Caffe models
    cp $ROOT_FOLDER/ncappzoo/caffe/AgeNet/graph $ROOT_FOLDER/MovidiusModels/AgeNet.graph
    cp $ROOT_FOLDER/ncappzoo/caffe/AlexNet/graph $ROOT_FOLDER/MovidiusModels/AlexNet.graph
    cp $ROOT_FOLDER/ncappzoo/caffe/GenderNet/graph $ROOT_FOLDER/MovidiusModels/GenderNet.graph
    cp $ROOT_FOLDER/ncappzoo/caffe/GoogLeNet/graph $ROOT_FOLDER/MovidiusModels/GoogleNet.graph
    cp $ROOT_FOLDER/ncappzoo/caffe/ResNet-18/graph $ROOT_FOLDER/MovidiusModels/ResNet-18.graph
    cp $ROOT_FOLDER/ncappzoo/caffe/SqueezeNet/graph $ROOT_FOLDER/MovidiusModels/SqueezeNet.graph
    cp $ROOT_FOLDER/ncappzoo/caffe/SSD_MobileNet/graph $ROOT_FOLDER/MovidiusModels/SSD_Mobilenet.graph
    cp $ROOT_FOLDER/ncappzoo/caffe/TinyYolo/graph $ROOT_FOLDER/MovidiusModels/TinyYolo.graph

    # Tensorflow models
    cp $ROOT_FOLDER/ncappzoo/tensorflow/facenet/facenet_celeb_ncs.graph $ROOT_FOLDER/MovidiusModels/facenet.graph
    cp $ROOT_FOLDER/ncappzoo/tensorflow/inception/model/v3/graph $ROOT_FOLDER/MovidiusModels/inception.graph
    cp $ROOT_FOLDER/ncappzoo/tensorflow/inception_resnet_v2/graph $ROOT_FOLDER/MovidiusModels/inception_resnet_v2.graph
    cp $ROOT_FOLDER/ncappzoo/tensorflow/inception_v1/graph $ROOT_FOLDER/MovidiusModels/inception_v1.graph
    cp $ROOT_FOLDER/ncappzoo/tensorflow/inception_v2/graph $ROOT_FOLDER/MovidiusModels/inception_v2.graph
    cp $ROOT_FOLDER/ncappzoo/tensorflow/inception_v3/graph $ROOT_FOLDER/MovidiusModels/inception_v3.graph
    cp $ROOT_FOLDER/ncappzoo/tensorflow/inception_v4/graph $ROOT_FOLDER/MovidiusModels/inception_v4.graph
    cp $ROOT_FOLDER/ncappzoo/tensorflow/mnist/mnist_inference.graph $ROOT_FOLDER/MovidiusModels/mnist.graph
    cp $ROOT_FOLDER/ncappzoo/tensorflow/mobilenets/model/graph $ROOT_FOLDER/MovidiusModels/mobilenets.graph
    cp $ROOT_FOLDER/ncappzoo/tensorflow/tiny_yolo_v2/graph $ROOT_FOLDER/MovidiusModels/tiny_yolo_v2.graph
}


function benchmark_movidus_models() {
    echo "BENCHMARK"    
    cp -a $ROOT_FOLDER/ncappzoo/ncapi2_shim/. $ROOT_FOLDER/benchmark/ncapi2_shim
    
    

}

function main() {
    check_ncsk
    #compile_app_zoo
    copy_movidius_graphs
    benchmark_movidus_models
}


main
exit 0



