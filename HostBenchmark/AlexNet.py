import os
import numpy

import csv
from enter_benchmark import enter_benchmark

os.environ['GLOG_minloglevel'] = '3'  # suprress Caffe verbose prints

import cv2
import caffe
import argparse
import re

parser = argparse.ArgumentParser()

parser.add_argument('--mode', help="Mode on which to run the net can be CPU of GPU default is CPU", default='CPU')
parser.add_argument('--data', help="Data for benchmarking")

args = parser.parse_args()

print("----- Caffe AlexNet ------------------------------------------")

print("Running caffe model on {}".format(args.mode))

if args.mode == "GPU":
    caffe.set_device(0)
    caffe.set_mode_gpu()





MODEL_FILE = '../ncappzoo/caffe/AlexNet/deploy.prototxt'
PRETRAINED = '../ncappzoo/caffe/AlexNet/bvlc_alexnet.caffemodel'

net = caffe.Classifier(MODEL_FILE, PRETRAINED)


EXAMPLES_BASE_DIR = '../ncappzoo/'
imgname = EXAMPLES_BASE_DIR + 'data/images/nps_electric_guitar.png'

ilsvrc_mean = numpy.load(EXAMPLES_BASE_DIR + 'data/ilsvrc12/ilsvrc_2012_mean.npy').mean(1).mean(1)  # loading the mean file

img = cv2.imread(imgname)
img = cv2.resize(img, (227, 227))

img = img.astype(numpy.float32)

img[:, :, 0] = (img[:, :, 0] - ilsvrc_mean[0])
img[:, :, 1] = (img[:, :, 1] - ilsvrc_mean[1])
img[:, :, 2] = (img[:, :, 2] - ilsvrc_mean[2])

labels_file = EXAMPLES_BASE_DIR + 'data/ilsvrc12/synset_words.txt'
labels = numpy.loadtxt(labels_file, str, delimiter='\t')

t1 = cv2.getTickCount()

pred = net.predict([img])

t2 = cv2.getTickCount()

time = ((t2 - t1) / cv2.getTickFrequency()) * 1000

# ***************************************************************
# Print the results of the inference form the NCS
# ***************************************************************
order = pred[0].argsort()[::-1][:6]
print('\n------- predictions time {} ms --------'.format(time))

print(order)
result = ""

for i in range(0, 5):
    label = re.search("n[0-9]+\s([^,]+)", labels[order[i]]).groups(1)[0]
    result = result + "\n%20s %0.2f %%" % (label, pred[0][order[i]] * 100)

print(result)

bench = ['HOST', 'AlexNet', args.mode, time]

enter_benchmark('HOST', 'AlexNet', 'Caffe', args.mode, time)

