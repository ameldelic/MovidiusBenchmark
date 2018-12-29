import os

os.environ['GLOG_minloglevel'] = '3'  # suprress Caffe verbose prints

import cv2
import matplotlib.pyplot as plt
import caffe
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--mode', help="Mode on which to run the net can be CPU of GPU default is CPU", default='CPU')
parser.add_argument('--data', help="Data for benchmarking")

args = parser.parse_args()

print("----- Caffe AlexNet ------------------------------------------")