#! /usr/bin/env python3

# Copyright(c) 2017 Intel Corporation.
# License: MIT See LICENSE file in root directory.

import sys
import numpy
import cv2
import time
import csv
import os
sys.path.insert(0, "ncapi2_shim")
import mvnc_simple_api as mvnc


mvnc.SetGlobalOption(mvnc.GlobalOption.LOG_LEVEL, 2)
devices = mvnc.EnumerateDevices()

if len(devices) == 0:
    print('No devices found')
    quit()

device = mvnc.Device(devices[0])
device.OpenDevice()
opt = device.GetDeviceOption(mvnc.DeviceOption.DEVICE_NAME)

# open the network blob files
blob = '../MovidiusModels/AgeNet.graph'

# categories for age and gender
age_list = ['0-2', '4-6', '8-12', '15-20', '25-32', '38-43', '48-53', '60-100']
gender_list = ['Male', 'Female']

# read in and pre-process the image:
ilsvrc_mean = numpy.load('../ncappzoo/data/age_gender/age_gender_mean.npy').mean(1).mean(1)  # loading the mean file

dim = (227, 227)
os.system('wget -O image.jpg -N http://vis-www.cs.umass.edu/lfw/images/Talisa_Bratt/Talisa_Bratt_0001.jpg')

img = cv2.imread('./image.jpg')
img = cv2.resize(img, dim)
img = img.astype(numpy.float32)

img[:,:,0] = (img[:,:,0] - ilsvrc_mean[0])
img[:,:,1] = (img[:,:,1] - ilsvrc_mean[1])
img[:,:,2] = (img[:,:,2] - ilsvrc_mean[2])

# execute the network with the input image on the NCS
with open(blob, mode='rb') as f:
    blob = f.read()

graph = device.AllocateGraph(blob)

t1 = cv2.getTickCount()

graph.LoadTensor(img.astype(numpy.float16), 'user object')
output, userobj = graph.GetResult()

t2 = cv2.getTickCount()

time = ((t2 - t1) / cv2.getTickFrequency()) * 1000


graph.DeallocateGraph()
device.CloseDevice()

print('\n------- predictions --------')
print("Prediction done in {} ms".format(time))

order = output.argsort()
last = len(order)-1
predicted=int(order[last])
print('the age range is ' + age_list[predicted] + ' with confidence of %3.1f%%' % (100.0*output[predicted]))