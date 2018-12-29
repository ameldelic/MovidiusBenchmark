import os

os.environ['GLOG_minloglevel'] = '3'  # suprress Caffe verbose prints

import cv2
import matplotlib.pyplot as plt
import caffe
import argparse
import csv


parser = argparse.ArgumentParser()

parser.add_argument('--mode', help="Mode on which to run the net can be CPU of GPU default is CPU", default='CPU')
parser.add_argument('--data', help="Data for benchmarking")

args = parser.parse_args()

print("----- Caffe AgeNet ------------------------------------------")

data_path = '/media/amel/data/BenchmarkData/AgeNet'


if not (args.mode != "GPU") and not (args.mode != "CPU"):
    print("Unknown mode {} need GPU or CPU".format(args.mode))
    quit(1)

if not os.path.exists("AgeGenderDeepLearning"):
    os.system('git clone https://github.com/GilLevi/AgeGenderDeepLearning.git')

print("Running caffe model on {}".format(args.mode))

if args.mode == "GPU":
    caffe.set_device(0)
    caffe.set_mode_gpu()


age_list=['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
gender_list=['Male', 'Female']

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

mean_filename = './AgeGenderDeepLearning/models/mean.binaryproto'

proto_data = open(mean_filename, "rb").read()
a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
mean = caffe.io.blobproto_to_array(a)[0]

age_net_pretrained = './AgeGenderDeepLearning/models/age_net.caffemodel'
age_net_model_file = './AgeGenderDeepLearning/age_net_definitions/deploy.prototxt'


age_net = caffe.Classifier(age_net_model_file, age_net_pretrained,
                       channel_swap=(2, 1, 0),
                       raw_scale=255,
                       image_dims=(256, 256))

if not os.path.exists("./agenet_image.jpg"):
    os.system('wget -O agenet_image.jpg -N http://vis-www.cs.umass.edu/lfw/images/Talisa_Bratt/Talisa_Bratt_0001.jpg')

input_image = caffe.io.load_image('./agenet_image.jpg')
prediction = age_net.predict([input_image])

data_read = []

with open(data_path + '/fold_0_data.txt', 'r') as f:
    reader = csv.reader(f,  delimiter='\t')
    j = 0
    for read in reader:
        if j > 0:
            data_read.append(read)
        j += 1


data_count = 0
miss_count = 0

for row in data_read:
    img_path = data_path + "/faces/{}/coarse_tilt_aligned_face.{}.{}".format(row[0], row[2], row[1])
    age = row[3]

    if age == 'None':
        # print("IMG {} has no age".format(img_path))
        continue

    if age not in age_list:
        # print("Age doesn't {} exist".format(age))
        continue

    index_age = age_list.index(age)
    input_image = caffe.io.load_image(img_path)

    prediction = age_net.predict([input_image])
    predict_age = prediction[0].argmax()

    if index_age != predict_age:
        miss_count += 1

    data_count += 1

    # print("img {} age {} prediction {}".format(img_path, index_age, predict_age))

print("Accuracy {}".format(miss_count / data_count))

quit()

t1 = cv2.getTickCount()
prediction = age_net.predict([input_image])

t2 = cv2.getTickCount()
time = ((t2 - t1) / cv2.getTickFrequency()) * 1000

print("Prediction time {} ms".format(time))
print('predicted age:', age_list[prediction[0].argmax()])