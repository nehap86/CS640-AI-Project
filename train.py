# USAGE
# python train.py --dataset dataset --model fashion.model --labelbin mlb.pickle

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from pyimagesearch.smallervggnet import SmallerVGGNet
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
import random
import pickle
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()

#ap.add_argument("-d", "--dataset", required=True,
#	help="path to input dataset (i.e., directory of images)")
#ap.add_argument("-m", "--model", required=True,
#	help="path to output model")
#ap.add_argument("-l", "--labelbin", required=True,
#	help="path to output label binarizer")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
EPOCHS =30
INIT_LR = 1e-3
BS = 32
IMAGE_DIMS = (96, 96, 3)

# grab the image paths and randomly shuffle them
#print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images('dataset')))
#print(imagePaths)
random.seed(42)
random.shuffle(imagePaths)

data_train = []
labels_train = []
annotations = []
train_image_paths = sorted(list(paths.list_images('news_imgs/train')))
#print(train_image_paths)

with open('news_imgs/annot_train.txt', 'rt', encoding="utf8") as f:
	for line in f:
		annotations.append(line.split())

annotations = annotations[1:];
#print(annotations[:100])


i = 0
for imagePath in train_image_paths:
	if i > 3000:
		break;
	annotation = annotations[i]
	label = []
	if (annotation[1] == '1'):
		label.append('protest')
	if (annotation[3] == '1'):
		label.append('sign')
	if (annotation[4] == '1'):
		label.append('photo')
	if (annotation[5] == '1'):
		label.append('fire')
	if (annotation[6] == '1'):
		label.append('police')
	if (annotation[7] == '1'):
		label.append('children')
	if (annotation[8] == '1'):
		label.append('group_20')
	if (annotation[9] == '1'):
		label.append('group_100')
	if (annotation[10] == '1'):
		label.append('flag')
	if (annotation[11] == '1'):
		label.append('night')
	if (annotation[12] == '1'):
		label.append('shouting')
	labels_train.append(label)
	'''
	if i % 100 == 0:
		print("loading image ", i)
		print(imagePath)
		print(annotation)
		print(label)
	i = i + 1
	'''
	i = i + 1

	image = cv2.imread(imagePath)
	image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
	image = img_to_array(image)
	data_train.append(image)

data_train = np.array(data_train, dtype="float") / 255.0
labels_train = np.array(labels_train)
print("[INFO] data matrix: {} images ({:.2f}MB)".format(
	len(train_image_paths), data_train.nbytes / (1024 * 1000.0)))
print("[INFO] class labels:")
mlb_train = MultiLabelBinarizer()
labels_train = mlb_train.fit_transform(labels_train)
for (i, label) in enumerate(mlb_train.classes_):
	print("{}. {}".format(i + 1, label))


(trainX, testX, trainY, testY) = train_test_split(data_train,
	labels_train, test_size=0.2, random_state=42)
# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

# initialize the model using a sigmoid activation as the final layer
# in the network so we can perform multi-label classification
print("[INFO] compiling model...")
model = SmallerVGGNet.build(
	width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
	depth=IMAGE_DIMS[2], classes=len(mlb_train.classes_),
	finalAct="sigmoid")

# initialize the optimizer (SGD is sufficient)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

# compile the model using binary cross-entropy rather than
# categorical cross-entropy -- this may seem counterintuitive for
# multi-label classification, but keep in mind that the goal here
# is to treat each output label as an independent Bernoulli
# distribution
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the network

print("[INFO] training network...")
H = model.fit_generator(
	aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY),
	steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1)

# save the model to disk
print("[INFO] serializing network...")
model.save('fashion.model')

# save the multi-label binarizer to disk
print("[INFO] serializing label binarizer...")
f = open('mlb.pickle', "wb")
f.write(pickle.dumps(mlb_train))
f.close()

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig(args["plot"])