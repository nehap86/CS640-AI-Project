# USAGE
# python classify.py --model fashion.model --labelbin mlb.pickle --image examples/example_01.jpg

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
from imutils import paths
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
#ap.add_argument("-m", "--model", required=True,
#	help="path to trained model model")
#ap.add_argument("-l", "--labelbin", required=True,
#	help="path to label binarizer")
#ap.add_argument("-i", "--image", required=True,
#	help="path to input image")
args = vars(ap.parse_args())



model = load_model('fashion.model')
mlb = pickle.loads(open('mlb.pickle', "rb").read())

test_image_paths = sorted(list(paths.list_images('news_imgs/test')))

annotations = []
annotations_train = []
num_of_none_acc = 0.0
num_of_none_gt = 0.0
labels_test = []
num_of_errors = [0] * 12
label_recall = {
	'children': 0.0,
	'fire': 0.0,
	'flag': 0.0,
	'group_100': 0.0,
	'group_20': 0.0,
	'night': 0.0,
	'photo': 0.0,
	'police': 0.0,
	'protest': 0.0,
	'shouting': 0.0,
	'sign': 0.0,
}
label_total = {
	'children': 0.0,
	'fire': 0.0,
	'flag': 0.0,
	'group_100': 0.0,
	'group_20': 0.0,
	'night': 0.0,
	'photo': 0.0,
	'police': 0.0,
	'protest': 0.0,
	'shouting': 0.0,
	'sign': 0.0,
}
label_total_train = {
	'children': 0.0,
	'fire': 0.0,
	'flag': 0.0,
	'group_100': 0.0,
	'group_20': 0.0,
	'night': 0.0,
	'photo': 0.0,
	'police': 0.0,
	'protest': 0.0,
	'shouting': 0.0,
	'sign': 0.0,
}




with open('news_imgs/annot_test.txt', 'rt', encoding="utf8") as f:
	for line in f:
		annotations.append(line.split())
with open('news_imgs/annot_train.txt', 'rt', encoding="utf8") as f:
	for line in f:
		annotations_train.append(line.split())
annotations = annotations[1:]
annotations_train = annotations_train[1:]

j = 0
for ant in annotations_train:
	if j > 30000:
		break;
	if (ant[1] == '1'):
		label_total_train['protest'] = label_total_train['protest'] + 1
	if (ant[3] == '1'):
		label_total_train['sign'] = label_total_train['sign'] + 1
	if (ant[4] == '1'):
		label_total_train['photo'] = label_total_train['photo'] + 1
	if (ant[5] == '1'):
		label_total_train['fire'] = label_total_train['fire'] + 1
	if (ant[6] == '1'):
		label_total_train['police'] = label_total_train['police'] + 1
	if (ant[7] == '1'):
		label_total_train['children'] = label_total_train['children'] + 1
	if (ant[8] == '1'):
		label_total_train['group_20'] = label_total_train['group_20'] + 1
	if (ant[9] == '1'):
		label_total_train['group_100'] = label_total_train['group_100'] + 1
	if (ant[10] == '1'):
		label_total_train['flag'] = label_total_train['flag'] + 1
	if (ant[11] == '1'):
		label_total_train['night'] = label_total_train['night'] + 1
	if (ant[12] == '1'):
		label_total_train['shouting'] = label_total_train['shouting'] + 1



i = 0
for imagePath in test_image_paths:
	'''
	if i < 100:
		print(imagePath)
	i = i + 1
	'''
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (96, 96))
	image = image.astype("float") / 255.0
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	proba = model.predict(image)[0]

	predict_label = []
	for (l, p) in zip(mlb.classes_, proba):
		if p * 100 > 15:
			predict_label.append(l)




	annotation = annotations[i]
	label = []
	error = 0
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

	if len(label) == 0:
		num_of_none_gt = num_of_none_gt + 1
	if len(label) == 0 and len(predict_label) == 0:
		num_of_none_acc = num_of_none_acc + 1

	for c in mlb.classes_:
		if (c in predict_label and c not in label) or (c not in predict_label and c in label):
			error = error + 1
		if c in label:
			label_total[c] = label_total[c] + 1
		if c in label and c in predict_label:
			label_recall[c] = label_recall[c] + 1
	num_of_errors[error] = num_of_errors[error] + 1

	if i < 5:
		print(mlb.classes_)
		print(proba)
		print('annotation', annotation)
		print('label', label)
		print('predict_label', predict_label)
		print(label_recall)
		print()
	i = i + 1




print("num_of_none_gt is: ", num_of_none_gt)
print("num_of_none_predict_right is: ", num_of_none_acc)
print("num_of_none_acc is: ", num_of_none_acc/num_of_none_gt)
print(num_of_errors)
for c in label_recall:
	label_recall[c] = label_recall[c] / label_total[c]
print("label_total is: ", label_total)
print("label_recall is: ", label_recall)

plabels = ['0', '1', '2', '3', '4', '5']

fig = plt.figure()
plt.pie(num_of_errors[:6], labels=plabels, autopct='%1.2f%%')
plt.title("number of error-labeling")

plt.savefig("PieChart.png")




fig = plt.figure()
#draw the recall picture
label_list = ['children', 'fire', 'flag', 'group_100', 'group_20', 'night', 'photo', 'police',
 'protest', 'shouting', 'sign']    # 横坐标刻度显示值
num_list1 = [label_recall[k] for k in label_list]      # 纵坐标值1
x = range(len(num_list1))
label_list = ['children', 'fire', 'flag', 'gp100', 'gp20', 'night', 'photo', 'police',
 'protest', 'shouting', 'sign']    # 横坐标刻度显示值

rects1 = plt.bar(x, height=num_list1, width=0.4, alpha=0.8, color='blue')
plt.ylim(0, 1)     # y轴取值范围
plt.ylabel("label's recall")


plt.xticks([index + 0.2 for index in x], label_list)
plt.xlabel("label names")
plt.title("the recalls of labels")
plt.legend()     # 设置题注

plt.savefig("Histogram.png")



fig = plt.figure()
#draw the recall picture
label_list = ['children', 'fire', 'flag', 'group_100', 'group_20', 'night', 'photo', 'police',
 'protest', 'shouting', 'sign']    # 横坐标刻度显示值
num_list1 = [label_total_train[k] for k in label_list]      # 纵坐标值1
x = range(len(num_list1))
label_list = ['children', 'fire', 'flag', 'gp100', 'gp20', 'night', 'photo', 'police',
 'protest', 'shouting', 'sign']    # 横坐标刻度显示值

rects1 = plt.bar(x, height=num_list1, width=0.4, alpha=0.8, color='red')
plt.ylim(0, 10000)     # y轴取值范围
plt.ylabel("")


plt.xticks([index + 0.2 for index in x], label_list)
plt.xlabel("label names")
plt.title("number of training samples")
plt.legend()     # 设置题注

plt.savefig("Histogram2.png")