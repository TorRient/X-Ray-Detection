# USE : python split.py --dir XR_SHOULDER
# Nhiệm vụ: cấu trúc lại dữ liệu
import os
from imutils import paths
import random
import shutil
import argparse
# tạo arg
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dir", required=True,
	help="path to input dataset (i.e., directory of images)")
args = vars(ap.parse_args())
# chọn tập class
ORIG_INPUT_DATASET1 = "MURA-v1.1/train/" + args["dir"]
ORIG_INPUT_DATASET2 = "MURA-v1.1/valid/" + args["dir"]
 
#output gốc
BASE_PATH = args["dir"]
 
#thư mục con của gốc
TRAIN_PATH = os.path.sep.join([BASE_PATH, "training"])
VAL_PATH = os.path.sep.join([BASE_PATH, "validation"])
TEST_PATH = os.path.sep.join([BASE_PATH, "testing"])
 
TRAIN_SPLIT = 0.85
 
VAL_SPLIT = 0.15
 

# đọc ảnh
imagePaths1 = list(paths.list_images(ORIG_INPUT_DATASET1))
imagePaths2 = list(paths.list_images(ORIG_INPUT_DATASET2))
imagePaths = imagePaths1 + imagePaths2

random.seed(42)
random.shuffle(imagePaths)
# 15% cho test
i = int(len(imagePaths) * TRAIN_SPLIT)
trainPaths = imagePaths[:i]
testPaths = imagePaths[i:]
 
# 70% train, 15% valid
i = int(len(trainPaths) * VAL_SPLIT)
valPaths = trainPaths[:i]
trainPaths = trainPaths[i:]
# tuple
datasets = [
	("training", trainPaths, TRAIN_PATH),
	("validation", valPaths, VAL_PATH),
	("testing", testPaths, TEST_PATH)
]
# loop over the datasets
for (dType, imagePaths, baseOutput) in datasets:
	# show which data split we are creating
	print("[INFO] building '{}' split".format(dType))
 
	# tạo thư mục gốc
	if not os.path.exists(baseOutput):
		print("[INFO] 'creating {}' directory".format(baseOutput))
		os.makedirs(baseOutput)
 
	# vòng lặp copy image
	for (i,inputPath) in enumerate(imagePaths):
		# extract the filename of the input image along with its
		# corresponding class label
		filename = inputPath.split(os.path.sep)[-1]
		label = inputPath.split(os.path.sep)[-2].split("_")[1]
 
		# Thư mục chứa ảnh
		labelPath = os.path.sep.join([baseOutput, label])
 
		# Nếu 3 thư mục training, valid, test chưa tồn tại thì tạo
		if not os.path.exists(labelPath):
			print("[INFO] 'creating {}' directory".format(labelPath))
			os.makedirs(labelPath)

		# Nếu ảnh đã tồn tại thì rename , tránh trùng lặp
		if os.path.exists(os.path.sep.join([labelPath,filename])):
			old_file = os.path.sep.join([labelPath,filename])
			tmp = "tmp" + str(i) +".png"
			new_file = os.path.sep.join([labelPath,tmp])
			os.rename(old_file,new_file) 
		p = os.path.sep.join([labelPath,filename])
		shutil.copy2(inputPath,p)
