#USE : python trainResNet.py --dir XR_SHOULDER --plot plot.png

import matplotlib
matplotlib.use("Agg")

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from model.resnet import ResNet
from sklearn.metrics import classification_report
from sklearn.utils import class_weight
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dir", required=True , help = "dir of class")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

# epọch , Lr , Batch-size
NUM_EPOCHS = 1
INIT_LR = 1e-3
BS = 32
IMG_SIZE = [96,96,3]

# Giảm Lr theo từng epoch
def poly_decay(epoch):
	maxEpochs = NUM_EPOCHS
	baseLR = INIT_LR
	power = 1.0
	alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power
	return alpha

# path dir
TRAIN_PATH = args["dir"]+"training"
VAL_PATH = args["dir"]+"validation"
TEST_PATH = args["dir"]+"testing"

totalTrain = len(list(paths.list_images(TRAIN_PATH)))
totalVal = len(list(paths.list_images(VAL_PATH)))
totalTest = len(list(paths.list_images(TEST_PATH)))

# Khởi tạo ImageData
trainAug = ImageDataGenerator(
	rescale=1 / 255.0,
	rotation_range=20,
	zoom_range=0.05,
	width_shift_range=0.05,
	height_shift_range=0.05,
	shear_range=0.05,
	horizontal_flip=True,
	fill_mode="nearest")

valAug = ImageDataGenerator(rescale=1 / 255.0)

# Tạo tập trainGen mở rộng số lượng image
trainGen = trainAug.flow_from_directory(
	TRAIN_PATH,
	class_mode="categorical",
	target_size=(IMG_SIZE[0], IMG_SIZE[1]),
	color_mode="rgb",
	shuffle=True,
	batch_size=BS)

valGen = valAug.flow_from_directory(
	VAL_PATH,
	class_mode="categorical",
	target_size=(IMG_SIZE[0], IMG_SIZE[1]),
	color_mode="rgb",
	shuffle=False,
	batch_size=BS)

testGen = valAug.flow_from_directory(
	TEST_PATH,
	class_mode="categorical",
	target_size=(IMG_SIZE[0], IMG_SIZE[1]),
	color_mode="rgb",
	shuffle=False,
	batch_size=BS)

# Build ResNet
model = ResNet.build(IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2], 2, (3, 4, 6),
	(64, 128, 256, 512), reg=0.0005)
opt = SGD(lr=INIT_LR, momentum=0.9)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# callbacks giảm Lr và tính weights của Neg và Pos
callbacks = [LearningRateScheduler(poly_decay)]
weights = class_weight.compute_class_weight('balanced', np.unique(trainGen.classes), trainGen.classes)
H = model.fit_generator(
	trainGen,
	steps_per_epoch=totalTrain // BS,
	validation_data=valGen,
	validation_steps=totalVal // BS,
	epochs=NUM_EPOCHS,
	callbacks=callbacks)

print("[INFO] evaluating network...")
testGen.reset()
predIdxs = model.predict_generator(testGen,
	steps=(totalTest // BS) + 1)

predIdxs = np.argmax(predIdxs, axis=1)

# Show bảng F1, Acc , v.v
print(classification_report(testGen.classes, predIdxs,
	target_names=testGen.class_indices.keys()))

#save model 
print("[INFO] serializing network to '{}'...".format(args["model"]))
model.save(args["model"])

# Lưu plot
N = NUM_EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
