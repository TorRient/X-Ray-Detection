# USE : python classifyReNet.py --images example.png --model saved_model.model

from keras.preprocessing.image import img_to_array
from keras.models import load_model
from imutils import build_montages
from imutils import paths
import imutils
import numpy as np
import argparse
import random
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
	help="path to out input directory of images")
ap.add_argument("-m", "--model", required=True,
	help="path to pre-trained model")
args = vars(ap.parse_args())

# Load model đã train
print("[INFO] loading pre-trained network...")
model = load_model(args["model"])
# Read image
image = cv2.imread(args["images"], cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (128, 128))
image = image.astype("float") / 255.0
output = imutils.resize(image, width=400)
	# order channel dimensions (channels-first or channels-last)
	# depending on our Keras backend, then add a batch dimension to
	# the image
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# Pridict image
pred = model.predict(image)
tmp = pred
pred = pred.argmax(axis=1)[0]
result = tmp[0][pred]
print(result)
# 0 = Neg , 1 = Pos
label = "Negative : " + str(result*100) if pred == 0 else "Positive : " + str(result*100)
color = (22, 255, 22) if pred == 0 else (22, 255, 22)

# Write text vào image
cv2.putText(output, label, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
		color, 2)

# Show result
cv2.imshow("Results", output)
cv2.waitKey(0)