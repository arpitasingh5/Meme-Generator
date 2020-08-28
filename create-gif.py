from imutils import face_utils
from imutils import paths
import numpy as np
import argparse
import imutils
import shutil
import json
import dlib
import cv2
import sys
import os

def overlayImage(background, foreground, foregroundMask, coordinates):

    (height, width) = foreground.shape[:2]
    (x, y) = coordinates

    overlay = np.zeros(background.shape, dtype="uint8")
    overlay[y:y + height, x:x + width] = foreground

    alpha = np.zeros(background.shape[:2], dtype="uint8")
    alpha[y:y + height, x:x + width] = foregroundMask
    alpha = np.dstack([alpha] * 3)

    #perform alpha blending to merge the foreground, background,
    # and alpha channel together
    output = alpha_blend(overlay, background, alpha)

    return output

def alpha_blend(foreground, background, alpha):

    foreground = foreground.astype("float")
    background = background.astype("float")
    alpha = alpha.astype("float") / 255
    # perform alpha blending
    foreground = cv2.multiply(alpha, foreground)
    background = cv2.multiply(1 - alpha, background)

    output = cv2.add(foreground, background)
    return output.astype("uint8")


def create_gif(inputPath, outputPath, delay, finalDelay, loop):

    imagePaths = sorted(list(paths.list_images(inputPath)))
    lastPath = imagePaths[-1]
    imagePaths = imagePaths[:-1]
    cmd = "convert -delay {} {} -delay {} {} -loop {} {}".format(
        delay, " ".join(imagePaths), finalDelay, lastPath, loop,
        outputPath)
    os.system(cmd)

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--config", required=True,
    help="path to configuration file")
ap.add_argument("-i", "--image", required=True,
    help="path to input image")
ap.add_argument("-o", "--output", required=True,
    help="path to output GIF")
args = vars(ap.parse_args())

config = json.loads(open(args["config"]).read())
sunglasses = cv2.imread(config["sunglasses"])
sunglassesMask = cv2.imread(config["sunglasses_mask"])
shutil.rmtree(config["temp_dir"], ignore_errors=True)
os.makedirs(config["temp_dir"])

print("[INFO] loading models...")
detector = cv2.dnn.readNetFromCaffe(config["face_detector_prototxt"],
    config["face_detector_weights"])
predictor = dlib.shape_predictor(config["landmark_predictor"])

image = cv2.imread(args["image"])
(height, width) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
    (300, 300), (104.0, 177.0, 123.0))

print("[INFO] computing object detections...")
detector.setInput(blob)
detections = detector.forward()

i = np.argmax(detections[0, 0, :, 2])
confidence = detections[0, 0, i, 2]

if confidence < config["min_confidence"]:
    print("[INFO] no reliable faces found")
    sys.exit(0)

box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
(startX, startY, endX, endY) = box.astype("int")

rect = dlib.rectangle(int(startX), int(startY), int(endX), int(endY))
shape = predictor(image, rect)
shape = face_utils.shape_to_np(shape)

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
leftEyePts = shape[lStart:lEnd]
rightEyePts = shape[rStart:rEnd]

# compute the center of mass for each eye
leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
rightEyeCenter = rightEyePts.mean(axis=0).astype("int")
# compute the angle between the eye centroids
dY = rightEyeCenter[1] - leftEyeCenter[1]
dX = rightEyeCenter[0] - leftEyeCenter[0]
angle = np.degrees(np.arctan2(dY, dX)) - 180

sunglasses = imutils.rotate_bound(sunglasses, angle)
sunglassesWidth = int((endX - startX) * 0.9)
sunglasses = imutils.resize(sunglasses, width=sunglassesWidth)

sunglassesMask = cv2.cvtColor(sunglassesMask, cv2.COLOR_BGR2GRAY)
sunglassesMask = cv2.threshold(sunglassesMask, 0, 255, cv2.THRESH_BINARY)[1]
sunglassesMask = imutils.rotate_bound(sunglassesMask, angle)
sunglassesMask = imutils.resize(sunglassesMask, width=sunglassesWidth, inter=cv2.INTER_NEAREST)

steps = np.linspace(0, rightEyeCenter[1], config["steps"],
    dtype="int")
# start looping over the steps
for (i, y) in enumerate(steps):
    shiftX = int(sunglasses.shape[1] * 0.25)
    shiftY = int(sunglasses.shape[0] * 0.35)
    y = max(0, y - shiftY)

    # add the sunglasses to the image
    output = overlayImage(image, sunglasses, sunglassesMask, (rightEyeCenter[0] - shiftX, y))

    if i == len(steps) - 1:
        text = cv2.imread(config["deal_with_it"])
        textMask = cv2.imread(config["deal_with_it_mask"])
        textMask = cv2.cvtColor(textMask, cv2.COLOR_BGR2GRAY)
        textMask = cv2.threshold(textMask, 0, 255,
            cv2.THRESH_BINARY)[1]
        oW = int(width * 0.8)
        text = imutils.resize(text, width=oW)
        textMask = imutils.resize(textMask, width=oW,
            inter=cv2.INTER_NEAREST)

        oX = int(width * 0.1)
        oY = int(height * 0.8)
        output = overlayImage(output, text, textMask, (oX, oY))

    p = os.path.sep.join([config["temp_dir"], "{}.jpg".format(
        str(i).zfill(8))])
    cv2.imwrite(p, output)

print("[INFO] creating GIF...")
create_gif(config["temp_dir"], args["output"], config["delay"],
    config["final_delay"], config["loop"])

# cleanup by deleting our temporary directory
print("[INFO] cleaning up...")
shutil.rmtree(config["temp_dir"], ignore_errors=True)