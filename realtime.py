import dlib
from PIL import Image, ImageDraw, ImageFont
import argparse

import cv2

from imutils.video import VideoStream
from imutils import face_utils, translate, rotate, resize

import numpy as np

def start_stream():

    vs = VideoStream().start()

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('assets/shape_predictor_68_face_landmarks.dat')

    max_width = 700
    frame = vs.read()
    frame = resize(frame, width=max_width)

    fps = vs.stream.get(cv2.CAP_PROP_FPS) # need this for animating proper duration

    animation_length = fps * 5
    current_animation = 0
    glasses_on = fps * 3

    memePicture = Image.open("assets/memePicture.png")
    text = Image.open('assets/memeText.png')

    dealing = False

    while True:
        frame = vs.read()

        frame = resize(frame, width=max_width)

        imgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = []

        rects = detector(imgGray, 0)
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        for rect in rects:
            face = {}
            shades_width = rect.right() - rect.left()

            # predictor used to detect orientation in place where current face is
            shape = predictor(imgGray, rect)
            shape = face_utils.shape_to_np(shape)

            # grab the outlines of each eye from the input image
            leftEye = shape[36:42]
            rightEye = shape[42:48]

            # compute the center of mass for each eye
            leftEyeCenter = leftEye.mean(axis=0).astype("int")
            rightEyeCenter = rightEye.mean(axis=0).astype("int")

            # compute the angle between the eye centroids
            angleY = leftEyeCenter[1] - rightEyeCenter[1]
            angleX = leftEyeCenter[0] - rightEyeCenter[0]
            angle = np.rad2deg(np.arctan2(angleY, angleX))

            currentMemePicture = memePicture.resize((shades_width, int(shades_width * memePicture.size[1] / memePicture.size[0])),
                                   resample=Image.LANCZOS)
            currentMemePicture = currentMemePicture.rotate(angle, expand=True)
            currentMemePicture = currentMemePicture.transpose(Image.FLIP_TOP_BOTTOM)

            face['glasses_image'] = currentMemePicture
            leftEyeX = leftEye[0,0] - shades_width // 4
            leftEyeY = leftEye[0,1] - shades_width // 6
            face['final_pos'] = (leftEyeX, leftEyeY)


            if dealing:
                if current_animation < glasses_on:
                    currentY = int(current_animation / glasses_on * leftEyeY)
                    img.paste(currentMemePicture, (leftEyeX, currentY), currentMemePicture)
                else:
                    img.paste(currentMemePicture, (leftEyeX, leftEyeY), currentMemePicture)
                    img.paste(text, (75, img.height - 65), text)

        if dealing:
            current_animation += 1

            if current_animation > animation_length:
                dealing = False
                current_animation = 0
            else:
                frame = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

        cv2.imshow("Meme generator", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        if key == ord("d"):
            dealing = not dealing

    cv2.destroyAllWindows()
    vs.stop()

start_stream()
