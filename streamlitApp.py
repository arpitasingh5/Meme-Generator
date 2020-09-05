import streamlit as st
import dlib
from PIL import Image
import argparse
import os
from werkzeug.utils import secure_filename
from imutils import face_utils
import numpy as np
import moviepy.editor as mpy
import base64

def generate_gif():
    global img
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('assets/shape_predictor_68_face_landmarks.dat')

    # resize to a max_width to keep gif size small
    max_width = 500

    # open our image, convert to rgba
    img = Image.open('static/uploads/out.png').convert('RGBA')

    memePicture = Image.open("assets/memePicture.png")
    memeText = Image.open('assets/memeText.png')

    if img.size[0] > max_width:
        scaled_height = int(max_width * img.size[1] / img.size[0])
        img.thumbnail((max_width, scaled_height))

    img_gray = np.array(img.convert('L')) # need grayscale for dlib face detection

    rects = detector(img_gray, 0)

    if len(rects) == 0:
        print("No faces found, exiting.")

    print("%i faces found in source image. processing into gif now." % len(rects))

    faces = []

    for rect in rects:
        face = {}
        print(rect.top(), rect.right(), rect.bottom(), rect.left())
        shades_width = rect.right() - rect.left()

        # predictor used to detect orientation in place where current face is
        shape = predictor(img_gray, rect)
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

        # resize glasses to fit face width
        currentMemePicture = memePicture.resize((shades_width, int(shades_width * memePicture.size[1] / memePicture.size[0])),
                                   resample=Image.LANCZOS)
        # rotate and flip to fit eye centers
        currentMemePicture = currentMemePicture.rotate(angle, expand=True)
        currentMemePicture = currentMemePicture.transpose(Image.FLIP_TOP_BOTTOM)

        # add the scaled image to a list, shift the final position to the
        # left of the leftmost eye
        face['glasses_image'] = currentMemePicture
        leftEyeX = leftEye[0,0] - shades_width // 4
        leftEyeY = leftEye[0,1] - shades_width // 6
        face['final_pos'] = (leftEyeX, leftEyeY)
        faces.append(face)

    # how long our gif should be
    duration = 4

    def make_frame(t):
        finalImg = img.convert('RGBA') # returns copy of original image

        if t == 0: # no glasses first image
            return np.asarray(finalImg)

        for face in faces:
            if t <= duration - 2:
                currentX = int(face['final_pos'][0])
                currentY = int(face['final_pos'][1] * t / (duration - 2))
                finalImg.paste(face['glasses_image'], (currentX, currentY) , face['glasses_image'])
            else:
                finalImg.paste(face['glasses_image'], face['final_pos'], face['glasses_image'])
                finalImg.paste(memeText, (75, finalImg.height - 65), memeText)

        return np.asarray(finalImg)


    animation = mpy.VideoClip(make_frame, duration = duration)
    animation.write_gif("static/uploads/downloaded_gif.gif", fps=4)

generate_gif()

def meme_generator():

	global extension
	st.title("Meme Generator")
	activities = ["Image" ,"Webcam"]
	st.set_option('deprecation.showfileUploaderEncoding', False)
	choice = st.sidebar.selectbox("Meme Generator on?",activities)

	if choice == 'Image':
		st.subheader("Meme Generator")
		image_file = st.file_uploader("Upload Image",type=['png', 'jpg', 'jpeg', 'gif']) #upload image
		if image_file is not None:
			our_image = Image.open(image_file)
			im = our_image.save('static/uploads/out.png')
			saved_image = st.image(image_file , caption='image uploaded successfully', use_column_width=True)
			if st.button('Process'):
				file_ = open("static/uploads/downloaded_gif.gif", "rb")
				contents = file_.read()
				data_url = base64.b64encode(contents).decode("utf-8")
				file_.close()
				st.markdown(f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',unsafe_allow_html=True,)

	if choice == 'Webcam':
		st.subheader("Detection on webcam")
		st.text("This feature will be avilable soon")

meme_generator()
