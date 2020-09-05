# Meme-Generator

You all must have seen/heard of the infamous Deal With It meme!

This project is an automatic meme generator which takes any static image with faces as our input and outputs animated GIFs. It is written in python and powered through OpenCV and DLib.

This repository also includes a streamlit as well as a flask webapp to create memes

![DEAL WITH IT](https://github.com/arpita505/Meme-Generator/blob/master/static/uploads/downloaded_gif.gif?raw=true)

## Prerequisites:

Pillow, MoviePy, and NumPy for the Gif from still image generator, and OpenCV and Pillow for the real time DEAL generator.

You'll need a webcam to get real time video in OpenCV to work.

## Working
    
Clone this repository in a folder :
    
    git clone https://github.com/arpita505/Meme-Generator.git
    
To run the python file directly:

    python3 generator.py -image SOURCEIMAGE.jpg 
    
To run the realtime video stream to create memes:

    python3 realtime.py
    
To run the streamlit webapp :

    streamlit run streamlitApp.py
    
To run the flask webapp :

    python3 main.py
  
## Streamlit webapp

<img src="https://github.com/arpita505/Meme-Generator/blob/master/readme_images/webapp2.png" width="800"> 
 
## Flask webapp

![DEAL WITH IT](https://github.com/arpita505/Meme-Generator/blob/master/readme_images/flaskgif.gif?raw=true)

*Also added music effect in webapp on preview of meme (https://github.com/arpita505/Meme-Generator/blob/master/readme_images/flaskApp.mov)* 

Building a meme generator using OpenCV can teach us a number of valuable techniques used in practice, including:

  - How to perform deep learning-based face detection.

  - How to use the dlib library to apply facial landmark detection and extract the eye regions.

  - How to take these two regions and compute the rotation angle between the eyes.

  - And finally, how to generate animated GIFs with OpenCV (with a little help from ImageMagick).


In order to build our meme generator, we leveraged computer vision and deep learning in a number of practical ways, including:
- Face detection
- Facial landmark prediction
- Extracting regions of the face (in this case, the eyes)
- Computing the angle between the eyes, a requirement for face alignment
- Generating transparent overlays via alpha blending 
