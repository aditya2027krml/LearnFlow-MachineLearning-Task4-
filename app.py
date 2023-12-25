import streamlit as st
import cv2
import numpy as np
from urllib import request
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow_docs.vis import embed
import imageio
import os
import tempfile
import ssl

# Disable TensorFlow warnings
tf.get_logger().setLevel('ERROR')

# Function to load and process the video
def load_video(file):
    cap = cv2.VideoCapture(file)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, (224, 224))
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)
    finally:
        cap.release()
    return np.array(frames) / 255.0

# Function to crop the center square of a frame
def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y:start_y + min_dim, start_x:start_x + min_dim]

# Function to create a gif from images
def to_gif(images):
    converted_images = np.clip(images * 255, 0, 255).astype(np.uint8)
    imageio.mimsave('./animation.gif', converted_images, fps=25)
    return embed.embed_file('./animation.gif')

# Function to fetch UCF video
def fetch_ucf_video(video):
    video = video.strip()  # Remove leading and trailing spaces
    cache_path = os.path.join(_CACHE_DIR, video)
    if not os.path.exists(cache_path):
        urlpath = request.urljoin(UCF_ROOT, video)
        print("Fetching %s => %s" % (urlpath, cache_path))
        data = request.urlopen(urlpath, context=unverified_context).read()
        open(cache_path, "wb").write(data)
    return cache_path

# Function to make predictions
def predict(sample_video):
    model_input = tf.constant(sample_video, dtype=tf.float32)[tf.newaxis, ...]
    logits = i3d(model_input)['default'][0]
    probabilities = tf.nn.softmax(logits)

    st.subheader("Top 5 Actions:")
    for i in np.argsort(probabilities)[::-1][:5]:
        st.write(f"{labels[i]:22}: {probabilities[i] * 100:5.2f}%")

# Set up Streamlit app
        
st.text("AUTHOR : <ADITYA KUMAR>")
st.text("TASK 4")
st.text("Human Activity Recognition")
st.title("Human Activity Video Analysis App")

# UCF101 dataset information
UCF_ROOT = "https://www.crcv.ucf.edu/THUMOS14/UCF101/UCF101/"
_CACHE_DIR = tempfile.mkdtemp()
unverified_context = ssl._create_unverified_context()

# Load label information
KINETICS_URL = "https://raw.githubusercontent.com/deepmind/kinetics-i3d/master/data/label_map.txt"
with request.urlopen(KINETICS_URL) as obj:
    labels = [line.decode("utf-8").strip() for line in obj.readlines()]

# Load I3D model
i3d = hub.load("https://tfhub.dev/deepmind/i3d-kinetics-400/1").signatures['default']

# File input textbox
file_name = st.text_input("Enter video file name (e.g., v_LongJump_g01_c01.avi):")

# Main app logic
if st.button("Submit"):
    if file_name:
        # Fetch UCF video
        video_path = fetch_ucf_video(file_name)

        # Display the selected video
        st.video(video_path)

        # Load and process the video
        sample_video = load_video(video_path)
        to_gif(sample_video)

        # Display the processed video as a gif
        st.image('./animation.gif')

        # Make predictions
        predict(sample_video)
    else:
        st.warning("Please enter a video file name.")

st.success("SUCCESSFULLY COMPLETED!")
