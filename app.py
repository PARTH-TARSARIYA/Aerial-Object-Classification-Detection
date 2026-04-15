import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os
from PIL import Image
import time

st.set_page_config(page_title='Object Detection System', layout='wide')

st.title('Aerial Object Detection (Bird / Drone)')

@st.cache_resource
def load_model():
    model = YOLO('model/best.pt')
    return model

model = load_model()

# Image prediction
def predict_image(file):
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')

    try:
        tfile.write(file.read())
        tfile.close()

        results = model.predict(source=tfile.name, conf=0.5)
        output = results[0].plot()

        st.image(output, channels="BGR", use_container_width=True)

        st.success("Image processed!")

    finally:
        time.sleep(1)
        if os.path.exists(tfile.name):
            os.remove(tfile.name)


# Video
def predict_video(file):
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')

    try:
        tfile.write(file.read())
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)

        frame_placeholder = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=0.5)
            output = results[0].plot()

            frame_placeholder.image(output, channels="BGR", use_container_width=True)

        cap.release()
        st.success("Prediction Completed!")

    finally:
        time.sleep(1)
        if os.path.exists(tfile.name):
            os.remove(tfile.name)


choise = st.radio('Choose input type to detect drone or bird!', ['image', 'video', 'live camera'])

if choise == 'image':
    image = st.file_uploader('Upload an image', type = ['jpg', 'jpeg', 'png'])
    if image is not None:
        predict_image(image)


elif choise == 'video':
    video = st.file_uploader('Upload an video', type = ['mp4', 'mov', 'avi'])
    if video is not None:
        predict_video(video)

else:
    start = st.button("Start Camera")

    if start:
        cap = cv2.VideoCapture(0)

        cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Detection", 1200, 800)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=0.5)
            output = results[0].plot()

            cv2.imshow("Detection", output)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

            if cv2.getWindowProperty("Detection", cv2.WND_PROP_VISIBLE) < 1:
                break

        cap.release()
        cv2.destroyAllWindows()
