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

# ---------------- IMAGE ----------------
def predict_image(file):
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')

    try:
        tfile.write(file.read())
        tfile.close()

        results = model.predict(source=tfile.name, conf=0.5)
        output = results[0].plot()

        # ✅ Show in Streamlit instead of cv2 window
        st.image(output, channels="BGR", use_container_width=True)

        st.success("Image processed!")

    finally:
        time.sleep(1)
        if os.path.exists(tfile.name):
            os.remove(tfile.name)


# ---------------- VIDEO ----------------
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

            # ✅ Streamlit display
            frame_placeholder.image(output, channels="BGR", use_container_width=True)

        cap.release()
        st.success("Prediction Completed!")

    finally:
        time.sleep(1)
        if os.path.exists(tfile.name):
            os.remove(tfile.name)


# ---------------- CAMERA ----------------
def predict_camera():
    img_file_buffer = st.camera_input("Take a picture")

    if img_file_buffer is not None:
        image = Image.open(img_file_buffer)
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        results = model(frame, conf=0.5)
        output = results[0].plot()

        st.image(output, channels="BGR", use_container_width=True)
        st.success("Camera prediction done!")


# ---------------- UI ----------------
choice = st.radio('Choose input type to detect drone or bird!', ['image', 'video', 'live camera'])

if choice == 'image':
    image = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])
    if image is not None:
        predict_image(image)

elif choice == 'video':
    video = st.file_uploader('Upload a video', type=['mp4', 'mov', 'avi'])
    if video is not None:
        predict_video(video)

else:
    predict_camera()
