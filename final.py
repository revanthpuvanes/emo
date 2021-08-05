import av
import cv2
# import matplotlib.pyplot as plt
import numpy as np
# import pydub
import streamlit as st
from keras.models import load_model
from PIL import Image

from streamlit_webrtc import (
    AudioProcessorBase,
    ClientSettings,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
    VideoTransformerBase
)

WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={
        "video": True
        # "audio": True,
    },
)

face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')


class EmoTransformer(VideoTransformerBase):
    global model
    model = load_model('./models/_mini_XCEPTION.86-0.65.hdf5',compile=False)
    def __init__(self):
        self.threshold1 = 100
        self.threshold2 = 200
        self.i = 0
        # self.result_queue = queue.Queue()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # model = mini_XCEPTION()
        # emotion_model = mini_XCEPTION()
        # model = load_model('./models/_mini_XCEPTION.86-0.65.hdf5',compile=False)
        emotions = ['ANGRY', 'DISGUST', 'FEAR', 'HAPPY', 'SAD', 'SURPRISE', 'NEUTRAL']
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # gray = np.array(gray)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        i = self.i+1
        for (x,y,w,h) in faces:

            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2) #draw rectangle to main image
            
            #extract detected face
            detected_face = gray[int(y):int(y+h), int(x):int(x+w)] #crop detected face
            #vgg-face expects inputs (224, 224, 3)
            detected_face = cv2.resize(detected_face, (48, 48))

            img_pixels = np.array(detected_face,dtype=np.float)
            img_pixels = np.expand_dims(img_pixels, axis = 0)
            img_pixels /= 255
            
            predictions = model.predict(img_pixels)
            emo_index = emotions[np.argmax(predictions[0])]
            cv2.putText(img, emo_index,(x +10, y -10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        return av.VideoFrame.from_ndarray(img,format="bgr24")


def main():
    selected_box = st.sidebar.selectbox(
    'Pick Something Fun',
    ('Welcome','Emotion Detection Image','Live Emotion Detection (webcam)')
    )
    
    if selected_box == 'Welcome':
        welcome() 
    if selected_box == 'Emotion Detection Image':
        emotion_detection()
    if selected_box == 'Live Emotion Detection (webcam)':
        live_emotion_detection()

def welcome():
    
    st.title('IMAGE PROCESSING WITH ML')
    st.title('by .......REVANTH')
    st.write("Go to the left sidebar to explore")

def live_emotion_detection():
    st.title("Emotion Detection in Real-Time")
    webrtc_streamer(key="emotion", 
                    video_transformer_factory=EmoTransformer,
                    mode=WebRtcMode.SENDRECV,
                    client_settings=WEBRTC_CLIENT_SETTINGS)


def emo_detect(image):
    # model = mini_XCEPTION()
    # emotion_model = mini_XCEPTION()
    # model.load_weights('./models/_mini_XCEPTION.86-0.65.hdf5')
    model = load_model('./models/_mini_XCEPTION.86-0.65.hdf5',compile=False)
    emotions = ['ANGRY', 'DISGUST', 'FEAR', 'HAPPY', 'SAD', 'SURPRISE', 'NEUTRAL']
    face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
    image = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # gray = np.array(gray)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h )in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        #extract detected face
        detected_face = gray[int(y):int(y+h), int(x):int(x+w)] #crop detected face
          #vgg-face expects inputs (224, 224, 3)
        detected_face = cv2.resize(detected_face, (48, 48))

        img_pixels = np.array(detected_face,dtype=np.float)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)
        emo_index = emotions[np.argmax(predictions[0])]
        # emotion = emotions(emo_index)

        cv2.putText(image, emo_index,(x+0,y+0), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    return image, faces


def emotion_detection():
    st.title("Emotion Detection")
    image_file = st.file_uploader("Upload image", type=['jpeg', 'png', 'jpg', 'webp', 'jfif'])
    if image_file is not None:

    	image = Image.open(image_file)

    	if st.button("Process"):
            result_img, result_faces = emo_detect(image=image)
            st.image(result_img,use_column_width='auto')
            # st.success("Age {}\n".format(int(result_age)))
            st.success("Found {} faces\n".format(len(result_faces)))
            
    if st.button('See Original Image'):
        if image_file is not None:
            original = Image.open(image_file)
            st.image(original, use_column_width='auto')
        else:
            st.write("Please upload any image using browse files ")


if __name__ == "__main__":
    main()