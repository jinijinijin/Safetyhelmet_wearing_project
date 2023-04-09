# import necessary packages
import cvlib as cv
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import os
from gtts import gTTS
from playsound import playsound

__author__ = 'info-lab'

import time

# convert file path from one encoding to another
file_path = 'model.h5'
encoded_file_path = file_path.encode('cp949') # replace 'cp949' with the encoding of your file path
decoded_file_path = encoded_file_path.decode('utf-8')
model = load_model(decoded_file_path)



from PIL import ImageFont, ImageDraw, Image

tts1 = gTTS(
    text='헬멧을 착용해주세요',
    lang='ko', slow=False
)
tts1.save('audio2.mp3')

model = load_model('model.h5')
model.summary()

# open webcam
webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Could not open webcam")
    exit()

# loop through frames
while webcam.isOpened():

    # read frame from webcam
    status, frame = webcam.read()

    if not status:
        print("Could not read frame")
        exit()

    # apply face detection
    face, confidence = cv.detect_face(frame)

    # loop through detected faces
    for idx, f in enumerate(face):

        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        if 0 <= startX <= frame.shape[1] and 0 <= endX <= frame.shape[1] and 0 <= startY <= frame.shape[
            0] and 0 <= endY <= frame.shape[0]:

            face_region = frame[startY:endY, startX:endX]

            face_region1 = cv2.resize(face_region, (256, 256), interpolation=cv2.INTER_AREA)

            x = img_to_array(face_region1)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            prediction = model.predict(x)



            if prediction < 0.5:  # 헬멧 미착용으로 판별되면,

                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                Y = startY - 10 if startY - 10 > 10 else startY + 10
                text = "No helmet ({:.2f}%)".format((1 - prediction[0][0]) * 100)
                cv2.putText(frame, text, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                os.system("audio2.mp3")
                time.sleep(3)


            else:  # 헬멧 착용으로 판별되면

                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                Y = startY - 10 if startY - 10 > 10 else startY + 10
                text = "helmet ({:.2f}%)".format(prediction[0][0] * 100)
                cv2.putText(frame, text, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                #os.system('audio1.mp3')
                time.sleep(3)


    # display output
    cv2.imshow("helmet nohelmet classify", frame)

    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release resources
webcam.release()
cv2.destroyAllWindows()