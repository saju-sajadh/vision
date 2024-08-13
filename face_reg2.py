import os
import sys
import numpy as np
import math
import face_recognition
import cv2
import requests
from gtts import gTTS
import pygame
import io

def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val +((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'
 
class FaceRecognition:
    face_locations = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True
    current_spoken_name = None  # To keep track of the currently spoken name

    def __init__(self):
        self.encode_faces()
        pygame.mixer.init()  # Initialize the pygame mixer for audio playback 

    def encode_faces(self):
        for image in os.listdir('faces'):
            face_image = face_recognition.load_image_file(f'faces/{image}')
            face_encoding = face_recognition.face_encodings(face_image)[0]

            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(image)

    def speak(self, text):
        tts = gTTS(text=text, lang='en')
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        pygame.mixer.music.load(fp, 'mp3')
        pygame.mixer.music.play()

    def run_recognition(self):
        mobile_camera_url = 'https://192.168.1.38:8080/shot.jpg'
        video_capture = cv2.VideoCapture(mobile_camera_url)

        while True:
            raw = requests.get(mobile_camera_url, verify=False)
            img = np.array(bytearray(raw.content), dtype=np.uint8)
            frame = cv2.imdecode(img, -1)

            frame = cv2.flip(frame, 1)

            if self.process_current_frame:
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

                self.face_locations = face_recognition.face_locations(rgb_small_frame)
                self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

                self.face_names = []
                for face_encoding in self.face_encodings:
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = 'UnKnown'
                    confidence = 'Unknown'

                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)

                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = face_confidence(face_distances[best_match_index])

                    name_part = name.split(".")[0]
                    self.face_names.append(f'{name_part} ({confidence})')

                    # Speak the name of the recognized face only if it's different from the currently spoken name
                    if name_part != self.current_spoken_name:
                        if name_part == 'Unknown':
                            self.speak('Unknown person approching')
                        else:
                            self.speak(f'{name_part} Approching')
                            self.current_spoken_name = name_part

            self.process_current_frame = not self.process_current_frame

            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), -1)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

            cv2.imshow('face Recognition', frame)
            cv2.setWindowProperty('face Recognition', cv2.WND_PROP_TOPMOST, 1)

            if cv2.waitKey(1) == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()

fr = FaceRecognition()
fr.run_recognition()
