import face_recognition
import argparse
from utils import *
import pandas as pd


class VideoCapture(object):
    def __init__(self):
        #####################################################################
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--model-cfg', type=str, default='./cfg/yolov3-face.cfg',
                            help='path to config file')
        self.parser.add_argument('--model-weights', type=str,
                            default='./model-weights/yolov3-wider_16000.weights',
                            help='path to weights of model')
        self.parser.add_argument('--image', type=str, default='',
                            help='path to image file')
        self.parser.add_argument('--video', type=str, default='',
                            help='path to video file')
        self.parser.add_argument('--src', type=int, default=0,
                            help='source of the camera')
        self.parser.add_argument('--output-dir', type=str, default='outputs/',
                            help='path to the output directory')
        self.args = self.parser.parse_args()

        self.cap = cv2.VideoCapture('./unknown_clips/Trump_Accuse_Obama.mp4')
        #cap.set(cv2.CAP_PROP_FPS, 30)

        #####################################################################
        self.net = cv2.dnn.readNetFromDarknet(self.args.model_cfg, self.args.model_weights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        ### read existing face_encodings from excel file ###
        df = pd.read_excel('face_encoding_table' + '.xlsx', index_col=None)
        face_encoding_dim = 128
        self.obama_face_encoding = []
        for i in range(face_encoding_dim):
            self.obama_face_encoding.append(df['face_encoding_'+str(i)][0])

        self.obama_face_encoding = np.asarray((self.obama_face_encoding))

        # Create arrays of known face encodings and their names
        self.known_face_encodings = [
            self.obama_face_encoding,
            # self.trump_face_encoding,
            #self.edward_face_encoding,
            #self.ann_face_encoding
        ]

        self.known_face_names = [
            "Barack Obama",
            # "Donald Trump",
            #"Edward",
            #"Ann"
        ]

        # Initialize some variables
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.process_this_frame = True
        self.first_loop = True
        self.qualified_augmentation = False
        self.previous_matched_name_and_location = {}
        self.previous_matched_name_and_encodings = {}

        #self.cap = cv2.VideoCapture(0)
        self.video = []
        self.frame_count = 0
        self.change = 0
        self.body_temperature = 36.0
        print("=====init finished=======")

    def get_frame(self):
        success, frame = self.cap.read()
        frame = cv2.resize(frame, (640, 360))
        if success:
            # Create a 4D blob from a frame.
            blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),
                                         [0, 0, 0], 1, crop=False)
            # Sets the input to the network
            self.net.setInput(blob)
            # Runs the forward pass to get output of the output layers
            outs = self.net.forward(get_outputs_names(self.net))
            # Remove the bounding boxes with low confidence
            faces = post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)
            print('[i] ==> # detected faces: {}'.format(len(faces)))
            ################################################################
            new_face_locations = faces
            face_landmarks_list = face_recognition.face_landmarks(frame, face_locations=new_face_locations)
            face_encodings = face_recognition.face_encodings(frame, new_face_locations)
            face_names = []
            for face_encoding, new_face_location in zip(face_encodings, new_face_locations):
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.5)
                name = "Unknown"
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding) # face_distance < 0.4 seems the 2 faces are similar
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
                else:
                    pass
                face_names.append(name)
            print("face_names = ", face_names)
            # Display the results
            rect_color = (0, 0, 255)
            font = cv2.FONT_HERSHEY_DUPLEX
            for (top, right, bottom, left), name in zip(new_face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                cv2.rectangle(frame, (left, top), (right, bottom), rect_color, 2)
                cv2.rectangle(frame, (left, bottom + 10), (right, bottom), rect_color, cv2.FILLED)
                cv2.putText(frame, name, (left + 10, bottom + 10), font, 0.5, (255, 255, 255), 1)

        return frame, len(faces), face_names

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()

