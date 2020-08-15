import face_recognition
import glob
import numpy as np
import pandas as pd


def face_encoding_mean(file_dir):
    face_encodings = []
    for filename in glob.glob(file_dir):
        print("filename = ", filename)
        known_obama_image = face_recognition.load_image_file(filename)
        # make sure detect at least 1 face in the frame
        if len(face_recognition.face_encodings(known_obama_image)) > 0:
            face_encoding = face_recognition.face_encodings(known_obama_image)[0]
            face_encodings.append(face_encoding)

    face_encoding_avg = np.mean(face_encodings, 0)
    return face_encoding_avg


def face_encoding_main(imageFilePath, faceName):
    obama_face_encoding = face_encoding_mean('known_people/obama/*')
    face_encoding_len = len(obama_face_encoding) # 128
    face_encoding_table = {}
    face_encoding_table['name'] = faceName
    for i in range(face_encoding_len):
        face_encoding_table['face_encoding_' + str(i)] = obama_face_encoding[i]
    face_encoding_table_df = pd.DataFrame(face_encoding_table, index=[0])

    ### store new face encodings to excel file ###
    face_encoding_table_df.to_excel('face_encoding_table' + '.xlsx')


if __name__ == '__main__':
    imageFilePath = './known_people/obama/*'
    faceName = 'Obama'
    face_encoding_main(imageFilePath, faceName)
