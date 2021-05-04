# YoloFaceRecognition_OnCrowManagement

According to the COVID-19, this project try to develop a crowd management system which involves face detection, face recognition, body temperature detection, and related automatic sterilization sensors and hardwares and their stock management, then we stream the analysis results to a website. By doing so, the cost on human resources could be reduced and the prevention of COVID-19 could also be efficient. Here shows you the algorithm part, and some of them are still under development, it will be updated if the new part is developed.

## Steps

Since the yolo model(yolov3-wider_16000.weights) used in this project is too large to upload, you can try to download from https://drive.google.com/drive/folders/1oj9p04mPjbbCbq1qSK8ChMjOhMLMpk42?usp=sharing

1. Create a "Known_people", and put some images with faces.
2. Run "Face_encoding.py", then a excel file will be produced, which stores the features of the images that you used in step 1
3. Run "main_web.py" to start a flusk based web. (For now, only the "Video Streaming" work, please try on this) Then you will see the streamimg result with face detection and ecognition, you can also feed a clip video by modify the self.cap = cv2.VideoCapture('YOURPATH/Video.mp4') in "Face_detectionAndRecognition.py", the following shows you the example.

![image](https://github.com/edwardchang0112/YoloFaceRecognition_OnCrowManagement/blob/master/Demo01.png)

## Real case

We tried to combine our multispectral camera to make this application can also work in a dark environment, the following figure shows you a real example that we captured the streaming frame of a street scene in dark environment.

![image](https://github.com/edwardchang0112/YoloFaceRecognition_OnCrowManagement/blob/master/Demo02.png)

## Future work

The backend system, like Database, will be involved, and distributed computing structure as well

### All of the materials in this project provide you a basic example structure, all data/path used in this project need to be changed to fit your applications.
