import cv2
import dlib
import numpy as np
from tensorflow.keras.models import load_model
import os,os.path
from datetime import datetime
import math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#Load the trained tensorflow model...
loaded_model = load_model("Models/4Emotions.h5")

def predictEmotion(img):
    #This function returns the emotion based on face passed in img variable...

    #Resizing the image to the size that we specified while training the model...
    roi = cv2.resize(img, (48, 48))
    x = (roi[np.newaxis, :, :, np.newaxis])
    preds = loaded_model.predict(x)

    #Returns an integer ...
    return (np.argmax(preds))

def overlay_transparent(background, overlay, x, y):
    #This function is used to mask the transperent layer form the PNG image...

    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype = overlay.dtype) * 255
            ],
            axis = 2,
        )

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_image

    #The function returns the part of frame with emotion-Mask on top of it...
    return background

def setIcon(frame):
    #This function sets the mask icon on the top of the frame, it also shows the number of available masks...

    mask1 = cv2.imread("Masks/Mask1/icon.jpg")
    mask1 = cv2.resize(mask1, (30,30), interpolation = cv2.INTER_AREA)
    mask2 = cv2.imread("Masks/Mask2/icon.jpg")
    mask2 = cv2.resize(mask2, (30,30), interpolation = cv2.INTER_AREA)
    mask3 = cv2.imread("Masks/Mask3/icon.jpg")
    mask3 = cv2.resize(mask3, (30,30), interpolation = cv2.INTER_AREA)
    mask4 = cv2.imread("Masks/Mask4/icon.jpg")
    mask4 = cv2.resize(mask4, (30,30), interpolation = cv2.INTER_AREA)
    mask5 = cv2.imread("Masks/Mask5/icon.jpg")
    mask5 = cv2.resize(mask5, (30,30), interpolation = cv2.INTER_AREA)
    
    #Location of each mask icon...
    #MASK1 cv2.rectangle(frame, (15,15), (45,45), (0,255,0), 1)
    #MASK2 cv2.rectangle(frame, (55,15), (85,45), (0,255,0), 1)
    #MASK1 cv2.rectangle(frame, (95,15), (125,45), (0,255,0), 1)
    #MASK2 cv2.rectangle(frame, (135,15), (165,45), (0,255,0), 1)
    #MASK2 cv2.rectangle(frame, (175,15), (205,45), (0,255,0), 1)

    roi1 = frame[15:45,15:45]
    img = cv2.addWeighted(mask1, 1, roi1, 1, 0)
    frame[15:45,15:45] = img

    roi2 = frame[15:45,55:85]
    img = cv2.addWeighted(mask2, 1, roi2, 1, 0)
    frame[15:45,55:85] = img

    roi3 = frame[15:45,95:125]
    img = cv2.addWeighted(mask3, 1, roi3, 1, 0)
    frame[15:45,95:125] = img

    roi4 = frame[15:45,135:165]
    img = cv2.addWeighted(mask4, 1, roi4, 1, 0)
    frame[15:45,135:165] = img

    roi5 = frame[15:45,175:205]
    img = cv2.addWeighted(mask5, 1, roi5, 1, 0)
    frame[15:45,175:205] = img

    return frame

def Mask1(landmarks_points,leftEye,rightEye,mouth):
    #The default values
    mouthExpandValue = 1.0      #To specify how big the mouth image size should be compared to person's mouth size
    eyeExpandValue = 1.0        #To specify how big the eye image size should be compared to person's eye size
    Y_Translaton_eye = 0        #To move the eye position up or down, this is important if eye image is not of standard size...
    Y_Translaton_mouth = 0      #To move the mouth up or down, if image has other elements then mouth then it needs to be configured...

    #The following if-elif block is used to configure the above values for each mask emotions...
    if emotionLabel[emotionIndex] == "Natural":
        mouthExpandValue, eyeExpandValue = 1.2, 2.3
    elif emotionLabel[emotionIndex] == "Happy":
        mouthExpandValue, eyeExpandValue, Y_Translaton_eye = 1.0, 2.1, 6
    elif emotionLabel[emotionIndex] == "Angry":
        mouthExpandValue, eyeExpandValue, Y_Translaton_eye = 1.2, 2.5, 7
    elif emotionLabel[emotionIndex] == "Shock":
        mouthExpandValue, eyeExpandValue, Y_Translaton_mouth = 1.3, 3.2, -6

    #Calculating the dist of end points of a persons eye, and then multiplying it with eyeExpandValue to get desired size...
    eyeDist = int(math.sqrt((landmarks_points[36][0] - landmarks_points[39][0])**2 + (landmarks_points[36][1] - landmarks_points[39][1])**2) * eyeExpandValue)
    if eyeDist%2==1:
        eyeDist+=1      #Getting an even value
    mouthDist = int((landmarks_points[54][0] - landmarks_points[48][0])*mouthExpandValue)
    if mouthDist%2==1:
        mouthDist+=1

    #Resizing the images according  to eyeDist and mouthDist...
    resized_leftEye = cv2.resize(leftEye, (eyeDist,eyeDist), interpolation = cv2.INTER_AREA)
    resized_rightEye = cv2.resize(rightEye, (eyeDist,eyeDist), interpolation = cv2.INTER_AREA)
    resized_mouth = cv2.resize(mouth, (mouthDist,mouthDist), interpolation = cv2.INTER_AREA)

    #The following three lines first take the image of required size from frame...
    #Then pass that to overlay_transparent function to overlap the PNG image on top of it...
    #Then, replacing the actual frame area with that new image...
    roi_LeftEye = frame[landmarks_points[37][1]-eyeDist//2-Y_Translaton_eye:landmarks_points[37][1]+eyeDist//2-Y_Translaton_eye,landmarks_points[37][0]-eyeDist//2:landmarks_points[37][0]+eyeDist//2]
    out = overlay_transparent(roi_LeftEye, resized_leftEye, 0, 0)
    frame[landmarks_points[37][1]-eyeDist//2-Y_Translaton_eye:landmarks_points[37][1]+eyeDist//2-Y_Translaton_eye,landmarks_points[37][0]-eyeDist//2:landmarks_points[37][0]+eyeDist//2] = out
    
    roi_RightEye = frame[landmarks_points[44][1]-eyeDist//2-Y_Translaton_eye:landmarks_points[44][1]+eyeDist//2-Y_Translaton_eye,landmarks_points[44][0]-eyeDist//2:landmarks_points[44][0]+eyeDist//2]
    out = overlay_transparent(roi_RightEye, resized_rightEye, 0, 0)
    frame[landmarks_points[44][1]-eyeDist//2-Y_Translaton_eye:landmarks_points[44][1]+eyeDist//2-Y_Translaton_eye,landmarks_points[44][0]-eyeDist//2:landmarks_points[44][0]+eyeDist//2] = out

    mouthMid = ((landmarks_points[62][0]+landmarks_points[66][0])//2,(landmarks_points[62][1]+landmarks_points[66][1])//2)
    roi_Mouth =  frame[mouthMid[1]-mouthDist//2-Y_Translaton_mouth:mouthMid[1]+mouthDist//2-Y_Translaton_mouth,mouthMid[0]-mouthDist//2:mouthMid[0]+mouthDist//2]
    out = overlay_transparent(roi_Mouth, resized_mouth, 0, 0)
    frame[mouthMid[1]-mouthDist//2-Y_Translaton_mouth:mouthMid[1]+mouthDist//2-Y_Translaton_mouth,mouthMid[0]-mouthDist//2:mouthMid[0]+mouthDist//2] = out

def Mask2(landmarks_points,leftEye,rightEye,mouth):
    mouthExpandValue = 1.0
    eyeExpandValue = 1.0
    Y_Translaton_eye = 0
    Y_Translaton_mouth = 0

    if emotionLabel[emotionIndex] == "Natural":
        mouthExpandValue, eyeExpandValue = 1.9, 2.8
    elif emotionLabel[emotionIndex] == "Happy":
        mouthExpandValue, eyeExpandValue, Y_Translaton_mouth = 1.6, 2.7, -10
    elif emotionLabel[emotionIndex] == "Angry":
        mouthExpandValue, eyeExpandValue, Y_Translaton_mouth = 1.8, 2.4, -4
    elif emotionLabel[emotionIndex] == "Shock":
        mouthExpandValue, eyeExpandValue, Y_Translaton_mouth = 3, 3.1, -5

    eyeDist = int(math.sqrt((landmarks_points[36][0] - landmarks_points[39][0])**2 + (landmarks_points[36][1] - landmarks_points[39][1])**2) * eyeExpandValue)
    if eyeDist%2==1:
        eyeDist+=1
    mouthDist = int((landmarks_points[54][0] - landmarks_points[48][0])*mouthExpandValue)
    if mouthDist%2==1:
        mouthDist+=1

    resized_leftEye = cv2.resize(leftEye, (eyeDist,eyeDist), interpolation = cv2.INTER_AREA)
    resized_rightEye = cv2.resize(rightEye, (eyeDist,eyeDist), interpolation = cv2.INTER_AREA)
    resized_mouth = cv2.resize(mouth, (mouthDist,mouthDist), interpolation = cv2.INTER_AREA)

    roi_LeftEye = frame[landmarks_points[37][1]-eyeDist//2-Y_Translaton_eye:landmarks_points[37][1]+eyeDist//2-Y_Translaton_eye,landmarks_points[37][0]-eyeDist//2:landmarks_points[37][0]+eyeDist//2]
    out = overlay_transparent(roi_LeftEye, resized_leftEye, 0, 0)
    frame[landmarks_points[37][1]-eyeDist//2-Y_Translaton_eye:landmarks_points[37][1]+eyeDist//2-Y_Translaton_eye,landmarks_points[37][0]-eyeDist//2:landmarks_points[37][0]+eyeDist//2] = out
    
    roi_RightEye = frame[landmarks_points[44][1]-eyeDist//2-Y_Translaton_eye:landmarks_points[44][1]+eyeDist//2-Y_Translaton_eye,landmarks_points[44][0]-eyeDist//2:landmarks_points[44][0]+eyeDist//2]
    out = overlay_transparent(roi_RightEye, resized_rightEye, 0, 0)
    frame[landmarks_points[44][1]-eyeDist//2-Y_Translaton_eye:landmarks_points[44][1]+eyeDist//2-Y_Translaton_eye,landmarks_points[44][0]-eyeDist//2:landmarks_points[44][0]+eyeDist//2] = out

    mouthMid = ((landmarks_points[62][0]+landmarks_points[66][0])//2,(landmarks_points[62][1]+landmarks_points[66][1])//2)
    roi_Mouth =  frame[mouthMid[1]-mouthDist//2-Y_Translaton_mouth:mouthMid[1]+mouthDist//2-Y_Translaton_mouth,mouthMid[0]-mouthDist//2:mouthMid[0]+mouthDist//2]
    out = overlay_transparent(roi_Mouth, resized_mouth, 0, 0)
    frame[mouthMid[1]-mouthDist//2-Y_Translaton_mouth:mouthMid[1]+mouthDist//2-Y_Translaton_mouth,mouthMid[0]-mouthDist//2:mouthMid[0]+mouthDist//2] = out

def Mask3(landmarks_points,leftEye,rightEye,mouth):
    mouthExpandValue = 1.0
    eyeExpandValue = 1.0
    Y_Translaton_eye = 0
    Y_Translaton_mouth = 0

    if emotionLabel[emotionIndex] == "Natural":
        mouthExpandValue, eyeExpandValue, Y_Translaton_mouth = 1.2, 2.6, 2
    elif emotionLabel[emotionIndex] == "Happy":
        mouthExpandValue, eyeExpandValue, Y_Translaton_eye = 1.6,3.0,4
    elif emotionLabel[emotionIndex] == "Angry":
        mouthExpandValue, eyeExpandValue, Y_Translaton_mouth = 1.6,2.9,-4
    elif emotionLabel[emotionIndex] == "Shock":
        mouthExpandValue, eyeExpandValue, Y_Translaton_mouth, Y_Translaton_eye = 1.7,3.5,-8,7


    eyeDist = int(math.sqrt((landmarks_points[36][0] - landmarks_points[39][0])**2 + (landmarks_points[36][1] - landmarks_points[39][1])**2) * eyeExpandValue)
    if eyeDist%2==1:
        eyeDist+=1
    mouthDist = int((landmarks_points[54][0] - landmarks_points[48][0])*mouthExpandValue)
    if mouthDist%2==1:
        mouthDist+=1

    resized_leftEye = cv2.resize(leftEye, (eyeDist,eyeDist), interpolation = cv2.INTER_AREA)
    resized_rightEye = cv2.resize(rightEye, (eyeDist,eyeDist), interpolation = cv2.INTER_AREA)
    resized_mouth = cv2.resize(mouth, (mouthDist,mouthDist), interpolation = cv2.INTER_AREA)

    roi_LeftEye = frame[landmarks_points[37][1]-eyeDist//2-Y_Translaton_eye:landmarks_points[37][1]+eyeDist//2-Y_Translaton_eye,landmarks_points[37][0]-eyeDist//2:landmarks_points[37][0]+eyeDist//2]
    out = overlay_transparent(roi_LeftEye, resized_leftEye, 0, 0)
    frame[landmarks_points[37][1]-eyeDist//2-Y_Translaton_eye:landmarks_points[37][1]+eyeDist//2-Y_Translaton_eye,landmarks_points[37][0]-eyeDist//2:landmarks_points[37][0]+eyeDist//2] = out
    
    roi_RightEye = frame[landmarks_points[44][1]-eyeDist//2-Y_Translaton_eye:landmarks_points[44][1]+eyeDist//2-Y_Translaton_eye,landmarks_points[44][0]-eyeDist//2:landmarks_points[44][0]+eyeDist//2]
    out = overlay_transparent(roi_RightEye, resized_rightEye, 0, 0)
    frame[landmarks_points[44][1]-eyeDist//2-Y_Translaton_eye:landmarks_points[44][1]+eyeDist//2-Y_Translaton_eye,landmarks_points[44][0]-eyeDist//2:landmarks_points[44][0]+eyeDist//2] = out

    mouthMid = ((landmarks_points[62][0]+landmarks_points[66][0])//2,(landmarks_points[62][1]+landmarks_points[66][1])//2)
    roi_Mouth =  frame[mouthMid[1]-mouthDist//2-Y_Translaton_mouth:mouthMid[1]+mouthDist//2-Y_Translaton_mouth,mouthMid[0]-mouthDist//2:mouthMid[0]+mouthDist//2]
    out = overlay_transparent(roi_Mouth, resized_mouth, 0, 0)
    frame[mouthMid[1]-mouthDist//2-Y_Translaton_mouth:mouthMid[1]+mouthDist//2-Y_Translaton_mouth,mouthMid[0]-mouthDist//2:mouthMid[0]+mouthDist//2] = out

def Mask4(landmarks_points,leftEye,rightEye,mouth):
    mouthExpandValue = 1.0
    eyeExpandValue = 1.0
    Y_Translaton_eye = 0
    Y_Translaton_mouth = 0

    if emotionLabel[emotionIndex] == "Natural":
        mouthExpandValue, eyeExpandValue, Y_Translaton_mouth = 1.7, 4.8, 20
    elif emotionLabel[emotionIndex] == "Happy":
        mouthExpandValue, eyeExpandValue, Y_Translaton_mouth = 1.7, 4.8, 13
    elif emotionLabel[emotionIndex] == "Angry":
        mouthExpandValue, eyeExpandValue, Y_Translaton_mouth = 1.7, 5.0, 10
    elif emotionLabel[emotionIndex] == "Shock":
        mouthExpandValue, eyeExpandValue, Y_Translaton_mouth, Y_Translaton_eye = 2.9, 5.4, 10, 4


    eyeDist = int(math.sqrt((landmarks_points[36][0] - landmarks_points[39][0])**2 + (landmarks_points[36][1] - landmarks_points[39][1])**2) * eyeExpandValue)
    if eyeDist%2==1:
        eyeDist+=1
    mouthDist = int((landmarks_points[54][0] - landmarks_points[48][0])*mouthExpandValue)
    if mouthDist%2==1:
        mouthDist+=1

    resized_leftEye = cv2.resize(leftEye, (eyeDist,eyeDist), interpolation = cv2.INTER_AREA)
    resized_rightEye = cv2.resize(rightEye, (eyeDist,eyeDist), interpolation = cv2.INTER_AREA)
    resized_mouth = cv2.resize(mouth, (mouthDist,mouthDist), interpolation = cv2.INTER_AREA)

    roi_LeftEye = frame[landmarks_points[37][1]-eyeDist//2-Y_Translaton_eye:landmarks_points[37][1]+eyeDist//2-Y_Translaton_eye,landmarks_points[37][0]-eyeDist//2:landmarks_points[37][0]+eyeDist//2]
    out = overlay_transparent(roi_LeftEye, resized_leftEye, 0, 0)
    frame[landmarks_points[37][1]-eyeDist//2-Y_Translaton_eye:landmarks_points[37][1]+eyeDist//2-Y_Translaton_eye,landmarks_points[37][0]-eyeDist//2:landmarks_points[37][0]+eyeDist//2] = out
    
    roi_RightEye = frame[landmarks_points[44][1]-eyeDist//2-Y_Translaton_eye:landmarks_points[44][1]+eyeDist//2-Y_Translaton_eye,landmarks_points[44][0]-eyeDist//2:landmarks_points[44][0]+eyeDist//2]
    out = overlay_transparent(roi_RightEye, resized_rightEye, 0, 0)
    frame[landmarks_points[44][1]-eyeDist//2-Y_Translaton_eye:landmarks_points[44][1]+eyeDist//2-Y_Translaton_eye,landmarks_points[44][0]-eyeDist//2:landmarks_points[44][0]+eyeDist//2] = out

    mouthMid = ((landmarks_points[62][0]+landmarks_points[66][0])//2,(landmarks_points[62][1]+landmarks_points[66][1])//2)
    roi_Mouth =  frame[mouthMid[1]-mouthDist//2-Y_Translaton_mouth:mouthMid[1]+mouthDist//2-Y_Translaton_mouth,mouthMid[0]-mouthDist//2:mouthMid[0]+mouthDist//2]
    out = overlay_transparent(roi_Mouth, resized_mouth, 0, 0)
    frame[mouthMid[1]-mouthDist//2-Y_Translaton_mouth:mouthMid[1]+mouthDist//2-Y_Translaton_mouth,mouthMid[0]-mouthDist//2:mouthMid[0]+mouthDist//2] = out

def Mask5(landmarks_points,leftEye,rightEye,mouth):
    mouthExpandValue = 1.0
    eyeExpandValue = 1.0
    Y_Translaton_eye = 0
    Y_Translaton_mouth = 0

    if emotionLabel[emotionIndex] == "Natural":
        mouthExpandValue, eyeExpandValue, Y_Translaton_eye = 1, 4.7, 5
    elif emotionLabel[emotionIndex] == "Happy":
        mouthExpandValue, eyeExpandValue, Y_Translaton_eye = 1, 4.7, 3
    elif emotionLabel[emotionIndex] == "Angry":
        mouthExpandValue, eyeExpandValue, Y_Translaton_eye = 1, 4.7, 3
    elif emotionLabel[emotionIndex] == "Shock":
        mouthExpandValue, eyeExpandValue, Y_Translaton_eye = 1.5, 4.7, 4


    eyeDist = int(math.sqrt((landmarks_points[36][0] - landmarks_points[39][0])**2 + (landmarks_points[36][1] - landmarks_points[39][1])**2) * eyeExpandValue)
    if eyeDist%2==1:
        eyeDist+=1
    mouthDist = int((landmarks_points[54][0] - landmarks_points[48][0])*mouthExpandValue)
    if mouthDist%2==1:
        mouthDist+=1

    resized_leftEye = cv2.resize(leftEye, (eyeDist,eyeDist), interpolation = cv2.INTER_AREA)
    resized_rightEye = cv2.resize(rightEye, (eyeDist,eyeDist), interpolation = cv2.INTER_AREA)
    resized_mouth = cv2.resize(mouth, (mouthDist,mouthDist), interpolation = cv2.INTER_AREA)

    roi_LeftEye = frame[landmarks_points[37][1]-eyeDist//2-Y_Translaton_eye:landmarks_points[37][1]+eyeDist//2-Y_Translaton_eye,landmarks_points[37][0]-eyeDist//2:landmarks_points[37][0]+eyeDist//2]
    out = overlay_transparent(roi_LeftEye, resized_leftEye, 0, 0)
    frame[landmarks_points[37][1]-eyeDist//2-Y_Translaton_eye:landmarks_points[37][1]+eyeDist//2-Y_Translaton_eye,landmarks_points[37][0]-eyeDist//2:landmarks_points[37][0]+eyeDist//2] = out
    
    roi_RightEye = frame[landmarks_points[44][1]-eyeDist//2-Y_Translaton_eye:landmarks_points[44][1]+eyeDist//2-Y_Translaton_eye,landmarks_points[44][0]-eyeDist//2:landmarks_points[44][0]+eyeDist//2]
    out = overlay_transparent(roi_RightEye, resized_rightEye, 0, 0)
    frame[landmarks_points[44][1]-eyeDist//2-Y_Translaton_eye:landmarks_points[44][1]+eyeDist//2-Y_Translaton_eye,landmarks_points[44][0]-eyeDist//2:landmarks_points[44][0]+eyeDist//2] = out

    mouthMid = ((landmarks_points[62][0]+landmarks_points[66][0])//2,(landmarks_points[62][1]+landmarks_points[66][1])//2)
    roi_Mouth =  frame[mouthMid[1]-mouthDist//2-Y_Translaton_mouth:mouthMid[1]+mouthDist//2-Y_Translaton_mouth,mouthMid[0]-mouthDist//2:mouthMid[0]+mouthDist//2]
    out = overlay_transparent(roi_Mouth, resized_mouth, 0, 0)
    frame[mouthMid[1]-mouthDist//2-Y_Translaton_mouth:mouthMid[1]+mouthDist//2-Y_Translaton_mouth,mouthMid[0]-mouthDist//2:mouthMid[0]+mouthDist//2] = out

#Using dlib to get face landmarks...
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_81_face_landmarks.dat")

#Using haar cascade file for palm detection...
palm_cascade = cv2.CascadeClassifier("haarcascade_palm.xml")

#Capturing video input from webcam...
cap = cv2.VideoCapture(0)

#Specifing the labels, so when we predict an emotion we can get the label for that integer prediction...
emotionLabel =  ["Angry","Happy","Natural","Shock"]
emotionIndex = 0        #Setting the initial emotion as neutral, later it changes according to persons expression...

maskIndex = 2       #Setting up the first (default) mask when starting the program...
changeMask = 0      #Counter for changing the mask... when a palm is detected, the counter goes up... and when a certain number is hit the maskIndex increases/decreases based on hand you raised...

#Infinately looping over the video captured from webcam...
while True:
    ret, frame   = cap.read()

    #Flipping the video input, for better user-ecperience...
    frame = cv2.flip(frame, 1)
    #Converting to grayscale for prediction...
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    frame = setIcon(frame)

    faces = detector(gray)
    #Faces will contain the location of all detected faces in the frame...
    #Looping over all the faces to set mask over every every face...
    for face in faces:
        #We get the rectangle values in face, so gettig each value individually...
        x_axis, y_axis, w, h = face.left(), face.top(), face.width(), face.height()

        landmarks = predictor(gray, face)    #Prdicting the location of 81 landmark points...
        landmarks_points = []
        for n in range(0, 81):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_points.append((x, y))
            #cv2.circle(frame,(x,y),1,(0,0,128),-1)

        faceROI = gray[y_axis:y_axis+h,x_axis:x_axis+h]

        #Using try-except to avoid errors if face is not detected...
        try:
            predictedEmotion = predictEmotion(faceROI)   #Predicting the expression...
        except:
            predictedEmotion = 0
        emotionIndex = predictedEmotion

        #Loading the appropriate mask from appropriate folder...
        leftEye = cv2.imread(f"Masks/Mask{maskIndex}/{emotionLabel[emotionIndex]}/left_eye.png",-1)
        rightEye = cv2.imread(f"Masks/Mask{maskIndex}/{emotionLabel[emotionIndex]}/right_eye.png",-1)
        mouth = cv2.imread(f"Masks/Mask{maskIndex}/{emotionLabel[emotionIndex]}/mouth.png",-1)

        #Depending on selected mask, We execute the appropriate function...
        #Diffrent functions are required because diffrent masks have diffrent scaling and translation values...
        #Content of each function is same...
        #Splitting them into diffrent functions reduces the complexity and total steps to project mask on the face making the process slight faster...
        if maskIndex==1:
            cv2.rectangle(frame, (15,15), (45,45), (255,255,0), 1)
            Mask1(landmarks_points,leftEye,rightEye,mouth)
        elif maskIndex==2:
            cv2.rectangle(frame, (55,15), (85,45), (255,255,0), 1)
            Mask2(landmarks_points,leftEye,rightEye,mouth)
        elif maskIndex==3:
            cv2.rectangle(frame, (95,15), (125,45), (0,255,0), 1)
            Mask3(landmarks_points,leftEye,rightEye,mouth)
        elif maskIndex==4:
            cv2.rectangle(frame, (135,15), (165,45), (0,255,0), 1)
            Mask4(landmarks_points,leftEye,rightEye,mouth)
        elif maskIndex==5:
            cv2.rectangle(frame, (175,15), (205,45), (0,255,0), 1)
            Mask5(landmarks_points,leftEye,rightEye,mouth)
        else:
            pass

    #Program to detect the palm...
    palm = palm_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
    for (x,y,w,h) in palm:
        #Creating a yellow border to specify mask change process...
        cv2.rectangle(frame, (0,0), (640,480), (0,255,0), 2)

        #Getting the central value of palm box...
        centre_x = x+w // 2
        changeMask += 1
        #Using that value to locate the position of palm in the frame ... left or right
        #Depending on position, mask index increases or decreases, i.e. selecting the mask to left or to right...
        if centre_x > 320 and changeMask > 10:
            changeMask=0
            maskIndex+=1
            if maskIndex > 5:
                maskIndex=1
        elif centre_x < 320 and changeMask > 10:
            changeMask=0
            maskIndex-=1
            if maskIndex < 1:
                maskIndex=5

    cv2.imshow("frame",frame)
    #Checks if user pressed any key...
    key = cv2.waitKey(1)

    #When Pressed "q", the frame closes..
    if key == ord("q"):
        break
    #SPACEBAR to captuer screenshot...
    elif key == ord(" "):
        x=(len([iq for iq in os.scandir(r'C:\Codes\Python\Expression Recognition\Saved Images')]))
        cv2.imwrite('Saved Images/Screenshot_'+str(x)+'.jpg',frame)
        img = np.zeros([100,100,3],dtype=np.uint8)
        cv2.imshow("frame",img)

cap.release()
cv2.destroyAllWindows()