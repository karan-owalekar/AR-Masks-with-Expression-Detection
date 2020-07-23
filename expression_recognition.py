import cv2
import dlib
import numpy as np
from imutils import face_utils
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model
import os,os.path
from datetime import datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

loaded_model = load_model("4Emotions.h5")

def predictEmotion(img):
    roi = cv2.resize(img, (48, 48))
    x = (roi[np.newaxis, :, :, np.newaxis])
    preds = loaded_model.predict(x)
    return (np.argmax(preds))

def overlay_transparent(background, overlay, x, y):

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

    return background

def setIcon(frame):
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
    mouthExpandValue = 1.0
    eyeExpandValue = 1.0
    Y_Translaton_eye = 0
    Y_Translaton_mouth = 0

    if emotionLabel[emotionIndex] == "Natural":
        mouthExpandValue,eyeExpandValue = 1.2,1.8
    elif emotionLabel[emotionIndex] == "Happy":
        mouthExpandValue,eyeExpandValue,Y_Translaton_eye = 1.0,1.5,6
    elif emotionLabel[emotionIndex] == "Angry":
        mouthExpandValue,eyeExpandValue,Y_Translaton_eye = 1.2,1.6,7
    elif emotionLabel[emotionIndex] == "Shock":
        mouthExpandValue,eyeExpandValue,Y_Translaton_mouth = 1.3,2.3,-6


    eyeDist = int((landmarks_points[35][0] - landmarks_points[38][0])*eyeExpandValue)
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

def Mask2(landmarks_points,leftEye,rightEye,mouth):
    mouthExpandValue = 1.0
    eyeExpandValue = 1.0
    Y_Translaton_eye = 0
    Y_Translaton_mouth = 0

    if emotionLabel[emotionIndex] == "Natural":
        mouthExpandValue,eyeExpandValue = 1.9,2.0
    elif emotionLabel[emotionIndex] == "Happy":
        mouthExpandValue,eyeExpandValue,Y_Translaton_mouth = 1.6,1.9,-10
    elif emotionLabel[emotionIndex] == "Angry":
        mouthExpandValue,eyeExpandValue,Y_Translaton_mouth = 1.8,1.5,-4
    elif emotionLabel[emotionIndex] == "Shock":
        mouthExpandValue,eyeExpandValue,Y_Translaton_mouth = 3,2.3,-5


    eyeDist = int((landmarks_points[35][0] - landmarks_points[38][0])*eyeExpandValue)
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
        mouthExpandValue,eyeExpandValue,Y_Translaton_mouth = 1.2,1.9,2
    elif emotionLabel[emotionIndex] == "Happy":
        mouthExpandValue,eyeExpandValue,Y_Translaton_eye = 1.6,2.0,4
    elif emotionLabel[emotionIndex] == "Angry":
        mouthExpandValue,eyeExpandValue,Y_Translaton_mouth = 1.6,1.6,-4
    elif emotionLabel[emotionIndex] == "Shock":
        mouthExpandValue,eyeExpandValue,Y_Translaton_mouth,Y_Translaton_eye = 1.7,2.5,-8,7


    eyeDist = int((landmarks_points[35][0] - landmarks_points[38][0])*eyeExpandValue)
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
        mouthExpandValue,eyeExpandValue,Y_Translaton_mouth = 1.7,3.5,20
    elif emotionLabel[emotionIndex] == "Happy":
        mouthExpandValue,eyeExpandValue,Y_Translaton_mouth = 1.7,3.5,13
    elif emotionLabel[emotionIndex] == "Angry":
        mouthExpandValue,eyeExpandValue,Y_Translaton_mouth = 1.7,3.5,10
    elif emotionLabel[emotionIndex] == "Shock":
        mouthExpandValue,eyeExpandValue,Y_Translaton_mouth,Y_Translaton_eye = 2.9,3.8,10,4


    eyeDist = int((landmarks_points[35][0] - landmarks_points[38][0])*eyeExpandValue)
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
        mouthExpandValue,eyeExpandValue,Y_Translaton_eye = 1,3.5,5
    elif emotionLabel[emotionIndex] == "Happy":
        mouthExpandValue,eyeExpandValue,Y_Translaton_eye = 1,3.5,3
    elif emotionLabel[emotionIndex] == "Angry":
        mouthExpandValue,eyeExpandValue,Y_Translaton_eye = 1,3.5,3
    elif emotionLabel[emotionIndex] == "Shock":
        mouthExpandValue,eyeExpandValue,Y_Translaton_eye = 1.5,3.5,4


    eyeDist = int((landmarks_points[35][0] - landmarks_points[38][0])*eyeExpandValue)
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

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_81_face_landmarks.dat")

palm_cascade = cv2.CascadeClassifier("haarcascade_palm.xml")

cap = cv2.VideoCapture(0)

emotionLabel =  ["Angry","Happy","Natural","Shock"]
emotionIndex = 0

maskIndex = 2
changeMask = 0

while True:
    ret, frame   = cap.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    frame = setIcon(frame)

    faces = detector(gray)
    for face in faces:
        x_axis, y_axis, w, h = face.left(), face.top(), face.width(), face.height()
        landmarks = predictor(gray, face)
        landmarks_points = []
        for n in range(0, 81):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_points.append((x, y))
            #cv2.circle(frame,(x,y),1,(0,0,128),-1)

        faceROI = gray[y_axis:y_axis+h,x_axis:x_axis+h]
        try:
            predictedEmotion = predictEmotion(faceROI)
        except:
            predictedEmotion = 0
        emotionIndex = predictedEmotion

        leftEye = cv2.imread(f"Masks/Mask{maskIndex}/{emotionLabel[emotionIndex]}/left_eye.png",-1)
        rightEye = cv2.imread(f"Masks/Mask{maskIndex}/{emotionLabel[emotionIndex]}/right_eye.png",-1)
        mouth = cv2.imread(f"Masks/Mask{maskIndex}/{emotionLabel[emotionIndex]}/mouth.png",-1)

        
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

    palm = palm_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
    for (x,y,w,h) in palm:
        cv2.rectangle(frame, (0,0), (640,480), (0,255,0), 2)

        centre_x = x+w // 2
        changeMask += 1
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
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    elif key == ord(" "):
        x=(len([iq for iq in os.scandir(r'C:\Codes\Python\Expression Recognition\Saved Images')]))
        cv2.imwrite('Saved Images/Screenshot_'+str(x)+'.jpg',frame)
        img = np.zeros([100,100,3],dtype=np.uint8)
        cv2.imshow("frame",img)

cap.release()
cv2.destroyAllWindows()