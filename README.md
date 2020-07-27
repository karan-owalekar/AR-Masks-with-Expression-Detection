# AR-Masks: with Expression Detection
![Expression_git](https://user-images.githubusercontent.com/68480967/88527847-7131b980-d01b-11ea-8921-cef3d1d83b43.png)
expression_recognition.py - Main file

      You just need to run this file.
      
Masks folder - This contains all the images of masks, this folder must be present along with expression_recognition.py file.

Models folder - This contains the trained deep-learning model. 
      The required model for this project is 4Emotions.h5
      Other models are trained over 7 diffrent emotions.
      
haarcascade_palm.xml - This haar-cascade file is used to detect palms in the frame.

shape_predictor_81_face_landmarks.dat - This file is used to project the 81 landmark points on the face.

### Here is a small preview of the project...
![ExpressinonDetection](https://user-images.githubusercontent.com/68480967/88522162-99b5b580-d013-11ea-9fcc-b83217ea0354.gif)

> First the program detects user's face.

> Once the face is detected, It uses that as an input for the trained Deep-learning model.

> That model determines the expression on the face.

> Once the expression is detected, the program then applies selected MASK with detected expreddion on the face.

> As the expression changes, the mask expression also changes.

> Using my hands, I can switch between diffrent mask designs.

> The current mask design is shown in the top of the video frames.

> Using SPACEBAR I can take "screenshots" of the image, which will be stored in "Saved Image" folder.
