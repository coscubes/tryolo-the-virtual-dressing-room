#Import required modules
import cv2
import dlib
import math
import numpy as np
import csv
import pickle
'''
def right_ear(a, b):
    m = 
    n = 

def left_ear(a, b):
''' 

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

#Set up some required objects
video_capture = cv2.VideoCapture(0) #Webcam object
detector = dlib.get_frontal_face_detector() #Face detector
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #Landmark identifier. Set the filename to whatever you named the downloaded file

def get_landmarks(image):
    detections = detector(image, 1)
    for k,d in enumerate(detections): #For all detected face instances individually
        shape = predictor(image, d) #Draw Facial Landmarks with the predictor class
        xlist = []
        ylist = []
        for i in range(1,68): #Store X and Y coordinates in two lists
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))
            
        xmean = np.mean(xlist) #Find both coordinates of centre of gravity
        ymean = np.mean(ylist)
        xcentral = [(x-xmean) for x in xlist] #Calculate distance centre <-> other points in both axes
        ycentral = [(y-ymean) for y in ylist]
        
        if xlist[26] == xlist[29]: #If x-coordinates of the set are the same, the angle is 0, catch to prevent 'divide by 0' error in function
            anglenose = 0
        else:
            anglenose = int(math.atan((ylist[26]-ylist[29])/(xlist[26]-xlist[29]))*180/math.pi) #point 29 is the tip of the nose, point 26 is the top of the nose brigde

        if anglenose < 0: #Get offset by finding how the nose brigde should be rotated to become perpendicular to the horizontal plane
            anglenose += 90
        else:
            anglenose -= 90

        landmarks_vectorised = []
        for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
            landmarks_vectorised.append(x) #Add the coordinates relative to the centre of gravity
            landmarks_vectorised.append(y)
        #Get the euclidean distance between each point and the centre point (the vector length)
            meannp = np.asarray((ymean,xmean))
            coornp = np.asarray((z,w))
            dist = np.linalg.norm(coornp-meannp)
            landmarks_vectorised.append(dist)

            #Get the angle the vector describes relative to the image, corrected for the offset that the nosebrigde has when the face is not perfectly horizontal
            anglerelative = (math.atan((z-ymean)/(w-xmean))*180/math.pi) - anglenose
            landmarks_vectorised.append(anglerelative)
            
    if len(detections) < 1: 
        landmarks_vectorised = "error" #If no face is detected set the data to value "error" to catch detection errors
    
    return landmarks_vectorised

while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_image = clahe.apply(gray)
    a = get_landmarks(gray)
    detections = detector(clahe_image, 1) #Detect the faces in the image

    for k,d in enumerate(detections): #For each detected face
        
        shape = predictor(clahe_image, d) #Get coordinates
        #for i in range(1,68): #There are 68 landmark points on each face
        #    cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 1, (0,0,255), thickness=2) #For each point, draw a red circle with thickness2 on the original frame
        cv2.circle(frame, (shape.part(33).x, shape.part(33).y), 1, (0,255,0), thickness=2)
        cv2.circle(frame, (shape.part(2).x, shape.part(2).y), 1, (0,255,0), thickness=2)
        cv2.circle(frame, (shape.part(14).x, shape.part(14).y), 1, (0,255,0), thickness=2)


    cv2.imshow("image", frame) #Display the frame
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): #Exit program when the user presses 'q'
        break

video_capture.release()
out.release()
cv2.destroyAllWindows()
