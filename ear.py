import numpy as np
import cv2
left_ear = cv2.CascadeClassifier('haarcascade_mcs_leftear.xml')
right_ear = cv2.CascadeClassifier('haarcascade_mcs_rightear.xml')
cap = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

while True:
	_, frame = cap.read()

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.equalizeHist(gray)
	left = left_ear.detectMultiScale(gray, 1.3, 5)
	right = right_ear.detectMultiScale(gray, 1.3, 5)
	
	for (x,y,w,h) in left:
	    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
	
	for (x,y,w,h) in right:
	    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)    

	cv2.imshow('img', frame)
	out.write(frame)
	if cv2.waitKey(10) & 0xFF == ord('q'):
		break

# cv2.waitKey(0)
cv2.destroyAllWindows()
cap.release()