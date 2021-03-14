import time
import cv2
import os

# load OpenCV's Haar cascade for face detection from disk
detector = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
 
# initialize the video stream, allow the camera sensor to warm up,
# and initialize the total number of example faces written to disk
# thus far
vs = cv2.VideoCapture(0)
name = input("Enter you name: ")
# vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)
total = 0

namePath = "Face_Database/"+name+".jpg"

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream, clone it, (just
	# in case we want to write it to disk), and then resize the frame
	# so we can apply face detection faster
	ret, frame = vs.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# detect faces in the grayscale frame
	rects = detector.detectMultiScale(
		gray, scaleFactor=1.1, 
		minNeighbors=5, minSize=(30, 30))
 
	# loop over the face detections and draw them on the frame
	for (x, y, w, h) in rects:
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `k` key was pressed, write the *original* frame to disk
	# so we can later process it and use it for face recognition
	if key == ord("k"):
		cv2.imwrite(namePath, frame)
		break
 
	# if the `q` key was pressed, break from the loop
	elif key == ord("q"):
		break
# print the total faces saved and do a bit of cleanup
vs.release()		
cv2.destroyAllWindows()
