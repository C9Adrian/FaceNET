import smtplib 
from email.mime.multipart import MIMEMultipart 
from email.mime.text import MIMEText 
from email.mime.base import MIMEBase 
from email import encoders 
import cv2

'''
# load OpenCV's Haar cascade for face detection from disk
detector = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
 
# initialize the video stream, allow the camera sensor to warm up,
# and initialize the total number of example faces written to disk
# thus far
print("[INFO] starting video stream...")
vs = cv2.VideoCapture(0)
# vs = VideoStream(usePiCamera=True).start()
total = 1000
faceCap = True

# loop over the frames from the video stream
while faceCap:
	# grab the frame from the threaded video stream, clone it, (just
	# in case we want to write it to disk), and then resize the frame
	# so we can apply face detection faster
	ret, frame = vs.read()
 
	# detect faces in the grayscale frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1)
 
	# if the `k` key was pressed, write the *original* frame to disk
	# so we can later process it and use it for face recognition
	if key == ord("k"):
		cv2.imwrite('Caped_Face.png', frame)
		break
 
	# if the `q` key was pressed, break from the loop
	elif key == ord("q"):
		break
# print the total faces saved and do a bit of cleanup
vs.release()
cv2.destroyAllWindows()
'''

fromaddr = "ked.unlv@gmail.com"
toaddr = "ruiza12@unlv.nevada.edu"
   
# instance of MIMEMultipart 
msg = MIMEMultipart() 
  
# storing the senders email address   
msg['From'] = fromaddr 
  
# storing the receivers email address  
msg['To'] = toaddr 
  
# storing the subject  
msg['Subject'] = "Testing the Email"
  
# string to store the body of the mail 
body = "KED ALERT: BREAK IN DETECTED"
  
# attach the body with the msg instance 
msg.attach(MIMEText(body, 'plain')) 
  
# open the file to be sent  
filename = "tiredgirl.jpg"
attachment = open("Caped_Face.png", "rb") 
  
# instance of MIMEBase and named as p 
p = MIMEBase('application', 'octet-stream') 
  
# To change the payload into encoded form 
p.set_payload((attachment).read()) 
  
# encode into base64 
encoders.encode_base64(p) 
   
p.add_header('Content-Disposition', "attachment; filename= %s" % filename) 
  
# attach the instance 'p' to instance 'msg' 
msg.attach(p) 
  
# creates SMTP session 
s = smtplib.SMTP('smtp.gmail.com', 587) 
  
# start TLS for security 
s.starttls() 
  
# Authentication 
s.login(fromaddr, "gcslnfollmyrvaca") 
  
# Converts the Multipart msg into a string 
text = msg.as_string() 
  
# sending the mail 
s.sendmail(fromaddr, toaddr, text) 
  
# terminating the session 
s.quit() 