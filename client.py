import socket
import pickle
from PIL import Image

HOST = '192.168.1.19'  # The server's hostname or IP address
PORT = 4567        # The port used by the server


s= socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))
full_msg = b''
while True:
	msg = s.recv(1024)
	if len(msg) <= 0:
		break
	full_msg += msg

if len(full_msg) < 100:
	print("New Key Creation at Address 1234")
	print("Key\tCreation Date & Time\tExpiration Date & Time \tType")
	print(full_msg.decode('utf-8'))

else:
	breakIn = full_msg
	print(breakIn[0:37])
	newMsg = breakIn[37:]
	#print(newMsg)

	encode_img = pickle.loads(newMsg)
	encode_img.show()
s.close()

