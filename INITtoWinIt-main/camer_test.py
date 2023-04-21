import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)
text = "Student ID Found: 6129480"
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
color = (34,139,34)
thickness = 5
textThickness = 1
text_size, _ = cv2.getTextSize(text, font, font_scale, textThickness)
user_input = ""
# adjust the box size based on the text size
box_width = text_size[0] + 20
box_height = text_size[1] + 20
box_pos = (50,50)
box_size = (box_width, box_height)

# get the dimensions of the camera frame
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) + 200
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) + 200

# calculate the size and position of the square frame
frame_size = min(frame_width, frame_height) // 2
frame_pos = ((frame_width - frame_size) // 2, (frame_height - frame_size) // 2)

def on_text_box_change(text):
    global user_input 
    user_input = text

# Define the callback function for mouse clicks
def on_mouse_click(event, x, y):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.setMouseCallback('Camera', on_mouse_move)

# Define the callback function for mouse movement
def on_mouse_move(event, x, y, flags, param):
    if flags & cv2.EVENT_FLAG_LBUTTON:
        # Draw the text box and current user input on the camera frame
        cv2.rectangle(frame, (50, 50), (400, 150), (255, 255, 255), -1)
        cv2.putText(frame, 'Enter a number:', (60, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        if user_input:
            cv2.putText(frame, user_input, (60, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(frame, '|', (60 + cv2.getTextSize(user_input, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0], 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

# Create the window with the text box
cv2.namedWindow('Input', cv2.WINDOW_NORMAL)
cv2.createTrackbar('Input', 'Input', 50, 50, on_text_box_change)
cv2.setMouseCallback('Input', on_mouse_click)
cv2.imshow('Input', np.zeros((1,1), dtype=np.uint8))
#userFound controls if a user has been detected
userFound = False
#sets default color value
color = (255,255,255)
start = time.time()
while True:
    end = time.time()
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)
    #creates white box background
    #cv2.rectangle(frame, box_pos, (box_pos[0] + box_size[0], box_pos[1] + box_size[1]), (255,255,255), -1)
    #creates frame
    cv2.rectangle(frame, frame_pos, (frame_pos[0] + frame_size, frame_pos[1] + frame_size), color , thickness)
    #writes text to white box
    #cv2.putText(frame, text, (box_pos[0]+10, box_pos[1] + box_height - 10), font, font_scale, (0,0,0), thickness)
    if userFound:
        cv2.putText(frame, text, (box_pos[0]+10, box_pos[1] + box_height - 10), font, font_scale, (0,0,0), thickness)
        cv2.putText(frame, "Loss = 0.1065", (box_pos[0]+950, box_pos[1] + box_height - 10), font, font_scale, (0,0,0), thickness)

        color = (0,255,0)
    else:
        color = (0,0,255)
    #keeps the color white until 2 seconds have passed
    if ((end-start) < 4):
        color = (255,255,255)
    cv2.imshow('Camera', frame)
    if cv2.waitKey(1) & 0xFF == ord('g'):
        userFound = ~userFound
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.namedWindow('Input', cv2.WINDOW_NORMAL)
    cv2.createTrackbar('Input', 'Input', 50, 50, on_text_box_change)
    cv2.setMouseCallback('Input', on_mouse_click)
    cv2.imshow('Input', np.zeros((1,1), dtype=np.uint8))

cap.release()
cv2.destroyAllWindows()