# load config
import json
with open('roboflow_config.json') as f:
    config = json.load(f)

    ROBOFLOW_API_KEY = config["ROBOFLOW_API_KEY"]
    ROBOFLOW_MODEL = config["ROBOFLOW_MODEL"]
    ROBOFLOW_SIZE = config["ROBOFLOW_SIZE"]

    FRAMERATE = config["FRAMERATE"]
    BUFFER = config["BUFFER"]

import cv2
import base64
import numpy as np
import requests
import time
import json
from playsound import playsound

# Construct the Roboflow Infer URL
# (if running locally replace https://detect.roboflow.com/ with eg http://127.0.0.1:9001/)
# replaced https://detect.roboflow.com/ with http://localhost:9001/
upload_url = "".join([
    "http://localhost:9001/",
    ROBOFLOW_MODEL,
    "?api_key=",
    ROBOFLOW_API_KEY
])

# Get webcam interface via opencv-python
video = cv2.VideoCapture(0)

# Infer via the Roboflow Infer API and return the result
def infer(img):
    # Get the current image from the webcam
    
    # Resize (while maintaining the aspect ratio) to improve speed and save bandwidth
    height, width, channels = img.shape
    scale = ROBOFLOW_SIZE / max(height, width)
    img = cv2.resize(img, (round(scale * width), round(scale * height)))

    # Encode image to base64 string
    retval, buffer = cv2.imencode('.jpg', img)
    img_str = base64.b64encode(buffer)
    # Get prediction from Roboflow Infer API
    resp = requests.post(upload_url, data=img_str, headers={
        "Content-Type": "application/x-www-form-urlencoded"
    }, stream=True)
   
    response = resp.json()

    return response

# Main loop; infers sequentially until you press "q"
i = 0
while True:
    # On "q" keypress, exit
    if(cv2.waitKey(1) == ord('q')):
        break

    # Synchronously get a prediction from the Roboflow Infer API
    ret, img = video.read()
    response = infer(img)
    #print(response)
    if(response["predictions"] and (response["predictions"][0]["confidence"]) > 0.70):
        print(response["predictions"][0]["class"] + " " + str(response["predictions"][0]["confidence"]) + " " + str(i))
        if(response["predictions"][0]["class"] == "Poor Posture"):
            playsound("boosted.mp3")
    # And display the inference results
    #cv2.imshow('image', img)
    #cv2.waitKey(0)
    i = i + 1

# Release resources when finished
video.release()
cv2.destroyAllWindows()
