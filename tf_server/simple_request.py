# USAGE
# python simple_request.py

# import the necessary packages
import requests
import os

# initialize the Keras REST API endpoint URL along with the input
# image path
KERAS_REST_API_URL = "http://localhost:5000/predictbyfilename"
IMAGE_PATH = "D:\\tf_server\\imgs\\Pseudomonas.jpg"

# load the input image and construct the payload for the request
payload = {"path": IMAGE_PATH}

r = requests.get("http://localhost:5000/hello")
print(r)
# submit the request
r = requests.post(KERAS_REST_API_URL, json=payload).json()


print(r)
# ensure the request was successful
if r["success"]:
    # loop over the predictions and display them
    
    if r["result"] == "Predited":
        print("Request succeeded")
        print(r["predictions"])
# otherwise, the request failed
else:
    print("Request failed")

