from PIL import Image
import numpy as np
import flask
from skimage.io import imread
from DenseNet.myPredict3 import MyDenseNet

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None
model = MyDenseNet()
model.load_model()

@app.route('/', methods=['GET'])
def index():
    return "TensorFlow Sterility Test"

@app.route('/hello', methods=['GET'])
def hello():
    return "Hello Flask!"

@app.route("/predictbyfilename", methods=['POST'])
def predict_by_filename():
    data = {"success": False}
    if flask.request.method == "POST":
        json = flask.request.json
        # get path for the image
        path = json["path"]
        # read image and process
        # print(path)

        try: 
            img = imread(path)
            res = model.predict_one_image(img)
            
            data["predictions"] = [res["class"], float(res["score"]),float(res["distribution_score"])]
            data["result"] = "Predited"
            # indicate that the request was a success
            data["success"] = True

        except Exception as e:
            data["result"] = "Wrong Image" + path
            # print("nmsl")
            data["success"] = False
    
    return flask.jsonify(data)


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started"))

    app.run()