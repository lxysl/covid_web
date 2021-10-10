import os
import sys

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

import torch
from torchvision import transforms

# Some utilities
import numpy as np
from util import base64_to_pil

# Declare a flask app
app = Flask(__name__)

classes = ['正常', '肺炎（非COVID-19）', '肺炎（COVID-19）']
input_channel = 1
input_size = (224, 224)
crop_size = (340, 380)

model_path = '/Users/lxy/PycharmProjects/mnist_web/densenet201_3.pth'
weight_path = '/Users/lxy/PycharmProjects/mnist_web/densenet201_3_best_acc.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

to_tensor = transforms.Compose([
    transforms.CenterCrop(crop_size),
    transforms.Resize(input_size),
    transforms.Grayscale(input_channel),
    transforms.ToTensor(),
    transforms.Normalize([0.63507175], [0.3278614])
])

print('Model loading. Start serving...')
model = torch.load(model_path, map_location='cpu')
model.load_state_dict(torch.load(weight_path, map_location='cpu'))
model.eval()
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img, model):
    # Preprocessing the image
    # x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    # x = np.array(img)
    x = to_tensor(img)
    x = torch.unsqueeze(x, 0)
    x = x.to(device)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    # x = preprocess_input(x, mode='tf')

    with torch.no_grad():
        output = model(x)
        output = torch.nn.functional.softmax(output[0], dim=0)
        confidence, class_index = torch.max(output, 0)

    print(output)
    print(confidence, class_index)
    pred = classes[class_index]
    return pred, confidence.item()


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)

        # Save the image to ./uploads
        # img.save("./uploads/image.png")

        # Make prediction
        pred, confidence = model_predict(img, model)
        print(pred, confidence)
        # Process your result for human
        # pred_proba = "{:.3f}".format(np.amax(pred))  # Max probability
        # result = str(np.argmax(pred[0]))
        # pred_class = decode_predictions(preds, top=1)   # ImageNet Decode

        # result = str(pred_class[0][0][1])               # Convert to string
        # result = result.replace('_', ' ').capitalize()

        # Serialize the result, you can add additional fields
        return jsonify(result=pred, probability=confidence)

    return None


if __name__ == '__main__':
    # app.run(port=5002, threaded=False)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
