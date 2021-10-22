import os
import copy

# Flask
from flask import Flask, request, render_template, jsonify
from gevent.pywsgi import WSGIServer

import torch
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from torchcam.cams import GradCAMpp
from torchcam.utils import overlay_mask

# Some utilities
import numpy as np
import matplotlib.pyplot as plt
from util import base64_to_pil, pil_to_base64

# Declare a flask app
app = Flask(__name__)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
classes = ['正常', '肺炎（非COVID-19）', '肺炎（COVID-19）']
input_channel = 1
input_size = (224, 224)
crop_size = (340, 380)

model_path = '/Users/lxy/PycharmProjects/covid_web/densenet201_3.pth'
weight_path = '/Users/lxy/PycharmProjects/covid_web/densenet201_3_best_acc.pth'
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
model_cam = copy.deepcopy(model)
cam_extractor = GradCAMpp(model_cam, input_shape=(1, 224, 224))
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img, model):
    # Preprocessing the image
    x = to_tensor(img)
    x = torch.unsqueeze(x, 0)
    x = x.to(device)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!

    with torch.no_grad():
        output = model(x)
        output = torch.nn.functional.softmax(output[0], dim=0)
        confidence, class_index = torch.max(output, 0)

    print(output)
    print(confidence, class_index)
    pred = classes[class_index]
    return pred, confidence.item()


def grad_cam(img, model):
    x = to_tensor(img)
    x = torch.unsqueeze(x, 0)
    x = x.to(device)

    out = model(x)
    activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
    plt.imshow(activation_map.numpy())
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('/Users/lxy/PycharmProjects/mnist_web/activation_map.png')

    img = img.convert('RGB')
    result = overlay_mask(img, to_pil_image(activation_map, mode='F'), alpha=0.5)
    plt.imshow(result)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('/Users/lxy/PycharmProjects/mnist_web/overlay.png')
    return result


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)

        # Make prediction
        cam_img = grad_cam(img, model_cam)
        pred, confidence = model_predict(img, model)
        print(pred, confidence)

        # Serialize the result, you can add additional fields
        return jsonify(result=pred, probability=confidence, img=pil_to_base64(cam_img))

    return None


if __name__ == '__main__':
    # app.run(port=5002, threaded=False)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
