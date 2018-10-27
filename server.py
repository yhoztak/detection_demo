from keras.applications import inception_v3,imagenet_utils
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
import cv2 
import numpy as np
from flask import Flask, request, make_response,jsonify
import numpy as np
import json
import urllib.request
from urllib.request import Request, urlopen
import base64
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import logging
import urllib.request
from io import BytesIO
from PIL import Image
import pandas as pd
from flask_cors import CORS

model = None
app = Flask(__name__,static_url_path='')
CORS(app)
version = "v3"
label_path = "/tmp/labels_v3.csv".format(version)
label_df = pd.read_csv(label_path,names=['label','id'])
label_df
label_lookup = label_df.set_index('id').T.to_dict('records')[0]
label_lookup 
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})


def preprocess_img(img,target_size=(299,299)):
    if (img.shape[2] == 4):
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)    
    img = cv2.resize(img,target_size)
    img = np.divide(img,255.)
    img = np.subtract(img,0.5)
    img = np.multiply(img,2.)
    return img

def load_im_from_url(url):
    requested_url = urlopen(Request(url,headers={'User-Agent': 'Mozilla/5.0'})) 
    image_array = np.asarray(bytearray(requested_url.read()), dtype=np.uint8)
    print (image_array.shape)
    print (image_array)
    image_array = cv2.imdecode(image_array, -1)
    print (image_array.shape)
    return image_array

def load_im_from_system(url):
    image_url = url.split(',')[1]
    image_url = image_url.replace(" ", "+")
    image_array = base64.b64decode(image_url)
    im = Image.open(BytesIO(image_array))
    image = np.asarray(im.convert('RGB'))
    return image[:, :, ::-1].copy()

def predict(img):
    img=preprocess_img(img)
    # print (img.shape)
    global model
    if model is None:
        # model =inception_v3.InceptionV3()
        # model.compile(optimizer='adam', loss='categorical_crossentropy')
        model_path = "/tmp/debris_model_v3_10_6.h5"
        model = models.load_model(model_path, backbone_name='resnet50')

    preds = model.predict_on_batch(np.array([img]))
    return imagenet_utils.decode_predictions(preds)


def detect_objects(image):
    detected_objects = []
    image = preprocess_image(image)
    global model
    if model is None:
        model_path = "/tmp/debris_model_v3_10_6.h5"
        model = models.load_model(model_path, backbone_name='resnet50')

    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    detected_label =set()
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        #only shows the highest confidence box
        if label in detected_label:
            continue
        detected_label.add(label)
        if score < 0.15:
            break
        color = label_color(label)
        b = box.astype(int)
        detected_objects.append({'x1':b[0], 'y1':b[1], 'x2': b[2], 'y2':b[3],'label':label, 'label_name': label_lookup[label], 'score':float(score)})
    return detected_objects

def load_image_from_url(url):
    with urllib.request.urlopen(url) as url:
        f = BytesIO(url.read())
        image = np.asarray(Image.open(f).convert('RGB'))
        return  image[:, :, ::-1].copy()
    return None

@app.route('/classify_system', methods=['GET'])
def classify_system():
    image_url = request.args.get('imageurl')
    image_array = load_im_from_system(image_url)
    resp = detect_objects(image_array)
    result = []
    for r in resp:
        result.append({"class_name":r['label_name'],"score":r['score']})
    return jsonify({'results':result})

@app.route('/classify_url', methods=['GET'])
def classify_url():
    image_url = request.args.get('imageurl')
    image_array = load_image_from_url(image_url)
    resp = detect_objects(image_array)
    result = []
    for r in resp:
        result.append({"class_name":r['label_name'],"score":r['score']})
    return jsonify({'results':result})


@app.route('/classify-system', methods=['GET'])
def show_system():
    return app.send_static_file('system.html')

@app.route('/classify-url', methods=['GET'])
def show_url():
    return app.send_static_file('url.html')

@app.route('/density-map', methods=['GET'])
def show_map():
    return app.send_static_file('map.html')

@app.route('/test', methods=['GET'])
def test():
    return app.send_static_file('test.html')


@app.route('/tf', methods=['GET'])
def tf():
    return app.send_static_file('tf.html')


@app.route('/', methods=['GET'])
def root():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)

