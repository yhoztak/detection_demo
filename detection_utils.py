from keras.applications import inception_v3,imagenet_utils
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
import cv2 
import numpy as np
import json
import base64
import logging
import urllib.request
from io import BytesIO
from PIL import Image, ImageDraw
import pandas as pd
from wtforms import Form
from wtforms import ValidationError
from flask_wtf.file import FileField
from werkzeug.datastructures import CombinedMultiDict
from wtforms import Form
from os import path
import sys
import tempfile
from urllib.request import Request, urlopen
from io import StringIO
from datetime import datetime
import os 
import boto3
content_types = {'jpg': 'image/jpeg',
                 'jpeg': 'image/jpeg',
                 'png': 'image/png'}
extensions = sorted(content_types.keys())

model = None

#+=================== Model related =========================
version = 'v3'
label_path = "/tmp/labels_{}.csv".format(version)
label_df = pd.read_csv(label_path,names=['label','id'])
label_df
label_lookup = label_df.set_index('id').T.to_dict('records')[0]
label_lookup 


if 'heroku' in os.environ and os.environ['heroku']:
  s3_client = boto3.client('s3', aws_access_key_id=os.environ['S3_KEY'], aws_secret_access_key=os.environ['S3_secret'])
  s3_resource = boto3.resource('s3', aws_access_key_id=os.environ['S3_KEY'], aws_secret_access_key=os.environ['S3_secret'])
else:
  boto3.setup_default_session(profile_name='hawaii')
  s3_client = boto3.client('s3')
  s3_resource = boto3.resource('s3')

bucket_name = 'hawaii-marine-debris'
version='v3'

def load_label_lookup():
    label_path = "/tmp/labels_{}.csv".format(version)
    label_df = pd.read_csv(label_path,names=['label','id'])
    label_lookup = label_df.set_index('id').T.to_dict('records')[0]
    return label_lookup

def load_debris_model():
    global model
    if model is None:
        # model =inception_v3.InceptionV3()
        # model.compile(optimizer='adam', loss='categorical_crossentropy')
        model_path = "/tmp/debris_model_v3_10_6.h5"
        model = models.load_model(model_path, backbone_name='resnet50')
    return model
        

#just to get labels
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

def backup_to_s3(image_path):
    ts = datetime.now().strftime("%Y-%m-%d-%H")
    filename= path.basename(image_path).lower().replace(" ", "_")
    extension ="jpg" #guess for now

    s3_filepath = "images/from_app/external/{}_{}.{}".format(filename, ts,extension)
    

    s3_client.upload_file(image_path, bucket_name, s3_filepath)

def detect_marine_objects(image_path):
    objects_points_detected_so_far = []

    backup_to_s3(image_path)
    print("backed up image")
    image = Image.open(image_path).convert('RGB')
    image_array = im_to_im_array(image)
    preprocessed_image = preprocess_image(image_array)
    model = load_debris_model()
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(preprocessed_image, axis=0))
    # image.thumbnail((480, 480), Image.ANTIALIAS)
    result = {}
    new_images = {}
    debris_count = {}
    result['original'] = encode_image(image.copy())
    all_obj_image = image.copy()

    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        if score < 0.15: continue
        color = tuple(label_color(label))
        b = box.astype(int)
        points = {'x1':b[0], 'y1':b[1], 'x2': b[2], 'y2':b[3]}
        if len(objects_points_detected_so_far)>0:
            max_overlap = max([get_iou(points,v) for v in objects_points_detected_so_far])
            if max_overlap>0.2:
                continue

        cls = label
        if cls not in new_images.keys():
          new_images[cls] = image.copy()
          debris_count[cls]=1
        else:
          debris_count[cls]+=1

        draw_bounding_box_on_image(new_images[cls], box,color=color,
                                   thickness=int(score*10)-4)
        draw_bounding_box_on_image(all_obj_image, box,color=color,
                                   thickness=int(score*10)-4)

        objects_points_detected_so_far.append(points)

    result['all'] = encode_image(all_obj_image)
    result['summary'] = {}
    result['color'] = {}

    for cls, new_image in new_images.items():
      category = label_lookup[cls]
      result[category] = encode_image(new_image)
      result['summary'][category] = debris_count[cls]
      result['color'][category] = tuple(label_color(cls))

    result['summary']['all'] = sum(debris_count.values())
    # import pdb;pdb.set_trace()
    #                 <td><hr style="height:1px;border-top:1px solid {{'#%02x%02x%02x' % result['color']['Buoys'] }}" /></td>

#also calculate total number of debris, and counts by type of debris
    return result
# =================== Image related =========================


def preprocess_img(img,target_size=(600,600)):
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

def load_image_from_url(url):
    with urllib.request.urlopen(url) as url:
        f = BytesIO(url.read())
        image = np.asarray(Image.open(f).convert('RGB'))
        return  image[:, :, ::-1].copy()
    return None

def draw_bounding_box_on_image(image, box, color='red', thickness=4):
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  # ymin, xmin, ymax, xmax = box
  # (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
  #                               ymin * im_height, ymax * im_height)
# x1':b[0], 'y1':b[1], 'x2': b[2], 'y2':b[3]
  (left, top, right, bottom) = box
  draw.line([(left, top), (left, bottom), (right, bottom),
             (right, top), (left, top)], width=thickness, fill=color)


def encode_image(image):
  image_buffer = BytesIO()
  image.save(image_buffer, format='PNG')
  imgstr = 'data:image/png;base64,{:s}'.format(base64.b64encode(image_buffer.getvalue()).decode())
  return imgstr

def is_image():
  def _is_image(form, field):
    if not field.data:
      raise ValidationError()
    elif field.data.filename.split('.')[-1].lower() not in extensions:
      raise ValidationError()

  return _is_image

def im_to_im_array(rgb_im):
    image = np.asarray(rgb_im)
    return image[:, :, ::-1].copy()

#==================== Coordinates related ===================
def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] <= bb1['x2']
    assert bb1['y1'] <= bb1['y2']
    assert bb2['x1'] <= bb2['x2']
    assert bb2['y1'] <= bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

    return model


#image url to try:
#image file to try: /Users/yutakahosoai/project/data/object_detection/non-aerial-images/clean_up_beach_view.jpg

#test 1
# from PIL import Image
# import io
# image = Image.open("/Users/yutakahosoai/project/data/object_detection/non-aerial-images/clean_up_beach_view.jpg")
# image_buffer = io.BytesIO()
# image.save(image_buffer, format='PNG')
# imgstr = 'data:image/png;base64,{:s}'.format(
#   base64.b64encode(image_buffer.getvalue()))



