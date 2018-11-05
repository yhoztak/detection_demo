from flask import Flask, request, make_response,jsonify, render_template, request, redirect, url_for, send_from_directory
import json
import urllib.request
from urllib.request import Request, urlopen
import base64
import logging
from io import BytesIO
import pandas as pd
from flask_cors import CORS
from wtforms import Form
from wtforms import ValidationError
from flask_wtf.file import FileField
from werkzeug.datastructures import CombinedMultiDict
from wtforms import Form
from os import path
basepath = path.dirname(__file__)
import sys
import tempfile
from detection_utils import *
from io import StringIO

app = Flask(__name__,static_url_path='')
CORS(app)
version = "v3"
label_path = "/tmp/labels_v3.csv".format(version)
label_df = pd.read_csv(label_path,names=['label','id'])
label_df
label_lookup = label_df.set_index('id').T.to_dict('records')[0]
label_lookup 
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})


class PhotoForm(Form):
  input_photo = FileField(
      'File extension should be: %s (case-insensitive)' % ', '.join(extensions),
      validators=[is_image()])
  
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

@app.route('/')
def upload():
  photo_form = PhotoForm(request.form)
  return render_template('upload.html', photo_form=photo_form, result={})

@app.route('/post', methods=['GET', 'POST'])
def post():
  form = PhotoForm(CombinedMultiDict((request.files, request.form)))
  if request.method == 'POST' and form.validate():
    with tempfile.NamedTemporaryFile() as temp:
      form.input_photo.data.save(temp)
      temp.flush()
      lat = request.form['latitude']
      lon = request.form['longtitude']

      result = detect_marine_objects(temp.name, lat,lon)

    photo_form = PhotoForm(request.form)
    return render_template('upload.html',
                           photo_form=photo_form, result=result)
  else:
    return redirect(url_for('upload'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)

# debris image URL to try : https://3c1703fe8d.site.internapcdn.net/newman/csz/news/800/2018/esatestingde.jpg
