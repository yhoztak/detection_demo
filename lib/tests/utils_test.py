from PIL import Image
import io
import pytest
import base64
from detection_utils import *

def test_encode():
  image = Image.open("/Users/yutakahosoai/project/data/object_detection/non-aerial-images/clean_up_beach_view.jpg")
  encode_image(image)
  pass

def predict_based_on_url():
  image_url = "https://3c1703fe8d.site.internapcdn.net/newman/csz/news/800/2018/esatestingde.jpg"
  image_array = load_image_from_url(image_url)
  resp = detect_objects(image_array)
  