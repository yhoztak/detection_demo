from PIL import Image
import io
import pytest
import base64
from detection_utils import *

def test_encode():
  image = Image.open("/Users/yutakahosoai/project/data/object_detection/non-aerial-images/clean_up_beach_view.jpg")
  encode_image(image)
  pass
