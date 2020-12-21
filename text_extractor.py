from PIL import ImageOps
from PIL import Image
import pytesseract
from pytesseract import image_to_string
import re
import cv2
import params as prms

pattern = re.compile(r'[^A-Za-z ]+')

# Mention the installed location of Tesseract-OCR in your system
# pytesseract.pytesseract.tesseract_cmd = prms.tesseract_cmd_path


def extract_text_from_img(img):
    """
    Detect and extract text from a given image
    :param img: image to extract text from
    :return: the extracted text if detected, None otherwise
    """
    img = crop_name_from_image(img)
    im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    im = im.convert('L')
    im_inv = ImageOps.invert(im)
    name = image_to_string(im_inv)
    striped_name = pattern.sub('', name).strip()
    if striped_name:
        return striped_name
    return None


def crop_name_from_image(img):
    """
    Crops the name of the user from user picture from zoom conversation
    :param img: user image from zoom conversation to crop username image part from
    :return: cropped part of the image that contains the username of the user
    """
    height, width, channels = img.shape
    name_height = int(height * prms.name_height_ratio)
    name_width = int(width * prms.name_width_ratio)
    return img[height-name_height:height, 0:name_width]