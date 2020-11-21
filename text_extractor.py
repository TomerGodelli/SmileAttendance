from PIL import ImageOps
from PIL import Image
import pytesseract
from pytesseract import image_to_string
import re
import cv2
import params as prms

pattern = re.compile(r'[^A-Za-z ]+')

# Mention the installed location of Tesseract-OCR in your system
pytesseract.pytesseract.tesseract_cmd = prms.tesseract_cmd_path


def extract_text_from_img(img):
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
    height, width, channels = img.shape
    name_height = int(height * prms.name_height_ratio)
    name_width = int(width * prms.name_width_ratio)
    return img[height-name_height:height, 0:name_width]


def extract_text_from_video(file_name):
    print('extracting name from {}'.format(file_name))

    vc = cv2.VideoCapture(file_name)

    text_found = {}

    # extract text once in 24 frames (~once in 2 seconds)
    read_frame_rate = 24
    frame_counter = 1
    while True:
        ret, frame = vc.read()

        # End of file
        if not ret:
            break
        if frame_counter % read_frame_rate == 0:
            im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            im = im.convert('L')
            im_inv = ImageOps.invert(im)
            name = image_to_string(im_inv)
            striped_name = pattern.sub('', name).strip()

            if striped_name:
                if striped_name in text_found:
                    text_found[striped_name] += 1
                else:
                    text_found[striped_name] = 1

        frame_counter += 1

    return max(text_found, key=text_found.get)
