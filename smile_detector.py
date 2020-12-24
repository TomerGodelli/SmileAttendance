import cv2
import params as prms
import os

cascade_face = cv2.CascadeClassifier(os.path.abspath('cascades/haarcascade_frontalface_default.xml'))
cascade_eye = cv2.CascadeClassifier(os.path.abspath('cascades/haarcascade_eye.xml'))
cascade_smile = cv2.CascadeClassifier(os.path.abspath('cascades/haarcascade_smile.xml'))


def detect_smile_from_img(img):
    """
    Detect a smile in a given image
    :param img: image to detect smile in
    :return: Tuple with a boolean representing if a smile has detected, and the image with/without the detected smile
    """
    # convert to greyscale
    gray_scale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face = cascade_face.detectMultiScale(gray_scale_img, 1.3, 5)
    for (x, y, w, h) in face:
        # add blue face rectangle to original img
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # cut only face from img
        face_grayscale = gray_scale_img[y:y + h, x:x + w]
        face_color = img[y:y + h, x:x + w]

        smile = cascade_smile.detectMultiScale(face_grayscale, prms.scale_factor, 20)
        for (x_smile, y_smile, w_smile, h_smile) in smile:
            # add red smile rectangle to original img
            cv2.rectangle(face_color, (x_smile, y_smile), (x_smile + w_smile, y_smile + h_smile), (0, 0, 255), 2)
            # smile found - return true and the smile img
            return True,True, img

        # only face found
        return False, True, img

    # smile and face not found
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    return False, False, img


