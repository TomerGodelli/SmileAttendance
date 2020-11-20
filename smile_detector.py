import cv2
import params as prms

cascade_face = cv2.CascadeClassifier('cascades\\haarcascade_frontalface_default.xml')
cascade_eye = cv2.CascadeClassifier('cascades\\haarcascade_eye.xml')
cascade_smile = cv2.CascadeClassifier('cascades\\haarcascade_smile.xml')


def detect_smile_from_img(img):
    # convert to greyscale
    gray_scale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face = cascade_face.detectMultiScale(gray_scale_img, 1.3, 5)
    for (x, y, w, h) in face:
        # add blue face rectangle to original img
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # cut only face from img
        face_grayscale = gray_scale_img[y:y + h, x:x + w]
        face_color = img[y:y + h, x:x + w]

        # eye = cascade_eye.detectMultiScale(ri_grayscale, 1.2, 18)
        # for (x_eye, y_eye, w_eye, h_eye) in eye:
        #     cv2.rectangle(ri_color, (x_eye, y_eye), (x_eye + w_eye, y_eye + h_eye), (0, 180, 60), 2)

        smile = cascade_smile.detectMultiScale(face_grayscale, prms.scale_factor, 20)
        for (x_smile, y_smile, w_smile, h_smile) in smile:
            # add red smile rectangle to original img
            cv2.rectangle(face_color, (x_smile, y_smile), (x_smile + w_smile, y_smile + h_smile), (0, 0, 255), 2)
            # smile found - return true and the smile img
            return True, img

    # smile not found
    return False, img


def detect_smile_from_video(file_name):
    vc = cv2.VideoCapture(file_name)
    is_smile = False
    smile_img = None
    print('detecting smiles in {}'.format(file_name))

    # try detect smile once in 12 frames (~twice a second)
    # read_frame_rate = 1
    # frame_counter = 1
    while True:
        ret, img = vc.read()

        # End of file
        if not ret:
            break

        # if frame_counter % read_frame_rate == 0:

        is_smile_frame, rect_img = detect_smile_from_img(img)

        if is_smile_frame:
            is_smile = True
            smile_img = rect_img
            break
        # frame_counter += 1

    # For Debug only - show video while procesing
    # cv2.imshow('Video', rect_img)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

    vc.release()
    cv2.destroyAllWindows()

    return is_smile, smile_img
