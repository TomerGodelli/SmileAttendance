import os
import cv2
import smile_detector as sd
import text_extractor as te
import video_cropper as vcr
import params as prms


def main():
    input_file = 'input\\4_2.mp4'
    num_of_participants = 4  # 2 on 2

    # crop video into single-participants files
    vcr.crop_video(input_file, num_of_participants)

    # results dict
    participants = {}
    for filename in os.listdir('tmp'):
        print('processing file {}'.format(filename))
        file_path = 'tmp\\' + filename
        vc = cv2.VideoCapture(file_path)
        found_smile = False
        smile_img = None
        text_found = {}
        found_name = False
        name = ''

        # extract text and detect smiles only once in x frames
        text_read_frame_rate = prms.text_read_frame_rate
        smile_read_frame_rate = prms.smile_read_frame_rate
        frame_counter = 1

        while True:
            ret, frame = vc.read()

            # End of file
            if not ret:
                break

            # detect smile
            if not found_smile and frame_counter % smile_read_frame_rate == 0:
                found_smile, smile_img = sd.detect_smile_from_img(frame)

            if not found_name and frame_counter % text_read_frame_rate == 0:
                name = te.extract_text_from_img(frame)

                # if found text in frame
                if name:
                    if name in text_found:
                        text_found[name] += 1
                    else:
                        text_found[name] = 1

                    # if extract same text more than min_threshold times we assume its the name
                    if text_found[name] > prms.min_threshold:
                        found_name = True

            frame_counter += 1
            if found_smile and found_name:
                break

        print('{} proceeded done. {} is smiling={}'.format(filename,name,found_smile))
        participants[filename] = name, found_smile, smile_img

    for p in participants.values():
        if p[1]: # is_smile
            cv2.imwrite('output\\{}.jpg'.format(p[0]), p[2])
            print('{} is smiling!'.format(p[0]))

    print('end')



main()
