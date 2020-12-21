import os
import platform
from datetime import datetime
import cv2
import params as prms
import smile_detector as sd
import text_extractor as te
import video_cropper as vcr


def get_creation_date(input_file):
    """
    Try to get the date that a file was created, falling back to when it was
    last modified if that isn't possible.
    """
    res = None
    if platform.system() == 'Windows':
        res = os.path.getctime(input_file)
    else:
        stat = os.stat(input_file)
        try:
            res = stat.st_birthtime
        except AttributeError:
            # We're probably on Linux. No easy way to get creation dates here,
            # so we'll settle for when its content was last modified.
            res = stat.st_mtime
    return datetime.fromtimestamp(res).strftime('%Y-%m-%d %H%M%S')


def check_attendance(input_file, num_of_participants, output_folder, start_sec, end_sec):
    """
    Checks which users smiled in given video of zoom conversation and stores the results inside the output_folder
    :param start_sec: start of the attendance phase in seconds
    :param end_sec: end of the attendance phase in seconds
    :param input_file: path to the input video of the zoom conversation
    :param num_of_participants: number of participants in the zoom conversation
    :param output_folder: path to the folder to store the output txt file with the names,
           and the pictures of the users who smiled
    """
    # crop video into single-participants files
    vcr.crop_video(input_file, num_of_participants, start_sec, end_sec)
    creation_date = get_creation_date(input_file)

    # results dict
    participants = {}
    folder_path = os.path.abspath('tmp')
    for filename in os.listdir(folder_path):
        print('processing file {}'.format(filename))
        file_path = os.path.join(folder_path, filename)
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

        # Release original video
        vc.release()
        cv2.destroyAllWindows()

        print('{} proceeded done. {} is smiling={}'.format(filename, name, found_smile))
        participants[filename] = name, found_smile, smile_img

        # delete tmp file after processing it
        os.remove(file_path)

    # create dir for output
    output_dir_path = os.path.abspath(output_folder + '/' + creation_date)
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    output_file_path = os.path.abspath(output_folder + '/' + creation_date + '/Attendancy List.txt')
    if os.path.exists(output_file_path):
        os.remove(output_file_path)

    with open(output_file_path, 'a') as attendancy_list:
        for p in participants.values():
            if p[1]:  # is_smile
                cv2.imwrite(os.path.abspath(output_folder + '/' + creation_date + '/{}.jpg'.format(p[0])), p[2])
                username = 'Unidentified' if not p[0] else p[0]
                print('{} is smiling!'.format(username))
                attendancy_list.write(username + '\n')

    print('end')

# check_attendance('/Users/shaharfreiman/Desktop/Degree/Y4S1/Principles of Programming Languages/Project/SmileAttendance/input/2.mp4', 2, '/Users/shaharfreiman/Desktop/Degree/Y4S1/Principles of Programming Languages/Project/SmileAttendance/output')
