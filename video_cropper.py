import numpy as np
import cv2
import params as prms


def crop_video(file_path, n_participants):
    print('cropping participants video {}'.format(file_path))

    # get video and read first frame for dimensions
    cap = cv2.VideoCapture(file_path)
    ret, frame = cap.read()
    h, w, d = np.shape(frame)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print('fps: {}'.format(fps))
    # print('h: {}, w: {}'.format(h, w))

    # read offsets params
    params = prms.offset[str(n_participants)]
    horiz_divisions = params['horiz_divisions']  # Number of horizontally tiles
    vert_divisions = params['vert_divisions']  # Number of vertically tiles
    horiz_offset = params['horiz_offset']
    vert_offset = params['vert_offset']

    seg_h = int((h - (2 * vert_offset)) / vert_divisions)  # Tile height
    seg_w = int((w - (2 * horiz_offset)) / horiz_divisions)  # Tile width
    # print('seg_h: {}, seg_w: {}'.format(seg_h, seg_w))

    # Init output videos
    out_videos = [0] * n_participants

    for i in range(n_participants):
        out_videos[i] = cv2.VideoWriter('tmp\\out{}.avi'.format(str(i)), cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                        fps,
                                        (seg_w, seg_h))

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            vid_inx = 0  
            for i in range(vert_divisions):
                for j in range(horiz_divisions):
                    # Get top left corner coordinates of current tile
                    row = vert_offset + i * seg_h
                    col = horiz_offset + j * seg_w
                    roi = frame[row:row + seg_h, col:col + seg_w, 0:3]  # Copy the region of interest
                    out_videos[vid_inx].write(roi)
                    vid_inx += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Release original video
    cap.release()

    # Release each participant's video
    for i in range(n_participants):
        out_videos[i].release()
    # Release everything if job is finished
    cv2.destroyAllWindows()

