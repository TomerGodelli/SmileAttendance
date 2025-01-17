import os

# video offset params
offset = {
    '2': dict(horiz_divisions=2, vert_divisions=1, horiz_offset=0, vert_offset=185),
    '4': dict(horiz_divisions=2, vert_divisions=2, horiz_offset=0, vert_offset=0),
    '6': dict(horiz_divisions=3, vert_divisions=2, horiz_offset=13, vert_offset=125),
    '9': dict(horiz_divisions=3, vert_divisions=3, horiz_offset=13, vert_offset=12),
    '12': dict(horiz_divisions=4, vert_divisions=3, horiz_offset=0, vert_offset=90),

}

# smile detection params
smile_read_frame_rate = 1
scale_factor = 1.1  # smile detection sensitivity (should be above 1, default is 1.7)

# text extractor params
text_read_frame_rate = 1
min_threshold = 4
tesseract_cmd_path_64 = os.path.abspath('C:/Program Files/Tesseract-OCR/tesseract.exe')
tesseract_cmd_path_32 = os.path.abspath('C:/Program Files (x86)/Tesseract-OCR/tesseract.exe')
name_height_ratio = 0.2
name_width_ratio = 0.5






