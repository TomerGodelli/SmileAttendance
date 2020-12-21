import os

# video offset params
offset = {
    '2': dict(horiz_divisions=2, vert_divisions=1, horiz_offset=0, vert_offset=185),
    '4': dict(horiz_divisions=2, vert_divisions=2, horiz_offset=0, vert_offset=0),
    '6': dict(horiz_divisions=3, vert_divisions=2, horiz_offset=23, vert_offset=145),
}

# smile detection params
smile_read_frame_rate = 1
scale_factor = 2  # smile detection sensitivity (should be above 1, default is 1.7)

# text extractor params
text_read_frame_rate = 1
min_threshold = 3
# tesseract_cmd_path = os.path.abspath('C:/Program Files/Tesseract-OCR/tesseract.exe')
name_height_ratio = 0.083
name_width_ratio = 0.3125




