from face_detector import YoloDetector
import numpy as np
from PIL import Image

model = YoloDetector(target_size=720, device="cuda:0", min_face=90)
orgimg = np.array(Image.open("C:\\Users\\Ananya\\Downloads\\WIN_20210811_19_48_21_Pro.jpg"))
bboxes, points = model.predict(orgimg)
