import cv2
import numpy as np
"""
{"0.jpg": [ [[93, 22], [263, 191], 0.5214826979499776], head
            [[323, 250], [354, 320], 0.4218821440008469], left elbow1
            [[187, 317], [354, 320], 0.0005312466964824125], left elbow2
            [[64, 220], [89, 311], 0.409674733877182], right elbow1
            [[89, 311], [221, 321], 0.15903244002402062], right elbow2
            [[106, 265], [179, 312], 4.556352768976998e-05], left hand
            [[236, 294], [282, 320], 5.650643178404617e-06], right hand
            [[187, 250], [354, 320], 0.28138766052628245], left elbow
            [[64, 220], [221, 321], 0.2731476464784161] ], right elbow
"""
def np_load_frame_5(filename, annotation, img_size):
    image = cv2.imread(filename)
    slices = [0, 7, 8, 5, 6]
    res = []
    for i in slices:
        x1, y1 = annotation[i][0]
        x2, y2 = annotation[i][1]
        score = annotation[i][2]
        cropped = image[y1:y2, x1:x2]
        if cropped.shape[0] > 30 and cropped.shape[1] > 30 and score > 0.001:
            resized = cv2.resize(cropped, (img_size, img_size))
        else:
            resized = np.zeros((img_size, img_size, 3), dtype=np.float32)
        res.append((resized / 127.5) - 1.0)
    return res

def np_load_frame_7(filename, annotation, img_size):
    image = cv2.imread(filename)
    res = []
    for i in range(7):
        x1, y1 = annotation[i][0]
        x2, y2 = annotation[i][1]
        score = annotation[i][2]
        cropped = image[y1:y2, x1:x2]
        if cropped.shape[0] > 30 and cropped.shape[1] > 30 and score > 0.001:
            resized = cv2.resize(cropped, (img_size, img_size))
        else:
            resized = np.zeros((img_size, img_size, 3), dtype=np.float32)
        res.append((resized / 127.5) - 1.0)
    return res
