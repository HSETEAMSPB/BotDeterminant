from BotDeterminant.cv.config import *
from BotDeterminant.cv.weights import yolo_model, load_darknet_weights

import cv2
import tensorflow as tf
from PIL import Image

path = "pic.jpg"

def cv_recognize(class_name):
    image = path

    img = tf.image.decode_image(open(image, 'rb').read(), channels=3)
    img = tf.expand_dims(img, 0)
    img = tf.image.resize(img, (size, size)) / 255

    boxes, _, classes, nums = yolo_model(img)

    img = cv2.imread(image)
    
    boxes, classes, nums = boxes[0], classes[0], nums[0]
    wh = np.flip(img.shape[0:2])
    for i in range(nums):
        if class_names[int(classes[i])] in class_name:
            x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
            x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
            img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 221), 1)

    cv2.imwrite('detected_{:}'.format(path), img)
