from styx_msgs.msg import TrafficLight
import keras
from keras.models import load_model, model_from_json
from keras.preprocessing.image import img_to_array, load_img
import tensorflow as tensorflow
import cv2
import numpy as np


class TLClassifier(object):
    def __init__(self):
        self.cascade = None
        self.test_model = None
        self.graph = None

    def init(self):
        self.cascade = cv2.CascadeClassifier('./models/cascade.xml')

        if keras.__version__ < '2.0.0':
            model_json = ''
            with open('./models/m_structure.json', 'r') as file:
                model_json = file.read();
            self.test_model = model_from_json(model_json)
            self.test_model.load_weights('./models/m_weights.h5')
        else:
            print 'WARN! Using Keras version {}'.format(keras.__version__)
            self.test_model = load_model('./models/tl_state_aug_v3.h5')
        self.graph = tensorflow.get_default_graph()

    def non_max_suppression_fast(self, boxes, overlapThresh):
        if len(boxes) == 0:
            return []
            boxes = np.array(boxes)
        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")
        pick = []
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = x1 + boxes[:, 2]
        y2 = y1 + boxes[:, 3]
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y1)
        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            overlap = (w * h) / area[idxs[:last]]
            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > overlapThresh)[0])))
        return boxes[pick].astype("int")

    def get_classification(self, cv_image):
        """Determines the color of the traffic light in the image

        Args:
            cv_image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        box = self.cascade.detectMultiScale(cv_image, 1.3, 3)
        box = self.non_max_suppression_fast(box, 0.2)
        state = TrafficLight.UNKNOWN
        img_width, img_height = 150, 150
        for (x, y, w, h) in box:
            dh = int(round(h * 0.1))
            line = cv_image[(y + dh):(y + h - dh), int(round(x + w / 2)), :]
            if np.std(line) < 32:
                continue
            tl_img = cv_image[y:(y + h), x:(x + w)]
            tl_img_rgb = cv2.resize(tl_img, (img_width, img_height))
            tl_img_rgb = cv2.cvtColor(tl_img_rgb, cv2.COLOR_BGR2RGB)
            tl_img_data = img_to_array(tl_img_rgb)
            tl_img_data = np.expand_dims(tl_img_data, axis=0)
            with self.graph.as_default():
                predicted_class = self.test_model.predict_classes(tl_img_data, verbose=False)

            if int(predicted_class) == 2:
                state = TrafficLight.YELLOW
                continue
            elif int(predicted_class) == 1:
                state = TrafficLight.GREEN
                continue
            elif int(predicted_class) == 3:
                state = TrafficLight.RED
                break
            else:
                continue

        return state
