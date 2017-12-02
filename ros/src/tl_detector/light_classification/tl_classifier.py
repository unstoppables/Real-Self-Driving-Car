from styx_msgs.msg import TrafficLight
import tensorflow as tf
import cv2
import numpy as np
from keras.models import load_model

class TLClassifier(object):
    def __init__(self):
        self.init_ok = False
        self.model = load_model('./light_classification/1-model-gen.h5')
        #self.model._make_predict_function()
        self.graph = tf.get_default_graph()
        self.init_ok = True



    def get_classification(self, image):
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        #return TrafficLight.RED

        # Preprocessing Inception

        #img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #img = cv2.resize(img, dsize=(400, 400))
        #img = (img / 255 - 0.5) * 2         # Inception v3 Preprocessing
        #X = np.reshape(img, (1, 400, 400, 3))

        #cv2.imshow('test',image)
        #cv2.waitKey(0)
       # img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(image, dsize=(400, 400))
        X = img.astype(np.float64) / 255
        X = np.reshape(X, (1, 400, 400, 3))

        if self.init_ok == True:
            with self.graph.as_default():
                pred = self.model.predict(X)
                top_probability = np.argmax(pred)

            if top_probability== 0:
                print("Red Light!")
                return TrafficLight.RED

            else:
                print("No Red Light!")
                return TrafficLight.UNKNOWN

        else:
            return TrafficLight.UNKNOWN