from styx_msgs.msg import TrafficLight
import tensorflow as tf
import cv2
import numpy as np

class TLClassifier(object):
    def __init__(self):
        with tf.Session() as sess:
            # Load weights
            saver = tf.train.import_meta_graph('light_classification/model.cpkt.meta')
            graph = tf.get_default_graph()
            fc3 = graph.get_tensor_by_name("op_to_restore:0")
            prediction = tf.nn.softmax(fc3)
            top = tf.nn.top_k(prediction, k=1)
            x = graph.get_tensor_by_name("x:0")
            saver.restore(sess, './light_classification/model.cpkt')
        pass

    def get_classification(self, image):
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        shape = (1, 32, 32, 3)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # is it in BGR or RGB ?
        image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_CUBIC)
        img = np.reshape(image, shape)

 
        top_probability = self.sess.run(top, feed_dict={x: img})

        if top_probability[1][0][0]== 0:
            return TrafficLight.RED
        elif top_probability[1][0][0]== 1:
            return TrafficLight.GREEN
        elif top_probability[1][0][0]== 2:
            return TrafficLight.YELLOW
        else:
	    return TrafficLight.UNKNOWN
