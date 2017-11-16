import numpy as np
import tensorflow as tf
import glob
import cv2


all =glob.glob('/media/torben/0F5F0A370F5F0A37/Dokumente/Unterlagen/Udacity/Self-Driving-Car-ND/Term3/CarND-Capstone/training/Github_cropped/simulator/red/*.jpg')
all.extend(glob.glob('/media/torben/0F5F0A370F5F0A37/Dokumente/Unterlagen/Udacity/Self-Driving-Car-ND/Term3/CarND-Capstone/training/Github_cropped/udacity-sdc/red/*.jpg'))
all.extend(glob.glob('/media/torben/0F5F0A370F5F0A37/Dokumente/Unterlagen/Udacity/Self-Driving-Car-ND/Term3/CarND-Capstone/training/Bosch_dataset_rgb/crop/red/*.png'))
all.extend(glob.glob('/media/torben/0F5F0A370F5F0A37/Dokumente/Unterlagen/Udacity/Self-Driving-Car-ND/Term3/CarND-Capstone/training/Github_cropped/simulator/yellow/*.jpg'))
all.extend(glob.glob('/media/torben/0F5F0A370F5F0A37/Dokumente/Unterlagen/Udacity/Self-Driving-Car-ND/Term3/CarND-Capstone/training/Github_cropped/udacity-sdc/yellow/*.jpg'))
all.extend(glob.glob('/media/torben/0F5F0A370F5F0A37/Dokumente/Unterlagen/Udacity/Self-Driving-Car-ND/Term3/CarND-Capstone/training/Bosch_dataset_rgb/crop/yellow/*.png'))
all.extend(glob.glob('/media/torben/0F5F0A370F5F0A37/Dokumente/Unterlagen/Udacity/Self-Driving-Car-ND/Term3/CarND-Capstone/training/Github_cropped/simulator/green/*.jpg'))
all.extend(glob.glob('/media/torben/0F5F0A370F5F0A37/Dokumente/Unterlagen/Udacity/Self-Driving-Car-ND/Term3/CarND-Capstone/training/Github_cropped/udacity-sdc/green/*.jpg'))
all.extend(glob.glob('/media/torben/0F5F0A370F5F0A37/Dokumente/Unterlagen/Udacity/Self-Driving-Car-ND/Term3/CarND-Capstone/training/Bosch_dataset_rgb/crop/green/*.png'))
labels = ['red', 'green', 'yellow']


shape = (1, 32, 32, 3)

# Testing

rand = np.random.randint(0,len(all))
img_test = cv2.imread(all[rand])
cv2.imshow('test',img_test)
cv2.waitKey(0)
image = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_CUBIC)
#img = np.reshape(cv2.normalize(image, image, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F), shape)
img = np.reshape(image, shape)



with tf.Session() as sess:
    # Load weights
    saver = tf.train.import_meta_graph('model.cpkt.meta')
    graph = tf.get_default_graph()
    fc3 = graph.get_tensor_by_name("op_to_restore:0")
    prediction = tf.nn.softmax(fc3)
    top3 = tf.nn.top_k(prediction, k=1)
    x = graph.get_tensor_by_name("x:0")
    saver.restore(sess, './model.cpkt')
    print('Model restored with latest weights...')
    top3_probability = sess.run(top3, feed_dict={x: img})
    print('Probabilities predicted...')

print(top3_probability)
print("Predicted for image : " + labels[top3_probability[1][0][0]])

