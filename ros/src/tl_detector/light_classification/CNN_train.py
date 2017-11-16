import cv2
import glob
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

red_all =glob.glob('/media/torben/0F5F0A370F5F0A37/Dokumente/Unterlagen/Udacity/Self-Driving-Car-ND/Term3/CarND-Capstone/training/Github_cropped/simulator/red/*.jpg')
red_all.extend(glob.glob('/media/torben/0F5F0A370F5F0A37/Dokumente/Unterlagen/Udacity/Self-Driving-Car-ND/Term3/CarND-Capstone/training/Github_cropped/udacity-sdc/red/*.jpg'))
red_all.extend(glob.glob('/media/torben/0F5F0A370F5F0A37/Dokumente/Unterlagen/Udacity/Self-Driving-Car-ND/Term3/CarND-Capstone/training/Bosch_dataset_rgb/crop/red/*.png'))
yellow_all = (glob.glob('/media/torben/0F5F0A370F5F0A37/Dokumente/Unterlagen/Udacity/Self-Driving-Car-ND/Term3/CarND-Capstone/training/Github_cropped/simulator/yellow/*.jpg'))
yellow_all.extend(glob.glob('/media/torben/0F5F0A370F5F0A37/Dokumente/Unterlagen/Udacity/Self-Driving-Car-ND/Term3/CarND-Capstone/training/Github_cropped/udacity-sdc/yellow/*.jpg'))
yellow_all.extend(glob.glob('/media/torben/0F5F0A370F5F0A37/Dokumente/Unterlagen/Udacity/Self-Driving-Car-ND/Term3/CarND-Capstone/training/Bosch_dataset_rgb/crop/yellow/*.png'))
green_all = (glob.glob('/media/torben/0F5F0A370F5F0A37/Dokumente/Unterlagen/Udacity/Self-Driving-Car-ND/Term3/CarND-Capstone/training/Github_cropped/simulator/green/*.jpg'))
green_all.extend(glob.glob('/media/torben/0F5F0A370F5F0A37/Dokumente/Unterlagen/Udacity/Self-Driving-Car-ND/Term3/CarND-Capstone/training/Github_cropped/udacity-sdc/green/*.jpg'))
green_all.extend(glob.glob('/media/torben/0F5F0A370F5F0A37/Dokumente/Unterlagen/Udacity/Self-Driving-Car-ND/Term3/CarND-Capstone/training/Bosch_dataset_rgb/crop/green/*.png'))
labels = ['red', 'green', 'yellow']

# Reduce the sample size to reduce training time
features_red = []
features_green = []
features_yellow = []
red_selected= []
yellow_selected = []
green_selected = []

for file in red_all:
    image = cv2.imread(file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.shape[0] > 16 and image.shape[1] > 16:
        image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_CUBIC)
        features_red.append(image)
        red_selected.append(file)
for file in green_all:
    image = cv2.imread(file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.shape[0] > 16 and image.shape[1] > 16:
        image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_CUBIC)
        features_green.append(image)
        green_selected.append(file)
for file in  yellow_all:
    image = cv2.imread(file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.shape[0] > 16 and image.shape[1] > 16:
        image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_CUBIC)
        features_yellow.append(image)
        yellow_selected.append(file)

X = np.vstack((features_red, features_green,features_yellow)).astype(np.float64)
labels = ['red', 'green', 'yellow']
label_red = (np.ones(len(features_red))).transpose()*0
label_green = (np.ones(len(features_green))).transpose()*1
label_yellow = (np.ones(len(features_yellow))).transpose()*2
Y = np.hstack((label_red, label_green,label_yellow)).astype(int)


shape = (1, 32, 32, 3)

rand_state = np.random.randint(0, 100)
X, X_test, y, y_test = train_test_split(
    X, Y, test_size=0.1, random_state=rand_state)
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.1, random_state=rand_state)


X_train_grayscale = np.empty([len(X_train), 32, 32, 3])
X_test_grayscale = np.empty([len(X_test), 32, 32, 3])
X_valid_grayscale = np.empty([len(X_valid), 32, 32, 3])

# Converting Images to Grayscale and normalizing to [-1,1] via OpenCV
for i in range(0, len(X_train)):
    img = X_train[i]
    #X_train_grayscale[i] = np.reshape(cv2.normalize(img, img, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F), shape)
    X_train_grayscale[i] = np.reshape(img, shape)

for i in range(0, len(X_test)):
    img = X_test[i]
    X_test_grayscale[i] = np.reshape(img, shape)
    #X_test_grayscale[i] = np.reshape(cv2.normalize(img, img, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F), shape)

for i in range(0, len(X_valid)):
    img = X_valid[i]
    #X_valid_grayscale[i] = np.reshape(cv2.normalize(img, img, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F), shape)
    X_valid_grayscale[i] = np.reshape(img, shape)


X_train = X_train_grayscale
X_valid = X_valid_grayscale
X_test = X_test_grayscale



EPOCHS = 20
BATCH_SIZE = 128
mu = 0
sigma = 0.01

# Input
x = tf.placeholder(tf.float32, (None, 32, 32, 3), name='x')
y = tf.placeholder(tf.int32, (None), name='y')
one_hot_y = tf.one_hot(y, 3)

# Variables.
wc1 = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 16), mean=mu, stddev=sigma), name="wc1")
bc1 = tf.Variable(tf.zeros(16), name="bc1")
wc2 = tf.Variable(tf.truncated_normal(shape=(5, 5, 16, 32), mean=mu, stddev=sigma), name="wc2")
bc2 = tf.Variable(tf.zeros(32), name="bc2")
fc1_W = tf.Variable(tf.truncated_normal(shape=(800, 800), mean=mu, stddev=sigma), name="fc1_W")
fc1_b = tf.Variable(tf.zeros(800), name="fc1_b")
fc2_W = tf.Variable(tf.truncated_normal(shape=(800, 400), mean=mu, stddev=sigma), name="fc2_W")
fc2_b = tf.Variable(tf.zeros(400), name="fc2_b")
fc3_W = tf.Variable(tf.truncated_normal(shape=(400, 3), mean=mu, stddev=sigma), name="fc3_W")
fc3_b = tf.Variable(tf.zeros(3), name="fc3_b")

saver = tf.train.Saver()


def LeNet(x):
    # Input shape: [batch, in_height, in_width, in_channels]
    # Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    #
    # Determination of the filter shape:
    # [filter_height, filter_width, in_channels, out_channels]
    # out_height = ceil(float(in_height - filter_height + 1) / float(strides[1]))
    # out_width  = ceil(float(in_width - filter_width + 1) / float(strides[2]))
    #
    conv1 = tf.nn.conv2d(x, wc1, strides=[1, 1, 1, 1], padding='VALID')
    conv1 = tf.nn.bias_add(conv1, bc1)

    # Activation.
    conv1 = tf.nn.relu(conv1)

    # Pooling. Input = 28x28x6. Output = 14x14x16.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 2: Convolutional. Output = 10x10x32.
    conv2 = tf.nn.conv2d(conv1, wc2, strides=[1, 1, 1, 1], padding='VALID')
    conv2 = tf.nn.bias_add(conv2, bc2)

    # Activation.
    conv2 = tf.nn.relu(conv2)

    conv2 = tf.nn.local_response_normalization(conv2)

    # Pooling. Input = 10x10x32. Output = 5x5x32.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Flatten. Input = 5x5x32. Output = 800
    fc0 = tf.contrib.layers.flatten(conv2)

    # Layer 3: Fully Connected. Input = 800. Output = 800.
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b

    # Dropout Layer
    fc1 = tf.nn.dropout(fc1, keep_prob=0.5)

    # Activation.
    fc1 = tf.nn.relu(fc1)

    # Layer 4: Fully Connected. Input = 800. Output = 800.
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b

    fc2 = tf.nn.dropout(fc2, keep_prob=0.5)

    # Activation.

    fc2 = tf.nn.relu(fc2)

    #  Layer 5: Fully Connected. Input = 800. Output = 3.
    fc3 = tf.matmul(fc2, fc3_W) + fc3_b

    fc3 = tf.nn.relu(fc3, name="op_to_restore")

  #  fc3 = tf.nn.dropout(fc3, keep_prob=0.8)


    return fc3


from sklearn.utils import shuffle

# Training Pipeline
rate = 0.001
X_train, y_train = shuffle(X_train, y_train)

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
        training_accuracy = evaluate(X_train, y_train)
        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i + 1))
        print("Training Accuracy = {:.3f}".format(training_accuracy))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
    saver.save(sess, './model.cpkt')
    print("Model saved")


with tf.Session() as sess:
    saver.restore(sess, './model.cpkt')
    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))


#Test
logits = LeNet(x)
prediction = tf.nn.softmax(logits)
top3 = tf.nn.top_k(prediction, k=1)

# Read dataset

# Testing
#all = red_selected
all= (yellow_selected)
#all.extend(green_selected)
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
    saver.restore(sess, './model.cpkt')
    print('Model restored with latest weights...')
    top3_probability = sess.run(top3, feed_dict={x: img})
    print('Probabilities predicted...')

print(top3_probability)
print("Predicted for image : " + labels[top3_probability[1][0][0]])

