import numpy as np
from skimage import io
import cv2
import tensorflow as tf
import math

shotsNb = 2
imgNb = 100  # nb image par shot
batchSize = 2  # nb d'image de chaque shot dans un batch
batchNb = imgNb//batchSize
imgH = 1080
imgW = 1920
imgChannels = 3
splitImg = 3

# FUNCTION TO SPLIT AN IMAGE INTO PIECES


def splitInputImgFunc(inpImg, splitImg):
    newShpImg = np.zeros((splitImg * splitImg, int(imgH / splitImg), int(imgW / splitImg), 3), dtype=np.float32)
    for i in range(splitImg):
        for j in range(splitImg):
            newShpImg[i*splitImg + j, :, :, :] = inpImg[i * int(imgH / splitImg):(i + 1) * int(imgH / splitImg), j * int(imgW / splitImg):(j + 1) * int(imgW / splitImg), :] / 255
    return newShpImg


def splitOutputImgFunc(inpImg, splitImg):
    newShpImg = np.zeros((splitImg * splitImg, int(imgH / splitImg), int(imgW / splitImg), 1), dtype=np.float32)
    for i in range(splitImg):
        for j in range(splitImg):
            newShpImg[i*splitImg + j, :, :, 0] = inpImg[i * int(imgH / splitImg):(i + 1) * int(imgH / splitImg), j * int(imgW / splitImg):(j + 1) * int(imgW / splitImg)] / 255
    return newShpImg


# Creates Inputs and Outputs variables

inputs = np.zeros((batchSize*shotsNb*splitImg*splitImg, int(imgH / splitImg), int(imgW / splitImg), imgChannels), dtype=np.float32)
outputs = np.zeros((batchSize*shotsNb*splitImg*splitImg, int(imgH / splitImg), int(imgW / splitImg), 1), dtype=np.float32)

# CONV NEURAL NET SETUP

n_epochs = 100

# Conv Layer 1
filter_size1 = 5
num_filters1 = 32

# Conv Layer 2
filter_size2 = 1
num_filters2 = 32

# Conv Layer 3
filter_size3 = 3
num_filters3 = 16

# Conv Layer 4
filter_size4 = 1
num_filters4 = 1

learning_rate = tf.placeholder(dtype=tf.float32, shape=[])
X = tf.placeholder(dtype=tf.float32, shape=(None, int(imgH / splitImg), int(imgW / splitImg), imgChannels))
Y = tf.placeholder(dtype=tf.float32, shape=(None, int(imgH / splitImg), int(imgW / splitImg), 1))

# with tf.device('/gpu:0'):
# with tf.device('/cpu:0'):

CL1 = tf.layers.conv2d(X, filters=num_filters1, kernel_size=filter_size1, strides=[1, 1], padding="SAME",
                       activation=tf.nn.relu)

SPLITPOOL = tf.split(CL1, num_or_size_splits=16, axis=3)

index = 0
for layer in SPLITPOOL:
    MAXPOOL = tf.reduce_max(layer, axis=3, keepdims=True)
    if index == 0:
        GLOBPOOL = MAXPOOL
    else:
        TEMPPOOL = tf.concat([GLOBPOOL, MAXPOOL], 3)
        GLOBPOOL = TEMPPOOL
    index += 1

CL2 = tf.layers.conv2d(GLOBPOOL, filters=num_filters2, kernel_size=filter_size2, strides=[1, 1], padding="SAME",
                       activation=tf.nn.relu)
CL3 = tf.layers.conv2d(CL2, filters=num_filters3, kernel_size=filter_size3, strides=[1, 1], padding="SAME",
                       activation=tf.nn.relu)
CL4 = tf.layers.conv2d(CL3, filters=num_filters4, kernel_size=filter_size4, strides=[1, 1], padding="SAME",
                       activation=tf.sigmoid)

error = CL4 - Y
mse = tf.reduce_mean(tf.square(error), name="mse")

optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)

trainingOp = optimizer.minimize(mse)
saver = tf.train.Saver()
mse_summary = tf.summary.scalar("MSE", mse)


# RUN THE GRAPH

config = tf.ConfigProto()
config.log_device_placement = True
config.allow_soft_placement = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9
sess = tf.Session(config=config)
sess.run(tf.initialize_all_variables())
File_Writer = tf.summary.FileWriter('C:\\AI\\TENSORGREEN\\graph01', sess.graph)

for epoch in range(n_epochs):

    for batch_index in range(batchNb):

        if batch_index % 10 == 0:
            save_path = saver.save(sess, "C:\\AI\\TENSORGREEN\\MODEL\\trainingCheckPt.ckpt")

        # Read Inputs and Outputs Images in order

        for j in range(batch_index*batchSize, (batch_index+1)*batchSize):
            for i in range(shotsNb):
                imgPath = 'C:\\AI\\DATA\\INPUTS\\Data.%02d' % (i + 1) + '.RGB.%04d.png' % j
                print('Reading Path ' + imgPath)
                currImg = cv2.imread(imgPath, cv2.IMREAD_COLOR)
                currImg = cv2.cvtColor(currImg, cv2.COLOR_BGR2RGB)
                currSplit = splitInputImgFunc(currImg, splitImg)
                for k in range(splitImg * splitImg):
                    inputs[i + j - batch_index * batchSize + k, :, :, :] = currSplit[k, :, :, :]

                imgPath = 'C:\\AI\\DATA\OUTPUTS\\Data.%02d' % (i + 1) + '.ALPHA.%04d.png' % j
                print('Reading Path ' + imgPath)
                currImg = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
                currSplit = splitOutputImgFunc(currImg, splitImg)
                for k in range(splitImg * splitImg):
                    outputs[i + j - batch_index*batchSize + k, :, :, 0] = currSplit[k, :, :, 0]

        learning_rate_init = (0.001 + 0.25*(1-epoch/n_epochs))
        sess.run(trainingOp, feed_dict={X: inputs, Y: outputs, learning_rate: learning_rate_init})
        summary_str = mse_summary.eval(session=sess, feed_dict={X: inputs, Y: outputs})
        step = epoch*batchNb+batch_index
        File_Writer.add_summary(summary_str, step)

save_path = saver.save(sess, "C:\\AI\\TENSORGREEN\\MODEL\\finalModel.ckpt")

File_Writer.close()

sess.close()
