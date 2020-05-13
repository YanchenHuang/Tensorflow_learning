# Load pickled data
import pickle

DATASET_DIR = './dataset/german-pickled-dataset/'
LABEL_NAMES_FILE = DATASET_DIR + 'signnames.csv'

# Load training and testing data
training_file = DATASET_DIR + 'train.p'
testing_file = DATASET_DIR + 'test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']
print(type(X_train),'\n')
print(type(y_train),'\n')

from sklearn.utils import shuffle
print("shuffle the data")
X_train, y_train = shuffle(X_train, y_train, random_state=1)


from sklearn.model_selection import train_test_split

X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

print ("Updated Image Shape: {}".format(X_train[0].shape))

### 显示使用sklearn的train_set_split工具后的数据分类的多少
import numpy as np

# Number of training examples
n_train = len(X_train)

# Number of testing examples.
n_test = len(X_test)

# What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# How many unique classes/labels there are in the dataset.
sorted_classes, class_indices, class_counts = np.unique(y_train, return_index=True, return_counts=True)
n_classes = len(sorted_classes)


print("Number of training examples =", n_train)
print("Number of validation examples =", len(X_validation))
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

import tensorflow as tf

# One Hot Encoding for Y, because the loss（cross entropy）can handel only one-hot code
#对其进行one-hot编码，因为我们使用的kereas内置的cross-entropy仅仅可以处理one-hot（理由之一）
one_hot_y_train = tf.one_hot(y_train, n_classes)
one_hot_y_test = tf.one_hot(y_test, n_classes)
one_hot_y_validation = tf.one_hot(y_validation, n_classes)
print(type(one_hot_y_train),'\n')

#数据的可视化
import matplotlib.pyplot as plt#


# import cv2
# print("Open CV Version:", cv2.__version__)
# print("Matplotlib Version:", plt.__version__)

# Get ClassId -> SignName Mapping from LABEL_NAMES_FILE

labels = dict()

with open(LABEL_NAMES_FILE, 'r') as f:
    labels_data = f.readlines()

    for i, line in enumerate(labels_data):
        if i == 0:
            continue  # Skip Header

        labelid, labelname = line.strip().split(',')

        if int(labelid) not in labels:
            labels[int(labelid)] = labelname

print("Label \tCount (Freq)\t SignName")

for label, count in zip(sorted_classes, class_counts):
    print("%4d \t %4d \t\t %s" % (label, count, labels[label]))

print("Title Legend: [Index] SignName {Count}")

############################################Do the pre processing of Image data
###########################################数据的预处理
#使用cv2 进行grey scale
import cv2
def rgb_to_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

image=X_train[class_indices[0]]
print("RGB Shape", image.shape)
#
# gray = rgb_to_grayscale(image)
# plt.figure()
# plt.imshow(gray, cmap='gray')
# plt.show()


##标准化，随机选取训练集中的数据进行标准化，randomly choose one image to visualizing
import random

rand_idx = random.randrange(0, 10000)
# rand_idx = 5461

image = X_train[rand_idx]
print("RGB Shape1", image.shape)
gray = rgb_to_grayscale(image)
print("Gray Shape", gray.shape)


# plt.imshow(gray, cmap='gray')

def contrast_normalization(grayscale):
    dest = grayscale.copy()
    imgmin = grayscale.min()
    imgmax = grayscale.max()
    n_img=(grayscale-imgmin)/(imgmax-imgmin)
    return n_img


# Alternatively
#     dest = cv2.normalize(grayscale, dest, alpha=0, beta=255, norm_type=NORM_MINMAX)
#     return dest


# orig = gray.copy()
#
# plt.figure(figsize=(5, 5))
#
# plt.subplot(2, 2, 1)
# plt.imshow(orig, cmap='gray')
# plt.colorbar()
#
# print("Random Idx: %d" % rand_idx)
# print("Min, Max", orig.min(), orig.max())
#
# plt.subplot(2, 2, 2)
# normalized_gray = contrast_normalization(gray)
# print("Min, Max", normalized_gray.min(), normalized_gray.max())
# plt.imshow(normalized_gray, cmap='gray')
# plt.colorbar()
# plt.show()

#preprocess function，1. greyScale, 2. normalization([0-255] -> [0,1]), 3. Fomat change: Ndarray -> Tensor
#数据预处理函数，1.图像灰度化，2. 标准化， 3. 转array格式到tensor，为训练坐准备
def preprocess(input):
    L = len(input)
    print(L,'length\n')
    ImageSet=np.zeros((L,32,32))
    for i in range (0,L):
        ImageSetGrey=rgb_to_grayscale(input[i])
        ImageSet[i,:,:]=contrast_normalization(ImageSetGrey)
    ImageSet=ImageSet.reshape(len(ImageSet), 32, 32, 1)
    return tf.convert_to_tensor(ImageSet)


#preprocess for train- , validation- and test-data
#训练集，验证集，以及测试机的预处理
x_train=preprocess(X_train)
print('x min ',np.min(x_train))
print('x max ',np.max(x_train))
plt.figure()
plt.imshow(x_train[2,:,:,0],cmap='gray')
plt.show()
x_validation=preprocess(X_validation)
plt.figure()
plt.imshow(x_validation[2,:,:,0],cmap='gray')
plt.show()
x_test=preprocess(X_test)

##################################LeNet5 architecture LeNet5的结构

LOGDIR = "./logs/train/"
#一些参数预设
# Constants
EPOCHS =50
# BATCH_SIZE = 50
BATCH_SIZE = 128#

# n_input = 32*32*1   # We intend to convert 32*32 traffic sign images to grayscale
n_classes = 43      # There are 43 different signs to classify

mu = 0
sigma = 0.1 # Validation loss = 3.410; Validation accuracy = 0.103
# sigma = 1.0 # Validation loss = 3.729; Validation accuracy = 0.014 (1-2%)


#定义我的LeNet类，用于后续Model调用
#define my own LeNet class
class MyLeNet(tf.keras.Model):
    def __init__(self):
        super(MyLeNet, self).__init__()
        self.conv2d_1 = tf.keras.layers.Conv2D(filters=6,
                                               kernel_size=(3, 3), activation='relu',
                                               input_shape=(32, 32, 1))
        self.average_pool = tf.keras.layers.AveragePooling2D()
        self.conv2d_2 = tf.keras.layers.Conv2D(filters=16,
                                               kernel_size=(5, 5), activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.fc_1 = tf.keras.layers.Dense(120, activation='relu')
        self.fc_2 = tf.keras.layers.Dense(84, activation='relu')
        self.out = tf.keras.layers.Dense(43, activation='softmax')

    def call(self, input):
        x = self.conv2d_1(input)
        x = self.average_pool(x)
        x = self.conv2d_2(x)
        x = self.average_pool(x)
        x = self.flatten(x)
        x = self.fc_2(self.fc_1(x))
        return self.out(x)


#orininal version they used TF1.0, therefore multiple For loop is needed,
#but my version is based on TF 2.0 therefore the training session would based keras, simplified(hopfully).
#直接利用keras进行训练

learning_rate=0.01#设置初始学习率为0.01，配合后面指数衰减学习率，达到更好的效果
decay_rate=1e-5
Model=MyLeNet()
num_epochs =2

#定义学习率衰减函数，
#define decade learning rate
def scheduler(epoch):
    if epoch < num_epochs * 0.2:
        return learning_rate
    if epoch < num_epochs * 0.4:
        return learning_rate * 0.1
    if epoch < num_epochs * 0.6:
        return learning_rate * 0.01
    if epoch < num_epochs * 0.8:
        return learning_rate * 0.001
    return learning_rate * 0.0001
change_Lr = tf.keras.callbacks.LearningRateScheduler(scheduler)

print("OPTIMIZER\n")
#select optimizer，选择优化器
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, decay=decay_rate)
#checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#    weight_path, monitor='val_accuracy', verbose=1,
#    save_best_only=False, save_weights_only=True,
#    save_frequency=1 )
Model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.categorical_crossentropy,
    metrics=['accuracy']
)
print("Model fit\n")
print('is there any NaN in x_train? ',np.any(np.isnan(x_train)))
print('is there any NaN in y_train? ',np.any(np.isnan(y_train)))

training=Model.fit(x_train, one_hot_y_train, batch_size=BATCH_SIZE, validation_data=[x_validation,one_hot_y_validation], epochs=EPOCHS,callbacks=[change_Lr])#
print(training.history)
#To visualize the training process
def plot_history(training):
    plt.plot(training.history['loss'])
    plt.plot(training.history['val_loss'])
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['loss', 'val_loss'], loc='lower right')
    plt.show()

plot_history(training)
#evaluation
print("Evaluation start\n")
#Y_test=tf.cast(y_test, tf.float32)
loss, acc = Model.evaluate(x_test, one_hot_y_test,batch_size=BATCH_SIZE)
print("train model, accuracy:{:5.2f}%".format(100 * acc))

#save the model(parameter) in the same dictionary
print("Save the Model\n")
Model.save_weights('./save_weights/my_save_weights')
del Model#delate the Model for Next step

#reload the model, and do the evaluation again

# step8 重新创建模型
print("rebuild the Model\n")
model = MyLeNet()
model.compile(optimizer=optimizer,
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])

# load the weights
print("reload the weights\n")
model.load_weights('./save_weights/my_save_weights')

# test the model
loss, acc = model.evaluate(x_test, one_hot_y_test)
print("Restored model, accuracy:{:5.2f}%".format(100 * acc))
