import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential


# import tensorflow.keras.backend as K
#
# config = tf.ConfigProto(device_count={'GPU': 0})
# K.set_session(tf.Session(config=config))
from tensorflow.keras.models import load_model

# devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(devices[0], True)

base_dir = os.path.join(os.getcwd(), 'Resources')

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# print(train_dir)
# print(validation_dir)
# dataset = tf.keras.datasets.mnist
#
# #### train - test - split ####
# (X_train, y_train), (X_test, y_test) = dataset.load_data()
#
#
# #### normalize value to b/w 0and1 ###
# X_train= X_train/255.0
# X_test= X_test/255.0


### CNN (BATCH , HEIGHT, WIDTH, 1)
#### ANN (BATCH_SIZE, FEATURES)
#### FEATURES = WIDTH * HEIGHT
#### reshape array to fit in network ####

# print(X_train.shape)
#
# X_train = X_train.reshape(60000, 28, 28, 1)
# X_test = X_test.reshape(60000, 28, 28, 1)
# X_train = np.expand_dims(X_train, axis=-1)
# X_test = np.expand_dims(X_test, axis=-1)
#
# print(X_train.shape)

# (batch_size, height, width, 1)
#### ANN ########


from tensorflow.keras.layers import Dense, Dropout

# 0-1

model = Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    # tf.keras.layers.MaxPooling2D(2, 2),
    # tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    # tf.keras.layers.MaxPooling2D(2, 2),
    # tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    # tf.keras.layers.MaxPooling2D(2, 2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),

    # 512 neuron hidden layer
    # tf.keras.layers.Dense(700, activation='relu'),
    tf.keras.layers.Dense(512),
    # tf.keras.layers.Dropout(0.2),
    # tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("relu"),
    tf.keras.layers.Dense(128),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("relu"),
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('cats') and 1 for the other ('dogs')
    tf.keras.layers.Dense(9, activation='softmax')
])

model.summary()
# model = Sequential()
# model.add(Dense(128, activation='relu'))
# # model.add(Dropout(0.2))
#
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.2))
#
# ## [0-9] ##
# model.add(Dense(10, activation='softmax'))

# model.compile('adam', 'sparse_categorical_crossentropy', metrics=['accuracy'])

model.compile(optimizer='adam',
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])
train_datagen = ImageDataGenerator(rescale=1.0 / 255.,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1.0 / 255.)

# --------------------
# Flow training images in batches of 20 using train_datagen generator
# --------------------
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=10,
                                                    color_mode="grayscale",
                                                    target_size=(28, 28))
# --------------------
# Flow validation images in batches of 20 using test_datagen generator
# --------------------
validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                        batch_size=10,
                                                        color_mode="grayscale",
                                                        target_size=(28, 28))

model.fit(train_generator, validation_data=validation_generator,
          steps_per_epoch=720 / 10,
          epochs=25,
          validation_steps=180 / 10,
          verbose=2)

# model.fit(X_train, y_train, epochs=10, batch_size=12, validation_split=0.1)


#### making prediction #######
# plt.imshow(X_test[1255].reshape(28,28), cmap='gray')
# plt.xlabel(y_test[1255])
# plt.ylabel(np.argmax(model.predict(X_test)[1255]))


model.save('digit_trained.h5')

##### open cv for capture and predicting through camera #####
'''
##### cv2


cap = cv2.VideoCapture(0)
while True:
    ret, img = cap.read()
    #img = cv2.flip(img, 1)
    img = img[200:400, 200:400]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow("gray_wind", gray)
    gray = cv2.resize(gray, (28, 28))
    #cv2.imshow('resized')
    gray = gray.reshape(1, 784)
    result = np.argmax(model.predict(gray))
    result = 'cnn : {}'.format(result)
    cv2.putText(img, org=(25,25), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, text= result, color=(255,0,0), thickness=1)
    cv2.imshow("image", img)

    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()
#plt.imshow(img)
'''

############  prediction via paints ##########
### glob
# run = False
# ix,iy = -1,-1
# follow = 25
# img = np.zeros((512,512,1))
#
# ### func
# def draw(event, x, y, flag, params):
#     global run,ix,iy,img,follow
#     if event == cv2.EVENT_LBUTTONDOWN:
#         run = True
#         ix, iy = x, y
#     elif event == cv2.EVENT_MOUSEMOVE:
#         if run == True:
#             cv2.circle(img, (x,y), 20, (255,255,255), -1)
#
#     elif event == cv2.EVENT_LBUTTONUP:
#         run = False
#         cv2.circle(img, (x,y), 20, (255,255,255), -1)
#         gray = cv2.resize(img, (28, 28))
#         gray = np.expand_dims(gray, axis=-1)
#         gray = np.expand_dims(gray, axis=0)
#         print(gray.shape)
#         result = np.argmax(model.predict(gray))
#         result = 'cnn : {}'.format(result)
#         cv2.putText(img, org=(25,follow), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, text= result, color=(255,0,0), thickness=1)
#         follow += 25
#     elif event == cv2.EVENT_RBUTTONDOWN:
#         img = np.zeros((512,512,1))
#         follow = 25
#
#
# ### param
# cv2.namedWindow('image')
# cv2.setMouseCallback('image', draw)
#
#
#
# while True:
#     cv2.imshow("image", img)
#
#     if cv2.waitKey(1) == 27:
#         break
#
# cv2.destroyAllWindows()

########## THANKS ##########
