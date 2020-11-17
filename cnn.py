
# Import
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.optimizers import SGD
from keras import backend as K

# ---------------------------------------------------------------------------------------------------------------------
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


# ---------------------------------------------------------------------------------------------------------------------

classifier = Sequential()

classifier.add(Convolution2D(filters=64, kernel_size=3, strides=1, input_shape=(256, 256, 3), activation="relu"))

classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Convolution2D(filters=64, kernel_size=3, strides=1, activation="relu"))

classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Convolution2D(filters=64, kernel_size=3, strides=1, activation="relu"))

classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Flatten())

classifier.add(Dense(units=1024, activation="relu"))
classifier.add(Dense(units=1024, activation="relu"))
classifier.add(Dense(units=1024, activation="relu"))
classifier.add(Dense(units=4, activation="softmax"))

# ---------------------------------------------------------------------------------------------------------------------

opt = SGD(lr=0.01)
classifier.compile(optimizer= opt, loss="categorical_crossentropy", metrics=["accuracy",f1_m,precision_m, recall_m])

# ---------------------------------------------------------------------------------------------------------------------

# Entraînement du CNN

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=15,
)

test_datagen = ImageDataGenerator(
    rescale=1. / 255,
)

training_set = train_datagen.flow_from_directory(
    "Home/code/filbi/CNNshoulder/dataset/training_set",
    target_size=(256, 256),
    batch_size=16,
    class_mode='categorical'
)

test_set = test_datagen.flow_from_directory(
    'Home/code/filbi/CNNshoulder/dataset/test_set',
    target_size=(256, 256),
    batch_size=16,
    class_mode='categorical'
)



classifier.fit_generator(
    training_set,
    steps_per_epoch=32,
    epochs=30,
    validation_data=test_set,
    validation_steps=10
)

loss, accuracy, f1_score, precision, recall = classifier.evaluate(training_set, test_set, verbose=0)


classifier.save('model_save/save.h5')
