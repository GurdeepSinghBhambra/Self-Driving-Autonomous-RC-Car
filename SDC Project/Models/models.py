__author__ = "Gurdeep"

import tensorflow as tf
import random
import numpy as np
random_seed = 42
if(tf.__version__ == '1.15.0'):
    tf.set_random_seed(random_seed)
elif(tf.__version__ == '2.0.0'):
    tf.random.set_seed(random_seed)
else:
    print("Invalid tensorflow version, tf.__version__ =", tf.__version__)
    exit(0)
np.random.seed(random_seed)
random.seed(random_seed)
 
from PrepareDataset import PrepareDataset as ppd 
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import TensorBoard
from datetime import datetime
import os
#import h5py

def prepareData(directory):
    p = ppd(directory)
    df = p.createMasterDataframe()
    p.saveFile(df=df, mode='master')
    return p

def model():
    clf = Sequential()

    clf.add(Conv2D(32, (5, 5), input_shape=(480, 640, 1), activation='relu'))
    clf.add(MaxPooling2D(pool_size=(4, 4)))

    clf.add(Conv2D(16, (3, 3), activation='relu'))
    clf.add(MaxPooling2D(pool_size=(2, 2)))

    clf.add(Flatten())

    clf.add(Dense(units=512, activation='relu'))
    clf.add(Dense(units=128, activation='relu'))
    clf.add(Dense(units=8, activation='softmax'))

    clf.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return clf

#These load and save model functions are not finalized yet, they work fine but may not be the best option
def saveModel(model):
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model.h5")
    print("Saved model to disk")

#here if you load a model and want to train it then make sure you compile 
#the function in same way you did earlier with the model
def loadModel():
    json_file = open('tried models\\model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("tried models\\model.h5")
    #compile only when training or metrics and optimizer is required
    #loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return loaded_model

#Small dataset, outdated. dont use this
def main_old_dataset():
    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)

    p = prepareData('data')
    #p.train_test_split(validation=True, test_size=0.10, validation_size=0.15, random_state=42, stratify=True)
    #p = ppd('data')
    train_frames, train_decisions = p.getFileChunkArray(df=p.openFile(mode='train'), gray=True, dtype='float32', reshape_image=(480, 640, 1), a=0, b=1, to_categorical=True, cat_dtype='float32')
    test_frames, test_decisions = p.getFileChunkArray(df=p.openFile(mode='test'), gray=True, dtype='float32', reshape_image=(480, 640, 1), a=0, b=1, to_categorical=True, cat_dtype='float32')
    val_frames, val_decisions = p.getFileChunkArray(df=p.openFile(mode='validation'), gray=True, dtype='float32', reshape_image=(480, 640, 1), a=0, b=1, to_categorical=True, cat_dtype='float32')

    clf = model()

    logdir = "log\\scalars\\"+datetime.now().strftime("%Y%m%d_%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=logdir)
    clf.fit(x=train_frames,
            y=train_decisions, 
            epochs=32, 
            callbacks=[tensorboard_callback], 
            batch_size=32, 
            verbose=1, 
            validation_data=(val_frames, val_decisions), 
            use_multiprocessing=True)

    _, test_accuracy = clf.evaluate(x=test_frames, y=test_decisions)
    print("\n\n\nTest Accuracy = {}\n".format(test_accuracy))

    saveModel(clf)


#for current big data sets but not yet tested for new collage one
def main():

    #p = prepareData('C:\\Users\\gurde\\Documents\\Anaconda\\SDC laptop\\data')
    #p.train_test_split(validation=True, test_size=0.15, validation_size=0.25, random_state=42, stratify=True)

    batch_size = 32

    p = ppd("C:\\Users\\gurde\\Documents\\Anaconda\\SDC laptop\\data")
    #don't use float, try to do with the original image dtype ('uint8') so that no time will be
    #wasted in normalization
    #this was for the previous model
    #also try for image dim of 320, 240
    train_gen = p.datasetGenerator(mode='train', 
                                   batch_size=batch_size, 
                                   gray=True, 
                                   dtype='float32', 
                                   reshape_image=(480, 640, 1), 
                                   a=0, 
                                   b=1, 
                                   to_categorical=True, 
                                   cat_dtype='float32')
    test_gen = p.datasetGenerator(mode='test', 
                                  batch_size=batch_size, 
                                  gray=True, 
                                  dtype='float32', 
                                  reshape_image=(480, 640, 1), 
                                  a=0, 
                                  b=1, 
                                  to_categorical=True, 
                                  cat_dtype='float32')
    val_gen = p.datasetGenerator(mode='validation', 
                                 batch_size=batch_size, 
                                 gray=True, 
                                 dtype='float32', 
                                 reshape_image=(480, 640, 1), 
                                 a=0, 
                                 b=1, 
                                 to_categorical=True, 
                                 cat_dtype='float32')

    df = p.shapeAccordingToBatchSize(mode='train', batch_size=batch_size)
    train_steps_per_epoch = p.getFileRows(mode=None, df=df)/batch_size

    df = p.shapeAccordingToBatchSize(mode='validation', batch_size=batch_size)
    val_steps_per_epoch = p.getFileRows(mode=None, df=df)/batch_size

    df = p.shapeAccordingToBatchSize(mode='test', batch_size=batch_size)
    test_steps_per_epoch = p.getFileRows(mode=None, df=df)/batch_size

    try:
        os.mkdir("log")
        os.mkdir("log\\scalars")
    except Exception:
        pass

    logdir = "log\\scalars\\"+datetime.now().strftime("%Y%m%d_%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=logdir)
    
    clf = model()
    print("\n\nStarting now\n")

    clf.fit_generator(generator=train_gen, 
                      steps_per_epoch=train_steps_per_epoch,
                      epochs=16,
                      verbose=1,
                      callbacks=[tensorboard_callback],
                      validation_data=val_gen,
                      validation_steps=val_steps_per_epoch,
                      use_multiprocessing=False,
                      shuffle=True)


    print("\n\n Evaluating now ... \n")
    test_accuracy = clf.evaluate_generator(generator=test_gen, 
                                           steps=test_steps_per_epoch,
                                           use_multiprocessing=False)

    print("\n\n\nTest Accuracy = {}\n".format(test_accuracy[-1]))

    saveModel(clf)

def checkmain():

    p = ppd('../data')
    #df = p.createMasterDataframe()
    df = p.getChunkOfFile(start_index=17000, end_index=17001, mode='master')
    frames, decisions = p.getFileChunkArray(df=df, 
                                            gray=True, 
                                            reshape_image=(480, 640, 1), 
                                            dtype='float32', 
                                            a=0, 
                                            b=1, 
                                            to_categorical=True, 
                                            cat_dtype='float32')

    clf = loadModel()
    clf.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print("predicted:", p.categorical2Int(clf.predict(frames)))
    print("ground truth:", p.categorical2Int(decisions))



def new_model():

