import numpy as np
import matplotlib.pyplot as plt
import random, pickle, cv2, os
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import time
import datetime
DATADIR = "./training_files"
CATEGORIES = ["Normal", "Abnormal"] 
IMG_SIZE = 50

def create_training_data(DATADIR, category, IMG_SIZE):
    training_data= []
    for category in CATEGORIES:  
        print(category)
        path = os.path.join(DATADIR,category) # ./drive/My Drive/PartA_DFU_Dataset/ + Normal
        class_num = CATEGORIES.index(category)
        count = 0
        for img in os.listdir(path)[1:]:  # have to skip the first file which is a .DSStore file
            count +=1
            print("Count", count)
            try:
                img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_COLOR ) # Reads and returns the image in array form 
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB) # cv2 uses BGR rather than RGB, need to convert
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB) # cv2 uses BGR rather than RGB, need to convert
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) # to make sure they are all the same size
                training_data.append([new_array, class_num])
            except Exception as e:
                print("here")
            pass
    return training_data


def reshape_data(training_data):
    random.shuffle(training_data)
    for sample in training_data[:10]:
        print(sample[1])
    x = []
    y = []

    for features,label in training_data:
        x.append(features)
        y.append(label)

    print(x[0].reshape(-1, IMG_SIZE, IMG_SIZE, 3))

    x = np.array(x).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    return x, y

def pickle_data(x,y):
    pickle_out = open("./pickle_files/x.pickle","wb")
    pickle.dump(x, pickle_out)
    pickle_out.close()

    pickle_out = open("./pickle_files/y.pickle","wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()


def generate_model():
    pickle_in = open("./pickle_files/x.pickle","rb")
    X = pickle.load(pickle_in)
    

    pickle_in = open("./pickle_files/y.pickle","rb")
    y = pickle.load(pickle_in)
    NAME = "Diabetes-CNN{}".format(int(time.time()))

    tensorboard = TensorBoard(log_dir="logs\{}".format(NAME))

    X = X/255.0

    model = Sequential()

    model.add(Conv2D(256, (3, 3), input_shape=X.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

    model.add(Dense(64))
    model.add(Activation('relu'))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    # model.fit(X, y, batch_size=32, epochs=10, validation_split=0.3)
    
    model.fit(X, y,
          batch_size=32,
          epochs=20,
          validation_split=0.3,
          callbacks=[tensorboard])
    return model
def train():
    training_data = create_training_data(DATADIR, CATEGORIES, IMG_SIZE)
    print(len(training_data))
    x,y = reshape_data(training_data)
    pickle_data(x,y)  
    model = generate_model()   
    s = datetime.datetime.now().isoformat()
    s =s.split(".")[0]
    s= s.replace(":", "-")
    # '2017-07-24T23:54:45.203788'
    model.save(os.path.join("./models","mpg_model_"+s+".h5")) 
    print("training ended")           



