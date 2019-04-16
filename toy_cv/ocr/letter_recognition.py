import time
import numpy as np
import os
from collections import namedtuple

from keras.utils import np_utils
from keras.models import load_model
from keras.layers import (Dense, Conv2D, MaxPooling2D, 
                                        Reshape, Dropout, Flatten)
from keras.models import Sequential
from keras.optimizers import Adam

from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform

from PIL import Image
from sklearn.metrics import f1_score, precision_score, recall_score

Dataset = namedtuple('Dataset', ['base_dir', "classes"])

WIDTH, HEIGHT = 20, 20
BATCH_SIZE = 64
EPOCHS = 8

# 记录时间的装饰器
def time_recoder(msg):
    def decorator(func):
        def wrapper(*args, **kwargs):
            time_start = time.time()
            elements = func(*args, **kwargs)
            time_elapsed = round(time.time() - time_start, 3)
            print(f"{msg}耗时：{time_elapsed}秒")
            return elements
        return wrapper
    return decorator

def image_file_count(dir):
    return sum(sum(1 for file in files if file.endswith(".jpg"))
        for _, _, files in os.walk(dir))

def get_image_pixels(filepath):
    img = Image.open(filepath)
    assert (WIDTH, HEIGHT) == img.size
    
    return [img.getpixel((w, h)) for w in range(20) for h in range(20)]
    
# Convert Image Files to Numpy Array
def convert_image_array(base_dir, classes):
    image_count = image_file_count(base_dir)
    num_classes = len(classes)

    image_array = np.zeros((image_count, WIDTH*HEIGHT))
    label_array = np.zeros(image_count)
    
    index = 0
    for category in range(num_classes):
        dir = f"{base_dir}/{category}/"
        for _, _, files in os.walk(dir):
            for file in files:
                image_array[index] = get_image_pixels(os.path.join(dir, file))
                label_array[index] = category
                index += 1
    return image_array, label_array, classes

@time_recoder("加载图片")
def get_task_data(task, mode="train"):
    if mode == "test": mode = "val"
    tasks = {
        "area": Dataset(f"./SimpleLPR/dataset/{mode}/area", "ABCDEFGHIJKLMNOPQRSTUVWXYZ",),
        "letter": Dataset(f"./SimpleLPR/dataset/{mode}/letter", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",),
        "province": Dataset(f"./SimpleLPR/dataset/{mode}/province", ("皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新"))
    }
    if task not in tasks.keys():
        raise Exception(f"{taks} is not valid task name.")

    x_array, y_array, classes = convert_image_array(tasks[task].base_dir, tasks[task].classes)
    # 不除以255训练不收敛
    x_array = x_array.reshape(-1, WIDTH, HEIGHT, 1) / 255
    y_array = np_utils.to_categorical(y_array, num_classes=len(tasks[task].classes))
    
    return x_array, y_array, classes


class SimpleModel(object):
    
    def __init__(self, task, classes, train=True):
        self.num_classes = len(classes)
        self.task = task
        self.classes = classes
        
        if not train:
            self.load_model()
        else:
            self.build_model()
        
        # self.model.summary()
    
    def build_model(self):
        model = Sequential()
        
        # Conv1, channel 16
        # shape: (-1, 10, 10, 16)
        model.add(Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding="same", 
                         data_format="channels_last", activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))
        
        # Conv2, channel 32
        # shape: (-1, 10, 10, 32)
        model.add(Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding="same", 
                         data_format="channels_last", activation="relu"))
        model.add(MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding="same"))
        
        # Full connection
        # shape: (-1, 512)
        model.add(Flatten())
        model.add(Dense(512, activation="relu"))
        
        # Dropout
        model.add(Dropout(rate=0.1))
        
        # Output layer
        model.add(Dense(self.num_classes, activation="softmax"))
        
        # Optimizer
        model.compile(optimizer=Adam(lr=1e-4),
                    loss="categorical_crossentropy",
                    metrics=['accuracy'])

        self.model = model

    def fit(self, x_train, y_train, x_eval=None, y_eval=None):
        if not x_eval or not y_eval:
            self.model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.2)
        else:
            self.model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(x_eval, y_eval))

    def predict(self, x_test, y_test=None):
        y_predict = self.model.predict_classes(x_test)
        y_predict = list(map(lambda id: self.classes[id], y_predict))
        print(f"predict is {y_predict}")

    def save(self):
        self.model.save("{}_{}".format(self.task, len(self.classes)))
        
    def load_model(self):
        with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
            self.model = load_model("{}_{}".format(self.task, len(self.classes)))

    def predict_score(self, x_test, y_test):
        y_label = np.argmax(y_test, axis=1)
        y_predict = self.model.predict_classes(x_test)
        # y_predict = np_utils.to_categorical(y_predict)
        f1 = f1_score(y_label, y_predict, average="macro")
        print(f"f1_score is {f1}")

if __name__ == "__main__":
    import sys
    mode = sys.argv[1]
    task_names = sys.argv[2:]
    #mode = "test"
    #task_names = ["area"]

    if mode == "train":
        for task_name in task_names:
            X_train, Y_train, classes = get_task_data(task_name, "train")
            X_test, Y_test, _ =  get_task_data(task_name, "test")

            taskModel = SimpleModel(task_name, classes)
            taskModel.fit(X_train, Y_train)
            taskModel.save()
            taskModel.predict_score(X_test, Y_test)
    elif mode == "test":
        for task_name in task_names:
            X_test, Y_test, classes = get_task_data(task_name, "test")
            taskModel = SimpleModel(task_name, classes, train=False)
            taskModel.predict_score(X_test, Y_test)
            taskModel.predict(X_test, Y_test)
    

        