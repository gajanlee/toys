#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   model.py
@Time    :   2019/05/25 14:10:26
@Author  :   gajanlee 
@Version :   1.0
@Contact :   lee_jiazh@163.com
@Desc    :   None
'''

import tensorflow as tf
import os
from PIL import Image

def int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list = tf.train.Int64List(value=value))

def bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value=[value]))


def convert_example(image_buffer, label, text, height, width):
    image_format = b"JPEG"
    channels = 3
    color_space = b"RGB"
    text = bytes(text, encoding="utf-8")

    example = tf.train.Example(features=tf.train.Features(feature={
        "image/height": int64_feature(height),
        "image/width": int64_feature(width),
        "image/colorspace": bytes_feature(color_space),
        "image/channels": int64_feature(label),
        "image/class/label": int64_feature(label),
        "image/class/text": bytes_feature(text),
        "image/format": bytes_feature(image_format),
        # "image/filename": bytes_feature(os.path.basename(filename)),
        "image/encoded": bytes_feature(image_buffer),
    }))
    return example

def main():
    rootdir = "./train/"
    writer = tf.python_io.TFRecordWriter("./data.tfrecords")
    for filename in os.listdir(rootdir):
        if not filename.endswith("jpg"): continue

        label, index, _ = filename.split(".")
        if int(index) > 10: continue
        
        print(filename)

        label = 0 if label == "cat" else 1

        image = Image.open(rootdir + filename)
        width, height = image.size
        image_buffer = image.tobytes()

        example = convert_example(image_buffer, label, index, width, height)

        writer.write(example.SerializeToString())
    
    writer.close()

if __name__ == "__main__":
    main()