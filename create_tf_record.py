import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
from tqdm import tqdm
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width / 4) # because image resized 1/4
        xmaxs.append(row['xmax'] / width / 4) # because image resized 1/4
        ymins.append(row['ymin'] / height / 4) # because image resized 1/4
        ymaxs.append(row['ymax'] / height / 4) # because image resized 1/4
        classes_text.append(b'pineapple')
        classes.append(1)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main():
    output_path = '/media/trivu/data/DataScience/CV/dua/data/tf_val.record'
    csv_path = '/media/trivu/data/DataScience/CV/dua/data/labels.csv'
    image_path = '/media/trivu/data/DataScience/CV/dua/data/images/'
    examples = pd.read_csv(csv_path)
    grouped = split(examples, 'filename')
    writer = tf.python_io.TFRecordWriter(output_path)
    num_ex = len(grouped)
    num_train = int(num_ex*0.8)
    # train_group = grouped[:num_train]
    val_group = grouped[num_train:]
    from shutil import copyfile
    for group in tqdm(val_group):
        copyfile(image_path+group.filename, '/media/trivu/data/DataScience/CV/dua/data/train_images/'+group.filename)
        # tf_example = create_tf_example(group, image_path)
        # writer.write(tf_example.SerializeToString())

    writer.close()
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    main()


# import os
# import random
# import logging
# import tensorflow as tf

# def create_tf_record(output_filename, examples, labels_dir):
#     for idx, example in examples:
#         if idx % 100 == 0:
#             logging.info('On image %d of %d', idx, len(examples))
#         xml_path = os.path.join(labels_dir, example + '.xml')
#         with tf.gfile.GFile(xml_path, 'r') as fid:
#             xml_str = fid.read()
#         xml = etree.fromstring(xml_str)
#         print(xml)
#         break
        
# def main():
#     output_dir = None
#     data_dir = None
#     image_dir = os.path.join(data_dir, 'images')
#     labels_dir = os.path.join(data_dir, 'labels')
#     labels = os.listdir(labels_dir)

#     random.seed(59)
#     random.shuffle(labels)
#     num_examples = len(labels)
#     num_train = int(0.8*num_examples)
#     train_examples = labels[:num_train]
#     val_examples = labels[num_train:]

#     logging.info('%d training and %d validation examples.',
#                len(train_examples), len(val_examples))
#     train_output_path = os.path.join(output_dir, 'pineapple_train.record')
#     val_output_path = os.path.join(output_dir, 'pineapple_val.record')
