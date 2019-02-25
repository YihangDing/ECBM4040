import os
import numpy as np
import h5py
import random
import json
import pywt

from PIL import Image
import tensorflow as tf


#==========================================================================================================#
#================================        Spectral Transform        ==================================#
#==========================================================================================================#
def wavelet_transform(image):
    image_float = image.astype(float)
    gray = 0.2989 * image_float[:,:,0] + 0.5870 * image_float[:,:,1] + 0.1140 * image_float[:,:,2]
    layer1 = gray.astype(np.uint8)

    T = np.log((abs(np.fft.fftshift(np.fft.fft2(gray)))+1))
    T = (T-T.min())/(T.max()-T.min())*255
    layer2 = T.astype(np.uint8)

    coeffs2 = pywt.dwt2(layer1, 'haar')
    LL, (LH, HL, HH) = coeffs2
    
    if (LL.max()-LL.min() > 0):
        UL = (LL-LL.min())/(LL.max()-LL.min())*255
    else:
        UL = np.zeros((32,32),dtype='uint8')
    if (HL.max()-HL.min() > 0):
        DL = (HL-HL.min())/(HL.max()-HL.min())*255
    else:
        DL = np.zeros((32,32),dtype='uint8')
    if (LH.max()-LH.min() > 0):
        UR = (LH-LH.min())/(LH.max()-LH.min())*255
    else:
        UR = np.zeros((32,32),dtype='uint8')
    if (HH.max()-HH.min() > 0):
        DR = (HH-HH.min())/(HH.max()-HH.min())*255
    else:
        DR = np.zeros((32,32),dtype='uint8')


    UL = UL.astype(np.uint8)
    DL = DL.astype(np.uint8)
    UR = UR.astype(np.uint8)
    DR = DR.astype(np.uint8)

    layer3 = np.zeros((64,64),dtype='uint8')
    layer3[:32,:32] = UL
    layer3[:32,32:] = DL
    layer3[32:,:32] = UR
    layer3[32:,32:] = DR

    output = np.zeros((64,64,3),dtype='uint8')
    output[:,:,0] = layer1
    output[:,:,1] = layer2
    output[:,:,2] = layer3
    
    return output
#==========================================================================================================#
#================================        Decoder Information        =================================#
#==========================================================================================================#
def Encoder_TFrecord(img_file,struct_mat,WT_flag):
    index = int(img_file.split('/')[-1].split('.')[0]) - 1
    attrs = {}
    item = struct_mat['digitStruct']['bbox'][index].item()
    for key in ['label', 'left', 'top', 'width', 'height']:
        attr = struct_mat[item][key]
        values = [struct_mat[attr.value[i].item()].value[0][0]
                  for i in range(len(attr))] if len(attr) > 1 else [attr.value[0][0]]
        attrs[key] = values

    label_digits = attrs['label']
    length = len(label_digits)
    
    # According to the infomation from Data Extract & Analysis, the maximum of digit length is 6
    # But according to the info provided in the paper, the sample with 6 dight is limited and we choose to set 5 as the miximum
    # Here we use DIGIT 10 to represent there is no digit in the corresponding pos in the fig
    #According to the instruction of the mat file, the LABEL zero is expressed as 10
    digits = [10, 10, 10, 10, 10] 
    if length == 6:
        return None
    
    for idx, digit in enumerate(label_digits):
        digits[idx] = int(digit if digit != 10 else 0) 

    attrs_left, attrs_top, attrs_width, attrs_height = map(lambda x: [int(i) for i in x], [attrs['left'], attrs['top'], attrs['width'], attrs['height']])
    min_left, min_top, max_right, max_bottom = (min(attrs_left),min(attrs_top),max(map(lambda x, y: x + y, attrs_left, attrs_width)),
                                                max(map(lambda x, y: x + y, attrs_top, attrs_height)))
    center_x, center_y, max_side = ((min_left+max_right)/2.0, (min_top+max_bottom)/2.0, max((max_right-min_left), (max_bottom-min_top)))
    bbox_left, bbox_top, bbox_width, bbox_height = (center_x-max_side/2.0, center_y-max_side/2.0,max_side,max_side)
    cropped_left, cropped_top, cropped_width, cropped_height = (int(round(bbox_left - 0.15 * bbox_width)),int(round(bbox_top - 0.15 * bbox_height)),
                                                                int(round(bbox_width * 1.3)),
                                                                int(round(bbox_height * 1.3)))
    image = Image.open(img_file)
    image = image.crop([cropped_left, cropped_top, cropped_left+cropped_width, cropped_top+cropped_height])
    image = image.resize([64, 64])

    if WT_flag == 1:
        image = wavelet_transform(np.array(image)).tobytes()
    elif WT_flag == 0:
        image = np.array(image).tobytes()

    Processed_Data = tf.train.Example(features=tf.train.Features(feature={
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
        'length': tf.train.Feature(int64_list=tf.train.Int64List(value=[length])),
        'digits': tf.train.Feature(int64_list=tf.train.Int64List(value=digits))
    }))
    return Processed_Data


def Encoder_Control(source_file_option,tfrecords_option, WT_flag):
    
    num_examples = []
    writers = []
    for train_or_val in tfrecords_option:
        num_examples.append(0)
        writers.append(tf.python_io.TFRecordWriter(train_or_val))

    for image_folder, struct_file in source_file_option:
        img_files = tf.gfile.Glob(os.path.join(image_folder, '*.png'))
        total_cnt = len(img_files)
        print ('There are totally %d files for %s' % (total_cnt, image_folder))

        if total_cnt >= 33402:
            Label = np.random.rand(total_cnt)
            Label[Label>=0.1]=0
            Label[Label!=0]=1
        else:
            Label = np.zeros((total_cnt),dtype='int8')

        with h5py.File(struct_file, 'r') as struct_mat:
            for cnt_process, img_file in enumerate(img_files):
                if cnt_process%50 == 0:
                    print ('%d files have been processed' % (cnt_process))
                
                Processed_Data = Encoder_TFrecord(img_file, struct_mat, WT_flag)
                if Processed_Data is None:
                    pass
                else:
                    train_val_label = int(Label[cnt_process])
                    writers[train_val_label].write(Processed_Data.SerializeToString())
                    num_examples[train_val_label] += 1

    for writer in writers:
        writer.close()

    return num_examples


def encoder_main():
    structure_file = 'digitStruct.mat'
    train_dir, test_dir, extra_dir = './data/train', './data/test', './data/extra'
    train_tfrecords_file, val_tfrecords_file, test_tfrecords_file = './data/train.tfrecords', './data/val.tfrecords', './data/test.tfrecords'
    train_struct = train_dir+'/'+structure_file
    test_struct = test_dir+'/'+structure_file
    extra_struct = extra_dir+'/'+structure_file
    tfrecords_meta_file = './data/meta.json'

    extra_included = 1 # Extra Data included Flag: Default:1 (With application)
    if extra_included == 1:
        train_phase_source = [(train_dir, train_struct),(extra_dir, extra_struct)]
    elif extra_included == 0:
        train_phase_source = [(train_dir, train_struct)]
    test_phase_source = [(test_dir, test_struct)]

    WT_flag = 1 # Wazelet Transform Flag: Default:1(With application)

    print ('Processing Data for training and validation')
    [num_train, num_val] = Encoder_Control(train_phase_source, [train_tfrecords_file, val_tfrecords_file], WT_flag)
    print ('Processing Data for testing')
    [num_test] = Encoder_Control(test_phase_source, [test_tfrecords_file], WT_flag)

    with open(tfrecords_meta_file, 'w') as save_data:
        content = {
            'num_examples': {
                'train': num_train,
                'val': num_val,
                'test': num_test
            }
        }
        json.dump(content, save_data)

    print ('Info about data division has been saved to %s...' % tfrecords_meta_file)

#==========================================================================================================#
#================================        Decoder Information        =================================#
#==========================================================================================================#
def Decoder_TFrecord(tfrecords_file):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(tfrecords_file)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'length': tf.FixedLenFeature([], tf.int64),
            'digits': tf.FixedLenFeature([5], tf.int64)
        })

    image = tf.decode_raw(features['image'], tf.uint8)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.multiply(tf.subtract(image, 0.5), 2)
    image = tf.reshape(image, [64, 64, 3])
    image = tf.random_crop(image, [54, 54, 3])
    length = tf.cast(features['length'], tf.int32)
    digits = tf.cast(features['digits'], tf.int32)
    return image, length, digits

def Batch_Generation(tfrecords_path, num_examples, batch_size, shuffled):
    assert tf.gfile.Exists(tfrecords_path), '%s not found' % tfrecords_path

    tfrecords_file = tf.train.string_input_producer([tfrecords_path], num_epochs=None)
    image, length, digits = Decoder_TFrecord(tfrecords_file)

    min_queue_examples = int(0.4 * num_examples)
    if shuffled:
        image_batch, length_batch, digits_batch = tf.train.shuffle_batch([image, length, digits],batch_size=batch_size,num_threads=2,
                                                                         capacity=min_queue_examples + 3*batch_size,
                                                                         min_after_dequeue=min_queue_examples)
    else:
        image_batch, length_batch, digits_batch = tf.train.batch([image, length, digits],batch_size=batch_size,num_threads=2,
                                                                 capacity=min_queue_examples + 3*batch_size)
    return image_batch, length_batch, digits_batch
