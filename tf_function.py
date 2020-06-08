
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import os
import time
import tensorflow as tf
from io import StringIO
from PIL import Image
import tempfile 
import boto3
import label_map_util
import six.moves.urllib as urllib
import cv2
s3 = boto3.resource('s3')
MODEL_SOURCE = 'model-in/frozen_inference_graph.pb'
MODEL_LABEL_SOURCE = 'model-in/mscoco_label_map_nl.pbtxt'
MODEL_LOCAL = os.path.join(os.sep, 'tmp', 'model.pb')
MODEL_LABEL_LOCAL = os.path.join(os.sep, 'tmp', 'model_label.pbtxt')

print('Downloading Model from S3...')
print(MODEL_LOCAL + '      '+  MODEL_LABEL_LOCAL)
print(MODEL_SOURCE + '      '+  MODEL_LABEL_SOURCE)
print(os.environ['bucket_config'])
s3.Bucket(os.environ['bucket_config']).download_file(MODEL_SOURCE,MODEL_LOCAL)
s3.Bucket(os.environ['bucket_config']).download_file( MODEL_LABEL_SOURCE,MODEL_LABEL_LOCAL)

NUM_CLASSES = 90
label_map = label_map_util.load_labelmap(MODEL_LABEL_LOCAL)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

detection_graph = tf.Graph()

with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(MODEL_LOCAL, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

def run_inference_on_image(image):
    result = []
    if not tf.gfile.Exists(image):
        tf.logging.fatal('File does not exist %s', image)
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:

            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            max_boxes_to_draw = 20
            min_score_thresh = .5
            index = 0
            client = boto3.client('rekognition')
            myS3 = boto3.client('s3')
            image = Image.open(image)
            image_np = load_image_into_numpy_array(image)
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Actual detection.
                
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            # with classifier_graph.as_default():    
            
            im_height, im_width, im_dimension = image_np.shape
            answer = None
            scores_sq = np.squeeze(scores)
            boxes_sq = np.squeeze(boxes)
            classes_sq = np.squeeze(classes).astype(np.int32)
            if not max_boxes_to_draw:
                max_boxes_to_draw = boxes_sq.shape[0]
            for i in range(min(max_boxes_to_draw, boxes_sq.shape[0])):
                if scores_sq is None or scores_sq[i] > min_score_thresh:
                    box = tuple(boxes_sq[i].tolist())
                    ymin, xmin, ymax, xmax = box
                    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                            ymin * im_height, ymax * im_height)
                    print ('Tensorflow recognition done...')
                    print ('OpenCV ......saving image' )
                    tmp_up = tempfile.NamedTemporaryFile()
                    upload_path = str(tmp_up.name) + '.jpg'
                    image_crop = image_np[int(top):int(bottom),int(left):int(right)]
                    if image_crop.size != 0 :
                        imgname =  print(os.environ['save_folder']) + '/frame_' + str(index)  + '.jpg'
                        print('imgname  ' + imgname)
                        print('upload_path  ' + upload_path)
                        cv2.imwrite(upload_path,image_crop)
                        myS3.upload_file(upload_path, os.environ['bucket_test'], imgname)
                        
                        img_toAWS = cv2.imencode('.jpg', image_crop)[1].tostring()
                        rekresp = client.detect_labels(Image={'Bytes': img_toAWS}, MinConfidence=1)
                    
                        for label in rekresp['Labels']:
                            print (label['Name'] + ' : ' + str(label['Confidence']))
            result = 'Image recognition finished'                
            
    return result  

def lambda_handler(event, context):
    for record in event['Records']:
        bucket = record['s3']['bucket']['name']
        key = record['s3']['object']['key']

        print('Running Deep Learning example using Tensorflow library ...')
        print('Image to be processed, from: bucket [%s], object key: [%s]' % (bucket, key))

        # load image
        tmp = tempfile.NamedTemporaryFile()
        with open(tmp.name, 'wb') as f:
            s3.Bucket(bucket).download_file(key, tmp.name)
            tmp.flush()

            a = run_inference_on_image(tmp.name)

            print('run_inference_on_image return : ' + a)	
