from threading import Thread

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from io import BytesIO
#from matplotlib import pyplot as plt
from PIL import Image
import time

from utils import label_map_util
from utils import visualization_utils as vis_util
import logging
import config
import mjpgclient

class ODThread:
    od_list=[]

    MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
    MODEL_FILE = MODEL_NAME + '.tar.gz'
    DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
    PATH_TO_GRAPH = 'object_detection/' + MODEL_NAME + '/frozen_inference_graph.pb'
    PATH_TO_LABELS = os.path.join('object_detection','data', 'mscoco_label_map.pbtxt')
    NUM_CLASSES = 90

    def __init__(self):
        self.stopped=False
        #TF variables

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(ODThread.PATH_TO_GRAPH, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        self.label_map = label_map_util.load_labelmap(ODThread.PATH_TO_LABELS)
        logging.debug(self.label_map)
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=ODThread.NUM_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)
        self.sess = tf.Session(graph=self.detection_graph)
        #self.im = None
        self.proc_im = None
        #self.mjpg = mjpg
        #self.next_im = None
        ODThread.od_list.append(self)


    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            for camera_id in config.get_camera_ids():
                #self.process_image(mjpgclient.get_jpg(camera_id))
                #cam_conf = config.get_camera(camera_id)
                #w = cam_conf['width']
                #h=cam_conf['height']
                logging.debug('processing camera %s'%camera_id)
                im = self.bytes_to_np(mjpgclient.get_jpg(camera_id))
                if im is not None:
                    self.process_image(im)

    def read(self):
        # return the frame most recently read
        return self.proc_im

    def set_next(self,next_im):
        self.next_im=next_im

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

    def bytes_to_np(self,im):
        if im==None:
            return

        bio = BytesIO(im)
        try:
            imPil = Image.open(bio)
        except:
            logging.error("image is bad. do not want.")
            return
        (im_width, im_height) = imPil.size
        im_np = np.array(imPil.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
        return im_np
        #im = Image.open(bio)
        #im_np = np.frombuffer(bio.getbuffer())
        #im_np=np.fromstring(bio.read(),np.uint8)
        #logging.debug(im_np.size)

    def process_image(self,im,draw_box=False):
        #im = open('happysheep.jpg','rb').read()
        #im = self.mjpg._last_jpg
        start_t = time.time()
        image_np_expanded = np.expand_dims(im, axis=0)
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represents level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        # Actual detection.
        (boxes, scores, classes, num_detections) = self.sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})


        if draw_box:
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                im,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                self.category_index,
                use_normalized_coordinates=True,
                line_thickness=8)

        end_t = time.time()
        logging.debug('%.3f seconds'%(end_t-start_t))

        thresh = 0.5
        objects = [{'name': self.category_index[classes[0][i]]['name'],'score':scores[0][i]} for i in range(len(classes[0])) if scores[0][i]>=thresh]

        logging.debug(objects)
        for o in objects:
            if o['name']=='person':
                logging.debug('person detected')

    def mail(obj,confidence):
        import sendmail
        import tzctl
        import smtplib

        logging.debug('sending notification email')

        try:
            subject = sendmail.subjects['object_detected']
            message = sendmail.messages['object_detected']
            format_dict = {
                'camera': camera_config['@name'],
                'hostname': socket.gethostname(),
                'moment': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'object': obj,
                'confidence': confidence,
            }
            if settings.LOCAL_TIME_FILE:
                format_dict['timezone'] = tzctl.get_time_zone()

            else:
                format_dict['timezone'] = 'local time'

            message = message % format_dict
            subject = subject % format_dict

            old_timeout = settings.SMTP_TIMEOUT
            settings.SMTP_TIMEOUT = 10
            sendmail.send_mail(data['smtp_server'], int(data['smtp_port']), data['smtp_account'],
                               data['smtp_password'], data['smtp_tls'], data['from'], [data['addresses']],
                               subject=subject, message=message, files=[])

            settings.SMTP_TIMEOUT = old_timeout

            self.finish_json()

            logging.debug('notification email succeeded')

        except Exception as e:
            if isinstance(e, smtplib.SMTPResponseException):
                msg = e.smtp_error

            else:
                msg = str(e)

            msg_lower = msg.lower()
            if msg_lower.count('tls'):
                msg = 'TLS might be required'

            elif msg_lower.count('authentication'):
                msg = 'authentication error'

            elif msg_lower.count('name or service not known'):
                msg = 'check SMTP server name'

            elif msg_lower.count('connection refused'):
                msg = 'check SMTP port'

            logging.error('notification email failed: %s' % msg, exc_info=True)
            self.finish_json({'error': str(msg)})
