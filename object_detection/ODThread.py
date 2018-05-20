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
import motionctl
import settings
import subprocess

class ODThread:
    img_todo = {}
    img_done = {}

    MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
    MODEL_FILE = MODEL_NAME + '.tar.gz'
    DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
    PATH_TO_GRAPH = 'object_detection/' + MODEL_NAME + '/frozen_inference_graph.pb'
    PATH_TO_LABELS = os.path.join('object_detection','data', 'mscoco_label_map.pbtxt')
    NUM_CLASSES = 90

    # if none of the cameras need processing,
    #   the thread will sleep this long before checking again
    #   This is to avoid unnecessary resource consumption
    INACTIVE_SLEEP_TIME = 0.5

    def __init__(self):
        self.stopped=False

        for camera_id in config.get_camera_ids():
            ODThread.img_todo[camera_id] = None
            ODThread.img_done[camera_id] = None

        #TF variables
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(ODThread.PATH_TO_GRAPH, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        self.label_map = label_map_util.load_labelmap(ODThread.PATH_TO_LABELS)
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=ODThread.NUM_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)
        self.sess = tf.Session(graph=self.detection_graph)
        self.proc_im = None

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return

            od_active = False
            for camera_id in config.get_camera_ids():
                #logging.debug('processing camera %s'%camera_id)
                camera_config = config.get_camera(camera_id)
                if camera_config['@motion_detection']:
                    if motionctl.is_motion_detected(camera_id):
                        #logging.debug(camera_config)
                        #im_bytes = ODThread.img_todo[camera_id]
                        im_bytes = mjpgclient.get_jpg(camera_id)
                        #ODThread.img_todo[camera_id]=None
                        if im_bytes is not None:
                            # OD has been requested for this camera
                            od_active = True
                            im = self.bytes_to_np(im_bytes)
                            self.process_image(im)
            if not od_active:
                time.sleep(ODThread.INACTIVE_SLEEP_TIME)

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
        imPil = Image.open(bio)

        (im_width, im_height) = imPil.size
        im_np = np.array(imPil.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
        return im_np

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
            #sio = StringIO.StringIO()
            #image.save(sio, format='JPEG')

        end_t = time.time()
        logging.debug('%.3f seconds'%(end_t-start_t))

        thresh = 0.5
        objects = [{'name': self.category_index[classes[0][i]]['name'],'score':scores[0][i],'box':boxes[0][i]} for i in range(len(classes[0])) if scores[0][i]>=thresh]

        logging.debug(objects)

        #command = "/home/jorey/Code/Python/motioneye/motioneye/scripts/relayevent.sh"
        #on_event_start /home/jorey/Code/Python/motioneye/motioneye/scripts/relayevent.sh "../config/motioneye.conf" start %t; /usr/bin/python /home/jorey/Code/Python/motioneye/motioneye/meyectl.pyc sendmail -c ../config/motioneye.conf 'smtp.gmail.com' '587' 'boochcam@gmail.com' 'echodelta#1' 'True' 'boochcam@gmail.com' 'boochcam@gmail.com' 'motion_start' '%t' '%Y-%m-%dT%H:%M:%S' '0'
        #p = subprocess.Popen(command, stderr=subprocess.STDOUT, stdout=subprocess.PIPE)

        for o in objects:
            if o['name']=='person':
                logging.debug('person detected')
                center_x = (o['box'][0]+o['box'][2])/2
                center_y = (o['box'][1]+o['box'][3])/2
                angle_x = center_x*57.62/2
                if center_x < .5:
                    angle_x*=-1;
                angle_y = center_y*46.05/2
                if center_y < .5:
                    angle_y*=-1;



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
