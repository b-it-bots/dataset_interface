#! /usr/bin/env python

import rospy
import cv2
import os
import argparse
import face_recognition

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String

current_img = None
bridge = CvBridge()
image_capture_sleep_time = 4.

def image_cb(msg):
    global current_img
    current_img = msg

def extract_face_image(image_array):
    try:
	top, right, bottom, left = face_recognition.face_locations(image_array)[0]
	rospy.loginfo('[face_collection_script] Successfully extracted face from person image.')
	return image_array[top:bottom, left:right]
    except IndexError:
	rospy.logwarn('[face_collection_script] Failed to extract face from person image!')
	return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Person image collection script')

    parser.add_argument('-p', '--person_name', type=str,
                        help='The name of the person.')
    parser.add_argument('-n', '--num_images', type=int,
                        default=10,
                        help='The number of images to be taken.')
    parser.add_argument('-d', '--dir_path', type=str,
                        help='The path where the images will be saved.')
    parser.add_argument('-t', '--image_topic', type=str,
                        default='/hsrb/head_rgbd_sensor/rgb/image_raw',
                        help='The topic on which the images are published.')

    person_name = parser.parse_args().person_name
    num_images = int(parser.parse_args().num_images)
    dir_path = parser.parse_args().dir_path
    image_topic = parser.parse_args().image_topic
    
    person_dir_path = os.path.join(dir_path, person_name)
    try:
        os.mkdir(person_dir_path)
    except OSError:
        pass

    rospy.init_node('collect_face_images_test')
    rospy.Subscriber(image_topic, Image, image_cb)
    say_pub = rospy.Publisher('/say', String, queue_size=1)
    
    while current_img is None:
        rospy.sleep(0.1)

    for i in range(num_images):
        rospy.loginfo('[face_collection_script] Taking next image...')
        extracted_face = None
        while extracted_face is None:
            img_cv2 = bridge.imgmsg_to_cv2(current_img)
            extracted_face = extract_face_image(img_cv2)
        rospy.loginfo('[face_collection_script] Successfully extracted face {}!'.format(i+1))
        say_pub.publish('Successfully extracted face ' + str(i+1) + '!')
        
        cv2.imwrite(os.path.join(person_dir_path, person_name + '_' + str(i) + '.jpg'), extracted_face)
        rospy.sleep(image_capture_sleep_time)
