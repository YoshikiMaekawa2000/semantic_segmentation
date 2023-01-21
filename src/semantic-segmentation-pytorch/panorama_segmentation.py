#!/usr/bin/env python3
import rospy
import cv2
import sys, os
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from cv_bridge import CvBridge, CvBridgeError

import os
import argparse
from distutils.version import LooseVersion
# Numerical libs
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
import csv
import py360convert
# Our libs
from mit_semseg.dataset import TestDataset
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import colorEncode, find_recursive, setup_logger
from mit_semseg.lib.nn import user_scattered_collate, async_copy_to
from mit_semseg.lib.utils import as_numpy
from PIL import Image as PILIMAGE
from mit_semseg.config import cfg

class SemanticSegmentation:
    def __init__(self):

        self.node_name = "semantic_segmentation"
        rospy.init_node(self.node_name)

        self.cwd = "/home/amsl/ros_catkin_ws/src/semantic_segmentation/src/semantic-segmentation-pytorch/"
        self.cfg_fpath = self.cwd + "config/ade20k-resnet50dilated-ppm_deepsup.yaml"
        self.gpu = 0

        cfg.merge_from_file(self.cfg_fpath)
        cfg.MODEL.arch_encoder = cfg.MODEL.arch_encoder.lower()
        cfg.MODEL.arch_decoder = cfg.MODEL.arch_decoder.lower()
        # absolute paths of model weights
        cfg.MODEL.weights_encoder = os.path.join(
            self.cwd, cfg.DIR, 'encoder_' + cfg.TEST.checkpoint)
        cfg.MODEL.weights_decoder = os.path.join(
            self.cwd, cfg.DIR, 'decoder_' + cfg.TEST.checkpoint)

        assert os.path.exists(cfg.MODEL.weights_encoder) and \
            os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"

        self.segmentation_module, self.colors, self.names = self.init_module()

        self.image_sub = rospy.Subscriber("/CompressedImage",CompressedImage, self.image_callback, queue_size = 1)
        self.image_pub = rospy.Publisher("/segmentation", Image, queue_size=1)

        rospy.Timer(rospy.Duration(1.0), self.timerCallback)

    def timerCallback(self, event):

        dataset_test = TestDataset(
            self.input_img_list,
            cfg.DATASET,
            )
        loader_test = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=cfg.TEST.batch_size,
            shuffle=False,
            collate_fn=user_scattered_collate,
            num_workers=5,
            drop_last=True)

        self.test(loader_test)

    def image_callback(self, ros_image_compressed):

        #CompressedImage to ndarray
        try:
            np_arr = np.frombuffer(ros_image_compressed.data, np.uint8)
            input_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
            input_image = PILIMAGE.fromarray(input_image)
        except CvBridgeError as e:
            print(e)

        #equirectangular to cubemap
        # cube = py360convert.e2c(input_image, 320)
        # cube_h = py360convert.cube_dice2h(cube)
        # cube_list = py360convert.cube_h2list(cube_h)
        #
        # input_img_list=[]
        # for img in cube_list:
        #     input_img_list.append(PILIMAGE.fromarray(img))

        input_img_list=[]
        left= 0
        upper=0
        right = input_image.height
        lower = input_image.height
        for num in range(4):
            input_img_list.append(input_image.crop((left, upper, right, lower)))
            left += input_image.height
            right += input_image.height

        self.input_img_list=input_img_list

        # msg=Image()
        # msg.header.stamp=self.Time
        # msg.height = re_e.shape[0]
        # msg.width = re_e.shape[1]
        # msg.encoding="rgb8"
        # msg.data = re_e.tobytes()
        # msg.header.frame_id = "camera_color_optical_frame"
        # msg.step = msg.width*3
        # self.image_pub.publish(msg)



    def init_module(self):
        torch.cuda.set_device(self.gpu)
        # Network Builders
        net_encoder = ModelBuilder.build_encoder(
            arch=cfg.MODEL.arch_encoder,
            fc_dim=cfg.MODEL.fc_dim,
            weights=cfg.MODEL.weights_encoder)
        net_decoder = ModelBuilder.build_decoder(
            arch=cfg.MODEL.arch_decoder,
            fc_dim=cfg.MODEL.fc_dim,
            num_class=cfg.DATASET.num_class,
            weights=cfg.MODEL.weights_decoder,
            use_softmax=True)

        crit = nn.NLLLoss(ignore_index=-1)

        segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)

        segmentation_module.cuda()
        segmentation_module.eval()

        colors = loadmat(self.cwd + 'data/color150.mat')['colors']
        names = {}
        with open(self.cwd + 'data/object150_info.csv') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                names[int(row[0])] = row[5].split(";")[0]

        return segmentation_module, colors, names


    def visualize_result(self, pred):
        # print predictions in descending order
        pred = np.int32(pred)
        pixs = pred.size
        uniques, counts = np.unique(pred, return_counts=True)

        # print(pred.shape)
        # colorize prediction
        pred_color = colorEncode(pred, self.colors).astype(np.uint8)


        # aggregate images and save
        pred_color=PILIMAGE.fromarray(pred_color)
        msg=Image()
        msg.header.stamp=rospy.Time.now()
        msg.height = pred_color.height
        msg.width = pred_color.width
        msg.encoding="rgb8"
        msg.data = np.array(pred_color).tobytes()
        msg.header.frame_id = "camera_color_optical_frame"
        msg.step = msg.width*3
        self.image_pub.publish(msg)


    def test(self, loader):

        pred_list = []
        for batch_data in loader:
            # process data
            batch_data = batch_data[0]
            segSize = (batch_data['img_ori'].shape[0],
                       batch_data['img_ori'].shape[1])
            img_resized_list = batch_data['img_data']

            with torch.no_grad():
                scores = torch.zeros(1, cfg.DATASET.num_class, segSize[0], segSize[1])
                scores = async_copy_to(scores, self.gpu)

                for img in img_resized_list:
                    feed_dict = batch_data.copy()
                    feed_dict['img_data'] = img
                    del feed_dict['img_ori']
                    # del feed_dict['info']
                    feed_dict = async_copy_to(feed_dict, self.gpu)

                    # forward pass
                    pred_tmp = self.segmentation_module(feed_dict, segSize=segSize)
                    scores = scores + pred_tmp / len(cfg.DATASET.imgSizes)

                _, pred = torch.max(scores, dim=1)
                pred = as_numpy(pred.squeeze(0).cpu())
                pred_list.append(pred)

        result = np.concatenate([pred_list[0], pred_list[1], pred_list[2], pred_list[3]], 1)




        # cube_h = py360convert.cube_list2h(pred_list)
        #
        # # cube_h.expand_dims(np.zeros((3))
        # cube_h = np.expand_dims(cube_h, 2)
        # cubemap = py360convert.cube_h2dice(cube_h)
        # # pred = py360convert.c2e(cubemap, 320, 1280)
        # pred = py360convert.c2e(cubemap, 331, 1280)
        #
        # pred = np.squeeze(pred, 2)

        # visualization
        self.visualize_result(result)


if __name__=="__main__":
    SemanticSegmentation()
    rospy.spin()
