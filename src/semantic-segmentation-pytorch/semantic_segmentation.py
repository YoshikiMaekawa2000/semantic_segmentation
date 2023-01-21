#!/usr/bin/env python3
import rospy
import cv2
import sys, os
# os.getcwd()
# sys.path.append(os.path.join("/home/yoshiki/catkin_ws/src/semantic_segmentation/src/semantic-segmentation-pytorch/"))
sys.path.append(os.pardir)
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
# Our libs
from mit_semseg.dataset import TestDataset
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import colorEncode, find_recursive, setup_logger
from mit_semseg.lib.nn import user_scattered_collate, async_copy_to
from mit_semseg.lib.utils import as_numpy
from PIL import Image as PILIMAGE
from tqdm import tqdm
from mit_semseg.config import cfg

class SemanticSegmentation:
    def __init__(self):
        #init node
        self.node_name = "semantic_segmentation"
        rospy.init_node(self.node_name)

        #init_model
        self.bridge = CvBridge()
        self.cwd = "/home/amsl/catkin_ws/src/semantic_segmentation/src/semantic-segmentation-pytorch/"
        self.cfg_fpath = self.cwd + "config/ade20k-mobilenetv2dilated-c1_deepsup.yaml"
        # self.cfg_fpath = self.cwd + "config/ade20k-resnet50dilated-ppm_deepsup.yaml"
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

        self.image_sub = rospy.Subscriber("/CompressedImage", CompressedImage, self.image_callback, queue_size = 1)
        self.image_pub = rospy.Publisher("/segmentation", Image, queue_size=1)

        rospy.Timer(rospy.Duration(1.0), self.timerCallback)

    def timerCallback(self, event):


        dataset_test = TestDataset(
            self.input_image,
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
        try:

            np_arr = np.frombuffer(ros_image_compressed.data, np.uint8)
            input_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
            input_image = PILIMAGE.fromarray(input_image)
        except CvBridgeError as e:
            print(e)

        self.input_image = input_image

        # dataset_test = TestDataset(
        #     input_image,
        #     cfg.DATASET,
        #     )
        # loader_test = torch.utils.data.DataLoader(
        #     dataset_test,
        #     batch_size=cfg.TEST.batch_size,
        #     shuffle=False,
        #     collate_fn=user_scattered_collate,
        #     num_workers=5,
        #     drop_last=True)
        # self.test(loader_test)

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
        msg.header.frame_id = "camera"
        msg.step = msg.width*3
        self.image_pub.publish(msg)


    def test(self, loader):
        # segmentation_module.eval()

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

        #visualization
        self.visualize_result(pred)

if __name__=="__main__":
    SemanticSegmentation()
    rospy.spin()
