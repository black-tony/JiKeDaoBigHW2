import uuid

import numpy as np
import time
import six
import math
import random
import paddle
import paddle.fluid as fluid
import logging
import xml.etree.ElementTree
import codecs
import json
from paddle.fluid.initializer import MSRA
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay
from PIL import Image, ImageEnhance, ImageDraw
import os

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("paddle.static")
logger.setLevel(logging.INFO)






def custom_reader(file_list, data_dir, input_size, mode):
    def reader():
        np.random.shuffle(file_list)
        for line in file_list:
            if mode == 'train' or mode == 'eval':
                ######################  以下可能是需要自定义修改的部分   ############################
                parts = line.split('\t')
                image_path = parts[0]
                img = Image.open(os.path.join(data_dir, image_path))
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                im_width, im_height = img.size
                # bbox 的列表，每一个元素为这样
                # layout: label | x-center | y-cneter | width | height | difficult
                bbox_labels = []
                for object_str in parts[1:]:
                    if len(object_str) <= 1:
                        continue
                    bbox_sample = []
                    object = json.loads(object_str)
                    bbox_sample.append(float(train_parameters['label_dict'][object['value']]))
                    bbox = object['coordinate']
                    box = [bbox[0][0], bbox[0][1], bbox[1][0] - bbox[0][0], bbox[1][1] - bbox[0][1]]
                    bbox = box_to_center_relative(box, im_height, im_width)
                    bbox_sample.append(float(bbox[0]))
                    bbox_sample.append(float(bbox[1]))
                    bbox_sample.append(float(bbox[2]))
                    bbox_sample.append(float(bbox[3]))
                    difficult = float(0)
                    bbox_sample.append(difficult)
                    bbox_labels.append(bbox_sample)
                ######################  可能需要自定义修改部分结束   ############################
                if len(bbox_labels) == 0:
                    continue
                img, sample_labels = preprocess(img, bbox_labels, input_size, mode)
                # sample_labels = np.array(sample_labels)
                if len(sample_labels) == 0: continue
                boxes = sample_labels[:, 1:5]
                lbls = sample_labels[:, 0].astype('int32')
                difficults = sample_labels[:, -1].astype('int32')
                max_box_num = train_parameters['max_box_num']
                cope_size = max_box_num if len(boxes) >= max_box_num else len(boxes)
                ret_boxes = np.zeros((max_box_num, 4), dtype=np.float32)
                ret_lbls = np.zeros((max_box_num), dtype=np.int32)
                ret_difficults = np.zeros((max_box_num), dtype=np.int32)
                ret_boxes[0: cope_size] = boxes[0: cope_size]
                ret_lbls[0: cope_size] = lbls[0: cope_size]
                ret_difficults[0: cope_size] = difficults[0: cope_size]
                yield img, ret_boxes, ret_lbls
            elif mode == 'test':
                img_path = os.path.join(line)
                yield Image.open(img_path)

    return reader


class YOLOv3(object):
    def __init__(self, class_num, anchors, anchor_mask):
        self.outputs = []
        self.downsample_ratio = 1
        self.anchor_mask = anchor_mask
        self.anchors = anchors
        self.class_num = class_num

        self.yolo_anchors = []
        self.yolo_classes = []
        for mask_pair in self.anchor_mask:
            mask_anchors = []
            for mask in mask_pair:
                mask_anchors.append(self.anchors[2 * mask])
                mask_anchors.append(self.anchors[2 * mask + 1])
            self.yolo_anchors.append(mask_anchors)
            self.yolo_classes.append(class_num)

    def name(self):
        return 'YOLOv3'

    def get_anchors(self):
        return self.anchors

    def get_anchor_mask(self):
        return self.anchor_mask

    def get_class_num(self):
        return self.class_num

    def get_downsample_ratio(self):
        return self.downsample_ratio

    def get_yolo_anchors(self):
        return self.yolo_anchors

    def get_yolo_classes(self):
        return self.yolo_classes

    def conv_bn(self,
                input,
                num_filters,
                filter_size,
                stride,
                padding,
                use_cudnn=True):
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            act=None,
            use_cudnn=use_cudnn,
            param_attr=ParamAttr(initializer=fluid.initializer.Normal(0., 0.02)),
            bias_attr=False)

        # batch_norm中的参数不需要参与正则化，所以主动使用正则系数为0的正则项屏蔽掉
        # 在batch_norm中使用 leaky 的话，只能使用默认的 alpha=0.02；如果需要设值，必须提出去单独来
        out = fluid.layers.batch_norm(
            input=conv, act=None,
            param_attr=ParamAttr(initializer=fluid.initializer.Normal(0., 0.02), regularizer=L2Decay(0.)),
            bias_attr=ParamAttr(initializer=fluid.initializer.Constant(0.0), regularizer=L2Decay(0.)))
        out = fluid.layers.leaky_relu(out, 0.1)
        return out

    def downsample(self, input, num_filters, filter_size=3, stride=2, padding=1):
        self.downsample_ratio *= 2
        return self.conv_bn(input,
                            num_filters=num_filters,
                            filter_size=filter_size,
                            stride=stride,
                            padding=padding)

    def basicblock(self, input, num_filters):
        conv1 = self.conv_bn(input, num_filters, filter_size=1, stride=1, padding=0)
        conv2 = self.conv_bn(conv1, num_filters * 2, filter_size=3, stride=1, padding=1)
        out = fluid.layers.elementwise_add(x=input, y=conv2, act=None)
        return out

    def layer_warp(self, input, num_filters, count):
        res_out = self.basicblock(input, num_filters)
        for j in range(1, count):
            res_out = self.basicblock(res_out, num_filters)
        return res_out

    def upsample(self, input, scale=2):
        # get dynamic upsample output shape
        shape_nchw = fluid.layers.shape(input)
        shape_hw = fluid.layers.slice(shape_nchw, axes=[0], starts=[2], ends=[4])
        shape_hw.stop_gradient = True
        in_shape = fluid.layers.cast(shape_hw, dtype='int32')
        out_shape = in_shape * scale
        out_shape.stop_gradient = True

        # reisze by actual_shape
        out = fluid.layers.resize_nearest(
            input=input,
            scale=scale,
            actual_shape=out_shape)
        return out

    def yolo_detection_block(self, input, num_filters):
        assert num_filters % 2 == 0, "num_filters {} cannot be divided by 2".format(num_filters)
        conv = input
        for j in range(2):
            conv = self.conv_bn(conv, num_filters, filter_size=1, stride=1, padding=0)
            conv = self.conv_bn(conv, num_filters * 2, filter_size=3, stride=1, padding=1)
        route = self.conv_bn(conv, num_filters, filter_size=1, stride=1, padding=0)
        tip = self.conv_bn(route, num_filters * 2, filter_size=3, stride=1, padding=1)
        return route, tip

    def net(self, img):
        # darknet
        stages = [1, 2, 8, 8, 4]
        assert len(self.anchor_mask) <= len(stages), "anchor masks can't bigger than downsample times"
        # 256x256
        conv1 = self.conv_bn(img, num_filters=32, filter_size=3, stride=1, padding=1)
        downsample_ = self.downsample(conv1, conv1.shape[1] * 2)
        blocks = []

        for i, stage_count in enumerate(stages):
            block = self.layer_warp(downsample_, 32 * (2 ** i), stage_count)
            blocks.append(block)
            if i < len(stages) - 1:
                downsample_ = self.downsample(block, block.shape[1] * 2)
        blocks = blocks[-1:-4:-1]  # 取倒数三层，并且逆序，后面跨层级联需要

        # yolo detector
        for i, block in enumerate(blocks):
            # yolo 中跨视域链接
            if i > 0:
                block = fluid.layers.concat(input=[route, block], axis=1)
            route, tip = self.yolo_detection_block(block, num_filters=512 // (2 ** i))
            block_out = fluid.layers.conv2d(
                input=tip,
                num_filters=len(self.anchor_mask[i]) * (self.class_num + 5),  # 5 elements represent x|y|h|w|score
                filter_size=1,
                stride=1,
                padding=0,
                act=None,
                param_attr=ParamAttr(initializer=fluid.initializer.Normal(0., 0.02)),
                bias_attr=ParamAttr(initializer=fluid.initializer.Constant(0.0), regularizer=L2Decay(0.)))
            self.outputs.append(block_out)
            # 为了跨视域链接，差值方式提升特征图尺寸
            if i < len(blocks) - 1:
                route = self.conv_bn(route, 256 // (2 ** i), filter_size=1, stride=1, padding=0)
                route = self.upsample(route)

        return self.outputs


def single_custom_reader(file_path, data_dir, input_size, mode):
    file_path = os.path.join(data_dir, file_path)
    images = [line.strip() for line in open(file_path)]
    reader = custom_reader(images, data_dir, input_size, mode)
    reader = paddle.reader.shuffle(reader, train_parameters['train_batch_size'])
    reader = paddle.batch(reader, train_parameters['train_batch_size'])
    return reader


def optimizer_sgd_setting():
    batch_size = train_parameters["train_batch_size"]
    iters = train_parameters["image_count"] // batch_size
    iters = 1 if iters < 1 else iters
    learning_strategy = train_parameters['sgd_strategy']
    lr = learning_strategy['learning_rate']

    boundaries = [i * iters for i in learning_strategy["lr_epochs"]]
    values = [i * lr for i in learning_strategy["lr_decay"]]
    logger.info("origin learning rate: {0} boundaries: {1}  values: {2}".format(lr, boundaries, values))

    optimizer = fluid.optimizer.SGDOptimizer(
        learning_rate=fluid.layers.piecewise_decay(boundaries, values),
        # learning_rate=lr,
        regularization=fluid.regularizer.L2Decay(0.00005))

    return optimizer


def build_program_with_feeder(main_prog, startup_prog, place):
    max_box_num = train_parameters['max_box_num']
    ues_tiny = train_parameters['use_tiny']
    yolo_config = train_parameters['yolo_tiny_cfg'] if ues_tiny else train_parameters['yolo_cfg']
    with fluid.program_guard(main_prog, startup_prog):
        img = fluid.layers.data(name='img', shape=yolo_config['input_size'], dtype='float32')
        gt_box = fluid.layers.data(name='gt_box', shape=[max_box_num, 4], dtype='float32')
        gt_label = fluid.layers.data(name='gt_label', shape=[max_box_num], dtype='int32')
        feeder = fluid.DataFeeder(feed_list=[img, gt_box, gt_label], place=place, program=main_prog)
        reader = single_custom_reader(train_parameters['train_list'],
                                      train_parameters['data_dir'],
                                      yolo_config['input_size'], 'train')
        ues_tiny = train_parameters['use_tiny']
        yolo_config = train_parameters['yolo_tiny_cfg'] if ues_tiny else train_parameters['yolo_cfg']
        with fluid.unique_name.guard():
            model = get_yolo(ues_tiny, train_parameters['class_dim'], yolo_config['anchors'],
                             yolo_config['anchor_mask'])
            outputs = model.net(img)
        return feeder, reader, get_loss(model, outputs, gt_box, gt_label)


def get_loss(model, outputs, gt_box, gt_label):
    losses = []
    downsample_ratio = model.get_downsample_ratio()
    with fluid.unique_name.guard('train'):
        for i, out in enumerate(outputs):
            loss = fluid.layers.yolov3_loss(
                x=out,
                gt_box=gt_box,
                gt_label=gt_label,
                anchors=model.get_anchors(),
                anchor_mask=model.get_anchor_mask()[i],
                class_num=model.get_class_num(),
                ignore_thresh=train_parameters['ignore_thresh'],
                use_label_smooth=False,  # 对于类别不多的情况，设置为 False 会更合适一些，不然 score 会很小
                downsample_ratio=downsample_ratio)
            losses.append(fluid.layers.reduce_mean(loss))
            downsample_ratio //= 2
        loss = sum(losses)
        optimizer = optimizer_sgd_setting()
        optimizer.minimize(loss)
        return loss


def train():
    init_log_config()
    init_train_parameters()
    logger.info("start train YOLOv3, train params:%s", str(train_parameters))

    logger.info("create place, use gpu:" + str(train_parameters['use_gpu']))
    place = fluid.CUDAPlace(0) if train_parameters['use_gpu'] else fluid.CPUPlace()

    logger.info("build network and program")
    train_program = fluid.Program()
    start_program = fluid.Program()
    feeder, reader, loss = build_program_with_feeder(train_program, start_program, place)

    logger.info("build executor and init params")
    exe = fluid.Executor(place)
    exe.run(start_program)
    train_fetch_list = [loss.name]
    load_pretrained_params(exe, train_program)

    stop_strategy = train_parameters['early_stop']
    successive_limit = stop_strategy['successive_limit']
    sample_freq = stop_strategy['sample_frequency']
    min_curr_map = stop_strategy['min_curr_map']
    min_loss = stop_strategy['min_loss']
    stop_train = False
    successive_count = 0
    total_batch_count = 0
    valid_thresh = train_parameters['valid_thresh']
    nms_thresh = train_parameters['nms_thresh']
    current_best_loss = 10000000000.0
    for pass_id in range(train_parameters["num_epochs"]):
        logger.info("current pass: {}, start read image".format(pass_id))
        batch_id = 0
        total_loss = 0.0
        for batch_id, data in enumerate(reader()):
            t1 = time.time()
            loss = exe.run(train_program, feed=feeder.feed(data), fetch_list=train_fetch_list)
            period = time.time() - t1
            loss = np.mean(np.array(loss))
            total_loss += loss
            batch_id += 1
            total_batch_count += 1

            if batch_id % 10 == 0:  # 调整日志输出的频率
                logger.info(
                    "pass {}, trainbatch {}, loss {} time {}".format(pass_id, batch_id, loss, "%2.2f sec" % period))
        pass_mean_loss = total_loss / batch_id
        logger.info("pass {0} train result, current pass mean loss: {1}".format(pass_id, pass_mean_loss))
        # 采用每训练完一轮停止办法，可以调整为更精细的保存策略
        if pass_mean_loss < current_best_loss:
            logger.info("temp save {} epcho train result, current best pass loss {}".format(pass_id, pass_mean_loss))
            fluid.io.save_persistables(dirname=train_parameters['save_model_dir'], main_program=train_program,
                                       executor=exe)
            current_best_loss = pass_mean_loss

    logger.info("training till last epcho, end training")
    fluid.io.save_persistables(dirname=train_parameters['save_model_dir'], main_program=train_program, executor=exe)


if __name__ == '__main__':
    train()
