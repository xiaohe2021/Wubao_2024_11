import math

import cv2
import numpy as np


def letterbox(img, new_hw, color=(114, 114, 114)):
    src_h, src_w, _ = img.shape
    dst_h, dst_w = new_hw
    if src_h == dst_h and src_w == dst_w:
        return img, 1
    scale_w = src_w / dst_w
    scale_h = src_h / dst_h
    if scale_w > scale_h:
        scale = scale_w
        new_w = dst_w
        new_h = src_h / scale_w
    else:
        scale = scale_h
        new_w = src_w / scale_h
        new_h = dst_h

    resized = cv2.resize(img, (int(new_w), int(new_h)), interpolation=cv2.INTER_NEAREST)
    dst = cv2.copyMakeBorder(resized,
                             0, math.ceil(dst_h - new_h),
                             0, math.ceil(dst_w - new_w),
                             cv2.BORDER_CONSTANT, value=color)  # add border
    return dst, scale

# def yolo_det_preprocess(self, img: np.ndarray) -> tuple[np.ndarray, float]:
#     """预处理图像用于YOLO检测"""
#     input_hw = (self.input_shape[2], self.input_shape[3])  # 获取输入尺寸
#     if isinstance(self.input_shape[2], str) or isinstance(self.input_shape[3], str):
#         # 若模型形状为字符串，使用默认尺寸 (640, 640)
#         input_hw = (640, 640)
#
#     # 调整图像大小并保持长宽比
#     img, scale = letterbox(img, new_hw=input_hw)
#
#     # 确保图像类型为 float32，且能够进行归一化
#     img = img.astype(np.float32) / 255.0  # 归一化到 [0, 1]
#
#     # 检查图像的通道数
#     if len(img.shape) == 2:  # 灰度图像
#         img = np.expand_dims(img, axis=-1)  # 增加通道维度
#         img = np.repeat(img, 3, axis=-1)  # 转为 3 通道
#     elif len(img.shape) == 3 and img.shape[2] == 3:  # RGB图像
#         img = img[:, :, ::-1]  # 转换为RGB格式（如果是BGR格式）
#
#     # 转换为 NCHW 格式
#     img = img.transpose(2, 0, 1)
#     return np.ascontiguousarray(img[None, :, :, :]), scale  # 增加batch维度

def yolo_det_preprocess(img: np.ndarray, input_hw: tuple[int, int]) -> tuple[np.ndarray, float]:
    img, scale = letterbox(img, new_hw=input_hw)
    img = img.astype(np.float32) / 255.
    img = img[:, :, ::-1].transpose(2, 0, 1)
    return np.ascontiguousarray(img), scale


IMAGENET_MEAN = np.array([0.406, 0.485, 0.456], np.float32) * 255.  # BGR mean
IMAGENET_STD = np.array([0.225, 0.229, 0.224], np.float32) * 255.  # BGR standard deviation


def yolo_cls_preprocess(img: np.ndarray, input_hw: tuple[int, int]) -> np.ndarray:
    img = cv2.resize(img, (input_hw[1], input_hw[0]), interpolation=cv2.INTER_LINEAR)
    img = (img.astype(np.float32) - IMAGENET_MEAN) / IMAGENET_STD
    img = img[:, :, ::-1].transpose(2, 0, 1)
    return np.ascontiguousarray(img)
