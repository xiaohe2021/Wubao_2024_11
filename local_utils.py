import copy
import math  # 导入数学函数库
import os
import random
import time
import xml.etree.ElementTree as ET
from collections import defaultdict

import cv2
import numpy as np


# 导入必要的库
class BBox:
    def __init__(self, xmin: int | float, ymin: int | float, xmax: int | float, ymax: int | float):
        self._box = [xmin, ymin, xmax, ymax]

    def to_list(self):
        return self._box

    @classmethod
    def default(cls) -> 'Self':
        return cls(0, 0, 1, 1)

    def tojson(self) -> dict[str, float | int]:
        return {
            'xmin': self._box[0],
            'ymin': self._box[1],
            'xmax': self._box[2],
            'ymax': self._box[3],
        }

    @classmethod
    def fromjson(cls, json: dict[str, float | int]) -> Self:
        return cls(json['xmin'], json['ymin'], json['xmax'], json['ymax'])

    def __repr__(self) -> str:
        return f"<BBox>: {self._box}"

    __str__ = __repr__

    def __eq__(self, other: Self) -> bool:
        return self._box == other._box

    def __iter__(self) -> Iterator[float]:
        return iter(self._box)

    def __getitem__(self, index: int) -> float | int:
        return self._box[index]

    def __setitem__(self, index: int, value: float | int):
        self._box[index] = value

    def __len__(self) -> int:
        return len(self._box)

    def __copy__(self):
        return self.copy()

    def copy(self) -> "BBox":
        return BBox(*self._box.copy())

    def clone(self) -> "BBox":
        return BBox(*self._box.copy())

    @property
    def xmin(self) -> float:
        return self._box[0]

    def set_xmin(self, value: float):
        self._box[0] = value

    @property
    def ymin(self) -> float:
        return self._box[1]

    def set_ymin(self, value: float):
        self._box[1] = value

    @property
    def xmax(self) -> float:
        return self._box[2]

    def set_xmax(self, value: float):
        self._box[2] = value

    @property
    def ymax(self) -> float:
        return self._box[3]

    def set_ymax(self, value: float):
        self._box[3] = value

    @property
    def x1(self) -> float:
        return self._box[0]

    @property
    def y1(self) -> float:
        return self._box[1]

    @property
    def x2(self) -> float:
        return self._box[2]

    @property
    def y2(self) -> float:
        return self._box[3]

    @property
    def cx(self) -> float:
        return (self._box[0] + self._box[2]) * 0.5

    @property
    def cy(self) -> float:
        return (self._box[1] + self._box[3]) * 0.5

    @property
    def center(self) -> tuple[float, float]:
        return (self._box[0] + self._box[2]) * 0.5, (self._box[1] + self._box[3]) * 0.5

    @property
    def width(self) -> float:
        return self._box[2] - self._box[0]

    @property
    def height(self) -> float:
        return self._box[3] - self._box[1]

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def diagonal(self):
        box = self._box
        return math.sqrt((box[2] - box[0]) ** 2 + (box[3] - box[1]) ** 2)

    def intersection(self, rhs: "BBox") -> float:
        xmin1, ymin1, xmax1, ymax1 = self._box
        xmin2, ymin2, xmax2, ymax2 = rhs._box

        intersect_xmin = max(xmin1, xmin2)
        intersect_ymin = max(ymin1, ymin2)
        intersect_xmax = min(xmax1, xmax2)
        intersect_ymax = min(ymax1, ymax2)

        if intersect_xmin < intersect_xmax and intersect_ymin < intersect_ymax:
            return (intersect_xmax - intersect_xmin) * (intersect_ymax - intersect_ymin)
        else:
            return 0.0

    def iou(self, rhs: "BBox") -> float:
        intersection_area = self.intersection(rhs)
        union_area = self.area + rhs.area - intersection_area
        return intersection_area / union_area if union_area > 0 else 0.0

    def clip_self(self, width: float | int, height: float | int) -> "BBox":
        box = self._box
        box[0] = max(box[0], 0.)
        box[1] = max(box[1], 0.)
        box[2] = min(box[2], width)
        box[3] = min(box[3], height)
        return self

    def clip(self, width: float | int, height: float | int) -> "BBox":
        return self.copy().clip_self(width, height)

class CreateVoc:
    def __init__(self):
        self.filename = None
        self.labels = None
        self.boxes = None
        self.folder_path = None

    def write_xml(self):
        root = ET.Element('annotation')
        tree = ET.ElementTree(root)
        filename_element = self.add_element(root, 'filename', self.filename)
        dic = self.list2dict(self.labels, self.boxes)
        for keys, values in dic.items():
            for value in values:
                object_element = self.add_element(root, 'object')
                name_element = self.add_element(object_element, 'name', keys)
                self.add_element(object_element, 'pose', 'Unspecified')
                self.add_element(object_element, 'truncated', '0')
                self.add_element(object_element, 'difficult', '0')
                bndbox_element = self.add_element(object_element, 'bndbox')
                xmin_element = self.add_element(bndbox_element, 'xmin', str(value[0]))
                ymin_element = self.add_element(bndbox_element, 'ymin', str(value[1]))
                xmax_element = self.add_element(bndbox_element, 'xmax', str(value[2]))
                ymax_element = self.add_element(bndbox_element, 'ymax', str(value[3]))
        self.indent(root)
        xml_save_path = os.path.join(self.folder_path, self.filename)
        print(f"xml 保存路径为：", xml_save_path)
        tree.write(xml_save_path, encoding='utf-8', xml_declaration=True)

    @staticmethod
    def add_element(parent, tag, text=None):
        element = ET.SubElement(parent, tag)
        if text:
            element.text = text
        return element

    @staticmethod
    def list2dict(key_lis, val_list):
        result_dict = defaultdict(list)
        for k, v in zip(key_lis, val_list):
            result_dict[k].append(v)
        return dict(result_dict)

    def indent(self, elem, level=0):
        i = "\n" + level * "\t"
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "\t"
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                self.indent(elem, level + 1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i

    def set_info(self, filename, labels, boxes, folder_path):
        self.filename = filename
        self.labels = labels
        self.boxes = boxes
        self.folder_path = folder_path


class Videos:
    def __init__(self, path):
        self.path = path  # 单个视频文件路径或文件夹路径
        self.current_image = None  # 当前帧图像
        self.frame_count = 0  # 帧计数器
        self.frame_id = None  # 当前帧编号
        self.video_start_time = None  # 视频开始时间

    def __get_video_list(self):
        """
        获取视频文件列表。
        如果路径是文件，返回该文件路径。
        如果路径是文件夹，返回文件夹下所有视频文件路径。
        """
        if os.path.isfile(self.path):
            return [self.path]  # 单个文件
        elif os.path.isdir(self.path):
            # 遍历文件夹中的视频文件
            return [
                os.path.join(self.path, f) for f in os.listdir(self.path)
                if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))  # 常见视频格式
            ]
        else:
            print(f"路径无效: {self.path}")
            return []

    def read_video(self):
        """
        逐帧读取视频，支持单个视频或目录路径。
        """
        video_list = self.__get_video_list()
        if not video_list:
            print("没有找到视频文件")
            return

        print(f"找到的视频列表: {video_list}")
        for video in video_list:
            self.video_start_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
            print(f"开始处理视频: {video}, 开始时间: {self.video_start_time}")
            self.frame_count = 0

            # 打开视频文件
            cap = cv2.VideoCapture(video)
            if not cap.isOpened():
                print(f"无法打开视频文件: {video}")
                continue

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print(f"视频读取完毕: {video}")
                    break

                self.current_image = copy.deepcopy(frame)  # 复制当前帧
                self.frame_count += 1
                self.frame_id = f"{self.frame_count:06d}"  # 生成6位数字的帧ID

                yield frame  # 生成当前帧

            cap.release()  # 释放视频资源
        print("所有视频处理完成")


class Images:
    def __init__(self, path):
        self.path = path
        self.__image_types = {'.jpg', '.jpeg', '.png', '.bmp'}  # 支持的图像格式
        self.current_image = None
        self.current_image_name = None
        self.current_image_folder = None

    def read_img_ch(self, path):
        """
        读取单张图片数据
        """
        with open(path, 'rb') as file:
            data = file.read()
        data_array = np.frombuffer(data, dtype=np.uint8)
        im_ = cv2.imdecode(data_array, cv2.IMREAD_COLOR)
        return im_

    def read_images(self):
        """
        读取图片，可处理文件夹路径或单张图片路径。
        """
        if os.path.isfile(self.path):  # 如果路径是单个文件
            prefix, suffix = os.path.splitext(self.path)
            if suffix in self.__image_types:  # 确认文件格式为支持的类型
                frame = self.read_img_ch(self.path)
                self.current_image = copy.deepcopy(frame)
                self.current_image_name = os.path.basename(self.path)
                self.current_image_folder = os.path.dirname(self.path)
                yield frame
            else:
                raise ValueError(f"Unsupported image type: {suffix}")
        elif os.path.isdir(self.path):  # 如果路径是文件夹
            for parent, folders, files in os.walk(self.path):
                for file in files:
                    prefix, suffix = os.path.splitext(file)
                    if suffix not in self.__image_types:
                        continue
                    frame = self.read_img_ch(os.path.join(parent, file))
                    self.current_image = copy.deepcopy(frame)
                    self.current_image_name = file
                    self.current_image_folder = parent
                    yield frame
        else:
            raise FileNotFoundError(f"Path {self.path} does not exist.")


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        label = str(label)
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    return img


def letterbox(img, new_hw, color=(114, 114, 114)):
    """将图像调整为指定大小，保持长宽比，不足部分填充"""
    src_h, src_w, _ = img.shape  # 获取原始图像的高度和宽度
    dst_h, dst_w = new_hw  # 目标高度和宽度

    # 如果尺寸相同，直接返回原图
    if src_h == dst_h and src_w == dst_w:
        return img, 1

    # 计算缩放比例
    scale_w = src_w / dst_w
    scale_h = src_h / dst_h

    # 选择较大的缩放比例，确保图像能完全放入目标尺寸
    if scale_w > scale_h:
        scale = scale_w
        new_w = dst_w
        new_h = src_h / scale_w
    else:
        scale = scale_h
        new_w = src_w / scale_h
        new_h = dst_h

    # 调整图像大小
    resized = cv2.resize(img, (int(new_w), int(new_h)), interpolation=cv2.INTER_NEAREST)

    # 添加边框填充
    dst = cv2.copyMakeBorder(resized,
                             0, math.ceil(dst_h - new_h),
                             0, math.ceil(dst_w - new_w),
                             cv2.BORDER_CONSTANT, value=color)
    return dst, scale
