import os
import threading
import time

import cv2
import numpy as np

from local_utils import Videos, plot_one_box, CreateVoc, Images
from model import V10OnnxPredictor, V5OnnxPredictor, V5TorchPredictor, V8SegOnnxPredictor, V8TorchSegPredictor, \
    V10TorchPredictor


# 导入必要的库


def read_img_ch(path):
    with open(path, 'rb') as file:
        data = file.read()
    data_array = np.frombuffer(data, dtype=np.uint8)
    im_ = cv2.imdecode(data_array, cv2.IMREAD_COLOR)
    return im_


class Main:
    def __init__(self):
        # 加载配置文件
        self.config = self.load_config('config.yaml')

        self.save_path = ''
        # 初始化默认配置项
        self.camera_state = 'default_state'
        self.project = 'default_project'
        self.scene = 'default_scene'
        self.task = 'image'
        self.model_type = 'v5'
        self.labels = []
        self.model_task = 'detect'

        # 添加帧计数和时间控制变量
        self.last_save_time = 0  # 上次保存的时间
        self.saved_frame_count = 0  # 已保存的错误标签帧计数
        # 筛选启用的模型并初始化
        self.models = []  # 存储多个模型
        self.video_path = set()  # 使用集合存储视频路径
        self.image_path = set()  # 使用集合存储图像路径
        self.error_labels = set()  #使用集合存储error_labels
        for model_config in self.config['models']:
            if model_config.get('enabled', True):
                model, video_path, image_path,error_labels = self.process_model(model_config)
                if model:
                    self.models.append(model)  # 将模型添加到列表中
                    # 将视频路径和图像路径分别加入集合中，避免重复
                    if video_path:
                        self.video_path.add(video_path)  # 将视频路径添加到集合
                    if image_path:
                        self.image_path.add(image_path)  # 将图像路径添加到集合中
                    if isinstance(error_labels, list):
                        self.error_labels.update(error_labels)  # 如果是列表，使用 update 将列表中的每个元素添加到集合中
                    else:
                        self.error_labels.add(error_labels)  # 如果是单个元素，使用 add

        self.image_path = list(self.image_path)
        self.video_path = list(self.video_path)
        self.error_labels = list(self.error_labels)
        # 任务分发：路径交集计算完成后，再启动任务
        for model_config, model in zip(self.config['models'], self.models):
            task = model_config.get('task', 'image')
            if task == "video":
                # 启动视频推理线程
                thread = threading.Thread(target=self.detect_video,
                                          args=(self.video_path, self.error_labels))
                thread.start()
            elif task == "image":
                # 启动图像推理线程
                thread = threading.Thread(target=self.detect_images,
                                          args=(self.image_path, self.error_labels))
                thread.start()
            else:
                print(f"未知任务类型：{task}")

    @staticmethod
    def load_config(config_path):
        import yaml
        with open(config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)

    def process_model(self, model_config):
        print(f"正在初始化并处理模型：{model_config['model_path']}")

        # 初始化路径
        video_path = os.path.normpath(model_config['video_path'])
        image_path = os.path.normpath(model_config['image_path'])
        self.save_path = os.path.normpath(model_config['save_path'])
        model_path = os.path.normpath(model_config['model_path'])
        error_labels = model_config.get('error_labels')
        # 从启用的模型配置中读取配置项

        self.camera_state = model_config.get('camera_state', self.camera_state)  # 从模型配置读取，若没有则使用默认值
        self.project = model_config.get('project', self.project)  # 从模型配置读取，若没有则使用默认值
        self.scene = model_config.get('scene', self.scene)  # 从模型配置读取，若没有则使用默认值
        self.task = model_config.get('task', self.task)  # 从模型配置读取，若没有则使用默认值
        self.model_type = model_config.get('model_type', self.model_type)  # 从模型配置读取，若没有则使用默认值
        self.labels = model_config.get('labels', self.labels)  # 从模型配置读取，若没有则使用默认值
        self.model_task = model_config.get('model_task', self.model_task)  # 从模型配置读取，若没有则使用默认值
        # 加载模型
        model = self.load_model(
            model_path,
            model_config['model_type'],
            model_config['model_task'],
            model_config['labels']
        )

        return model, video_path, image_path, error_labels

    @staticmethod
    def load_model(model_path, model_type, model_task, labels):
        # 模型加载逻辑
        if model_task == "detect":
            if model_type == "v5":
                return V5OnnxPredictor(model_path, labels) if model_path.endswith('.onnx') else V5TorchPredictor(
                    model_path, labels)
            elif model_type == "v10":
                return V10OnnxPredictor(model_path, labels) if model_path.endswith('.onnx') else V10TorchPredictor(
                    model_path, labels)
        elif model_task == "segment":
            if model_type == "v8":
                return V8SegOnnxPredictor(model_path, labels) if model_path.endswith('.onnx') else V8TorchSegPredictor(
                    model_path, labels)
            return None

    def detect_video(self, video_path, error_labels):
        # 设置窗口显示的尺寸
        window_width = 800  # 窗口宽度
        window_height = 600  # 窗口高度
        cv2.namedWindow('Video Feed', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Video Feed', window_width, window_height)  # 设置窗口初始大小
        videos = Videos(video_path)
        try:
            print("******************start************")
            for frame in videos.read_video():
                all_labels = []  # 存储所有模型的标签
                all_boxes = []  # 存储所有模型的框

                # 对每个模型进行推理
                for model in self.models:
                    labels, boxes = model.infer(frame) if model else ([], [])
                    all_labels.extend(labels)  # 合并标签
                    all_boxes.extend(boxes)  # 合并框
                    for idx, box in enumerate(all_boxes):
                        label = all_labels[idx]   # 确保标签数量匹配
                        frame = plot_one_box(box, frame, label=label)
                # 根据窗口大小调整帧的内容
                frame_resized = cv2.resize(frame, (window_width, window_height))
                cv2.imshow('Video Feed', frame_resized)

                # 按 'q' 键退出循环
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                # 检测到错误标签，进行帧保存处理
                error_idx = [idx for idx, label in enumerate(all_labels) if label in error_labels]
                if error_idx:
                    current_time = time.time()  # 获取当前时间
                    # 判断当前时间和上次保存时间差
                    if current_time - self.last_save_time > 1:
                        self.saved_frame_count = 0  # 每秒重置计数器

                    # 每秒最多保存 5 帧
                    if self.saved_frame_count < 5:
                        print(f"************开始保存*********************")
                        self.saved_frame_count += 1
                        self.last_save_time = current_time  # 更新保存时间

                        # 过滤出错误标签的标签和框
                        labels_filtered = [label for idx, label in enumerate(all_labels) if idx in error_idx]
                        boxes_filtered = [box for idx, box in enumerate(all_boxes) if idx in error_idx]

                        # 创建XML文件并保存图像
                        voc = CreateVoc()  # 创建VOC的类
                        strftime = videos.video_start_time
                        prefix = f'{self.camera_state}_{self.project}_{self.scene}__{strftime}_{videos.frame_id}'
                        print(f"Generated prefix: {prefix}")  # 调试输出
                        xml_name = f'{prefix}.xml'
                        img_name = f'{prefix}.jpg'

                        # 编码图像并保存
                        success, encoded_image = cv2.imencode('.jpg', videos.current_image)
                        if success:
                            with open(os.path.join(self.save_path, img_name), 'wb') as f:
                                f.write(encoded_image.tobytes())

                        # 保存到 VOC 格式的 XML
                        voc.set_info(xml_name, labels_filtered, boxes_filtered, self.save_path)
                        voc.write_xml()
                    else:
                        print(f"错误标签帧过多，已跳过此帧：{videos.frame_id}")

        except Exception as e:
            print(f"Error reading video: {e}")

    def detect_images(self, image_path, error_labels):
        voc = CreateVoc()
        images = Images(image_path)
        for model in self.models:
            for image in images.read_images():
                labels, boxes = model.infer(image)
                print("**************start*************")
                # for idx, box in enumerate(boxes):
                #     image = plot_one_box(box, image, label=labels[idx])
                # img_name = self.images.current_image_name
                # prefix, suffix = os.path.splitext(img_name)
                current_time = time.strftime("%Y_%m_%d_%H%M%S", time.localtime())
                prefix = f'{self.camera_state}_{self.project}_{self.scene}__{current_time}'
                xml_name = f'{prefix}.xml'
                img_name = f'{prefix}.jpg'
                success, encoded_image = cv2.imencode('.jpg', images.current_image)

                # # 检查编码是否成功
                if success:
                    # 将编码后的图像数据保存为文件
                    with open(os.path.join(self.save_path, img_name), 'wb') as f:
                        f.write(encoded_image.tobytes())
                voc.set_info(xml_name, labels, boxes, images.current_image_folder)
                voc.write_xml()


if __name__ == "__main__":
    main_app = Main()  # 初始化并运行 Main 类

# # 加载YAML配置文件
# def load_config(config_path,model_name=None):
#     with open(config_path, 'r', encoding='utf-8') as file:
#         config = yaml.safe_load(file)
#     if model_name:
#         return config['models'].get(model_name,{})
#     return config
#
#
# class Main:
#     def __init__(self):
#         # 加载配置
#         self.config = load_config('config.yaml')
#
#         # 处理路径，确保兼容 Windows 平台
#         self.video_path = os.path.normpath(self.config['video_path'])
#         self.image_path = os.path.normpath(self.config['image_path'])
#         self.save_path = os.path.normpath(self.config['save_path'])
#         self.model_path = os.path.normpath(self.config['model_path'])
#
#         # 实例化相关类
#         self.videos = Videos(self.video_path)
#         self.images = Images(self.image_path)
#
# # 其他配置项
# self.camera_state = self.config['camera_state']
# self.project = self.config['project']
# self.scene = self.config['scene']
# self.task = self.config.get('task', 'image')  # 从配置文件中读取任务类型，默认是 'image'
# self.error_labels = self.config['error_labels']
# self.model_type = self.config['model_type']
# self.labels = self.config['labels']
# self.model_task = self.config['model_task']
#         # 添加帧计数和时间控制变量
#         self.last_save_time = 0  # 上次保存的时间
#         self.saved_frame_count = 0  # 已保存的错误标签帧计数
#
#     def detect_video(self):
#         model = None
#         # 根据任务类型和模型类型加载相应的推理模型
#         if self.model_task == "detect":  # 检测任务
#             if self.model_type == "v5":
#                 if self.model_path.endswith('.onnx'):
#                     model = V5OnnxPredictor(self.model_path, self.labels)
#                 elif self.model_path.endswith('.pt'):
#                     model = V5TorchPredictor(self.model_path, self.labels)
#             elif self.model_type == "v8":
#                 if self.model_path.endswith('.onnx'):
#                     # model = V8OnnxPredictor(self.model_path, self.labels)  # 检测任务模型
#                     pass
#                 elif self.model_path.endswith('.pt'):
#                     # model = V8TorchPredictor(self.model_path, self.labels)
#                     pass
#             elif self.model_type == "v10":
#                 if self.model_path.endswith('.onnx'):
#                     model = V10OnnxPredictor(self.model_path, self.labels)  # v10检测任务模型
#                 elif self.model_path.endswith('.pt'):
#                     model = V10TorchPredictor(self.model_path, self.labels)
#         elif self.model_task == "segment":  # 分割任务
#             if self.model_type == "v5":
#                 # v5 不支持分割任务
#                 pass
#             elif self.model_type == "v8":
#                 if self.model_path.endswith('.onnx'):
#                     model = V8SegOnnxPredictor(self.model_path, self.labels)  # 分割任务模型
#                 elif self.model_path.endswith('.pt'):
#                     model = V8TorchSegPredictor(self.model_path, self.labels)
#         else:
#             # 默认模型
#             pass
#
#         # 设置窗口显示的尺寸
#         window_width = 800  # 窗口宽度
#         window_height = 600  # 窗口高度
#         cv2.namedWindow('Video Feed', cv2.WINDOW_NORMAL)
#         cv2.resizeWindow('Video Feed', window_width, window_height)  # 设置窗口初始大小
#
#         try:
#             print("******************start************")
#             for frame in self.videos.read_video():
#                 labels, boxes = model.infer(frame) if model else ([], [])
#                 for idx, box in enumerate(boxes):
#                     label = labels[idx]   # 确保标签数量匹配
#                     frame = plot_one_box(box, frame, label=label)
#
#                 # 根据窗口大小调整帧的内容
#                 frame_resized = cv2.resize(frame, (window_width, window_height))
#                 cv2.imshow('Video Feed', frame_resized)
#
#                 # 按 'q' 键退出循环
#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     break
#                 # 检测到错误标签，进行帧保存处理
#                 error_idx = [idx for idx, label in enumerate(labels) if label in self.error_labels]
#                 if error_idx:
#                     current_time = time.time()  # 获取当前时间
#                     # 判断当前时间和上次保存时间差
#                     if current_time - self.last_save_time > 1:
#                         self.saved_frame_count = 0  # 每秒重置计数器
#
#                     # 每秒最多保存 5 帧
#                     if self.saved_frame_count < 5:
#                         print(f"************开始保存*********************")
#                         self.saved_frame_count += 1
#                         self.last_save_time = current_time  # 更新保存时间
#
#                         # 过滤出错误标签的标签和框
#                         labels_filtered = [label for idx, label in enumerate(labels) if idx in error_idx]
#                         boxes_filtered = [box for idx, box in enumerate(boxes) if idx in error_idx]
#
#                         # 创建XML文件并保存图像
#                         voc = CreateVoc()  # 创建VOC的类
#                         strftime = self.videos.video_start_time
#                         prefix = f'{self.camera_state}_{self.project}_{self.scene}__{strftime}_{self.videos.frame_id}'
#                         xml_name = f'{prefix}.xml'
#                         img_name = f'{prefix}.jpg'
#
#                         # 编码图像并保存
#                         success, encoded_image = cv2.imencode('.jpg', self.videos.current_image)
#                         if success:
#                             with open(os.path.join(self.save_path, img_name), 'wb') as f:
#                                 f.write(encoded_image.tobytes())
#
#                         # 保存到 VOC 格式的 XML
#                         voc.set_info(xml_name, labels_filtered, boxes_filtered, self.save_path)
#                         voc.write_xml()
#                     else:
#                         print(f"错误标签帧过多，已跳过此帧：{self.videos.frame_id}")
#
#         except Exception as e:
#             print(f"Error reading video: {e}")
#
#     def read_img_ch(self, path):
#         with open(path, 'rb') as file:
#             data = file.read()
#         data_array = np.frombuffer(data, dtype=np.uint8)
#         im_ = cv2.imdecode(data_array, cv2.IMREAD_COLOR)
#         return im_
#
#     def detect_images(self):
#         model = None
#         # 根据模型类型加载相应的推理模型
#         if self.model_type == "v5":
#             # model = Predictor("v5_model_path")  # 请根据需要替换路径
#             pass
#         elif self.model_type == "v8":
#             # model = Predictor("v8_model_path")  # 请根据需要替换路径
#             pass
#         elif self.model_type == "v10":
#             if self.model_path.endswith('.onnx'):
#                 print(self.model_path)
#                 model = V10OnnxPredictor(self.model_path, self.labels)  # 这是你现有的v10推理模型
#         else:
#             # 默认模型
#             pass
#         voc = CreateVoc()
#         for image in self.images.read_images():
#             labels, boxes = model.infer(image)
#             print("**************start*************")
#             # for idx, box in enumerate(boxes):
#             #     image = plot_one_box(box, image, label=labels[idx])
#             # img_name = self.images.current_image_name
#             # prefix, suffix = os.path.splitext(img_name)
#             current_time = time.strftime("%Y_%m_%d_%H%M%S", time.localtime())
#             prefix = f'{self.camera_state}_{self.project}_{self.scene}__{current_time}'
#             xml_name = f'{prefix}.xml'
#             img_name = f'{prefix}.jpg'
#             success, encoded_image = cv2.imencode('.jpg', self.images.current_image)
#
#             # # 检查编码是否成功
#             if success:
#                 # 将编码后的图像数据保存为文件
#                 with open(os.path.join(self.save_path, img_name), 'wb') as f:
#                     f.write(encoded_image.tobytes())
#             voc.set_info(xml_name, labels, boxes, self.images.current_image_folder)
#             voc.write_xml()
#
#     def run(self):
#         """
#         根据任务类型执行相应的检测方法。
#         """
#         if self.task == "video":
#             print("任务类型：视频检测")
#             self.detect_video()
#         elif self.task == "image":
#             print("任务类型：图像检测")
#             self.detect_images()
#         else:
#             print(f"未知的任务类型: {self.task}")
#
#
# if __name__ == "__main__":
#     main = Main()
#     main.run()
#     # 配置文件路径
#     # config_path = 'config.yaml'
#     #
#     # # 从配置文件中加载变量
#     # config = load_config(config_path)
#     #
#     # # 获取配置中的参数
#     # camera_state = config['camera_state']
#     # project = config['project']
#     # scene = config['scene']
#     # save_path = config['save_path']
#     # video_path = config['video_path']
#     # model_path = config['model_path']
#     # error_labels = config['error_labels']
#     # print(camera_state), print(project), print(scene), print(save_path), print(video_path)
