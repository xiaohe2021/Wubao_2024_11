import numpy as np  # 导入numpy用于数组操作
# 导入必要的库
import onnxruntime  # 导入ONNX运行时
import torch
from local_utils import letterbox
import cv2

class V10OnnxPredictor:
    """YOLOv10 ONNX模型预测器"""

    def __init__(self, model_path, labels):
        self.model_path = model_path  # ONNX模型路径
        self.labels = labels  # 动态标签集
        # 配置CUDA执行提供程序
        providers = [
            ('CUDAExecutionProvider', {
                'device_id': 0,  # GPU设备ID
                'arena_extend_strategy': 'kNextPowerOfTwo',  # 内存分配策略
                'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # GPU内存限制
                'cudnn_conv_algo_search': 'EXHAUSTIVE',  # cuDNN卷积算法搜索策略
                'do_copy_in_default_stream': True,  # 在默认流中复制
            })
        ]

        # 初始化ONNX运行时会话
        self.session = onnxruntime.InferenceSession(
            model_path,
            providers=providers
        )

        # 获取模型输入输出信息
        self.input_name = self.session.get_inputs()[0].name  # 输入节点名称
        self.input_shape = self.session.get_inputs()[0].shape  # 输入形状
        self.output_names = [output.name for output in self.session.get_outputs()]  # 输出节点名称

        # 配置会话选项
        sess_options = onnxruntime.SessionOptions()
        sess_options.log_severity_level = 3  # 设置日志级别
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL  # 启用所有图优化
        self.session.get_session_options().execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL  # 设置顺序执行模式

    def yolo_det_preprocess(self, img: np.ndarray) -> tuple[np.ndarray, float]:
        print("*************")
        """预处理图像用于YOLO检测"""

        input_hw = (self.input_shape[2], self.input_shape[3])  # 获取输入尺寸
        if isinstance(self.input_shape[2], str) or isinstance(self.input_shape[3], str):
            print(f"input_hw is a string: {input_hw}")
            # input_hw = (576, 1024)
            input_hw = (383,640)
        print(f"input_hw is {input_hw}")

        # 调整图像大小
        img, scale = letterbox(img, new_hw=input_hw)

        # 确保图像类型为 float32，且能够进行归一化
        if img.dtype != np.float32:
            print(f"Warning: img.dtype is {img.dtype}, converting to float32")
            img = img.astype(np.float32)  # 转换为 float32 类型
        img = img / 255.  # 归一化

        # 检查图像的通道数
        if len(img.shape) == 2:  # 灰度图像（1通道）
            print("Input image is grayscale. Converting to RGB.")
            img = np.expand_dims(img, axis=-1)  # 将灰度图像从 (height, width) 转换为 (height, width, 1)
            img = np.repeat(img, 3, axis=-1)  # 将灰度图复制为 3 通道，变为 RGB 图像

        elif len(img.shape) == 3 and img.shape[2] == 3:  # 如果是 RGB 图像
            img = img[:, :, ::-1]  # BGR 转 RGB（假设输入是 BGR）
        # 确保图像的形状为 (height, width, 3)
        img = img.transpose(2, 0, 1)  # 转换为 (3, height, width)
        # print(f"Image shape after processing: {img.shape}")
        # 返回处理后的图像和缩放比例
        return np.ascontiguousarray(img[None, :, :, :]), scale  # 增加 batch 维度并返回

    def postprocess(self, pred, scale):
        """后处理预测结果"""
        labels = []  # 存储标签
        boxes = []  # 存储边界框
        conf_threshold = 0.6  # 置信度阈值

        # 确保预测结果在CPU上
        if isinstance(pred, np.ndarray):
            pred_cpu = pred
        else:
            pred_cpu = pred.cpu().numpy() if hasattr(pred, 'cpu') else pred
        # print(f"pred_cpu 类型: {type(pred_cpu)}")
        # print(f"pred_cpu 形状: {pred_cpu.shape if hasattr(pred_cpu, 'shape') else '无形状'}")
        # print(f"pred_cpu 内容: {pred_cpu}")
        # 处理每个检测结果
        for xmin, ymin, xmax, ymax, prob, cls in pred_cpu[0]:
            if prob < conf_threshold:  # 过滤低置信度的检测
                break
            cls = int(cls)  # 类别索引

            label = self.labels[cls]  # 生成类别标签
            # 还原边界框坐标到原始图像尺寸
            xmin = float(xmin) * scale
            ymin = float(ymin) * scale
            xmax = float(xmax) * scale
            ymax = float(ymax) * scale
            box = [xmin, ymin, xmax, ymax]
            labels.append(label)
            boxes.append(box)
        # 转换边界框为整数列表
        boxes = [list(map(lambda x: int(x), box)) for box in boxes]
        return labels, boxes

    def predict(self, img):
        """执行预测"""
        try:
            orig_shape = img.shape  # 保存原始图像尺寸
            res = []
            # 预处理图像
            input_tensor, scale = self.yolo_det_preprocess(img)

            # 确保输入数据连续
            if not input_tensor.flags['C_CONTIGUOUS']:
                input_tensor = np.ascontiguousarray(input_tensor)

            # 运行推理
            outputs = self.session.run(
                self.output_names,
                {self.input_name: input_tensor}
            )
            # 后处理获取结果
            labels, boxes = self.postprocess(outputs[0], scale)
            # res.append((labels, boxes))
            return labels, boxes

        except Exception as e:
            print(f"推理错误: {str(e)}")
            raise


class V5OnnxPredictor:
    """YOLOv5 ONNX模型预测器"""

    def __init__(self, model_path, labels):
        self.model_path = model_path  # ONNX模型路径
        self.labels = labels  # 动态标签集

        # 配置CUDA执行提供程序
        providers = [
            ('CUDAExecutionProvider', {
                'device_id': 0,  # GPU设备ID
                'arena_extend_strategy': 'kNextPowerOfTwo',  # 内存分配策略
                'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # GPU内存限制
                'cudnn_conv_algo_search': 'EXHAUSTIVE',  # cuDNN卷积算法搜索策略
                'do_copy_in_default_stream': True,  # 在默认流中复制
            })
        ]

        # 初始化ONNX运行时会话
        self.session = onnxruntime.InferenceSession(
            model_path,
            providers=providers
        )

        # 获取模型输入输出信息
        self.input_name = self.session.get_inputs()[0].name  # 输入节点名称
        self.input_shape = self.session.get_inputs()[0].shape  # 输入形状
        self.output_names = [output.name for output in self.session.get_outputs()]  # 输出节点名称

        # 配置会话选项
        sess_options = onnxruntime.SessionOptions()
        sess_options.log_severity_level = 3  # 设置日志级别
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL  # 启用所有图优化
        self.session.get_session_options().execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL  # 设置顺序执行模式

    def yolo_det_preprocess(self, img: np.ndarray) -> tuple[np.ndarray, float]:
        """预处理图像用于YOLO检测"""
        input_hw = (self.input_shape[2], self.input_shape[3])  # 获取输入尺寸
        if isinstance(self.input_shape[2], str) or isinstance(self.input_shape[3], str):
            # 若模型形状为字符串，使用默认尺寸 (640, 640)
            input_hw = (640, 640)

        # 调整图像大小并保持长宽比
        img, scale = letterbox(img, new_hw=input_hw)

        # 确保图像类型为 float32，且能够进行归一化
        img = img.astype(np.float32) / 255.0  # 归一化到 [0, 1]

        # 检查图像的通道数
        if len(img.shape) == 2:  # 灰度图像
            img = np.expand_dims(img, axis=-1)  # 增加通道维度
            img = np.repeat(img, 3, axis=-1)  # 转为 3 通道
        elif len(img.shape) == 3 and img.shape[2] == 3:  # RGB图像
            img = img[:, :, ::-1]  # 转换为RGB格式（如果是BGR格式）

        # 转换为 NCHW 格式
        img = img.transpose(2, 0, 1)
        return np.ascontiguousarray(img[None, :, :, :]), scale  # 增加batch维度

    def postprocess(self, pred, scale):
        """后处理预测结果"""
        labels = []  # 存储标签
        boxes = []  # 存储边界框
        conf_threshold = 0.6  # 置信度阈值

        # 确保预测结果在CPU上
        pred_cpu = pred.cpu().numpy() if hasattr(pred, 'cpu') else pred

        # 处理每个检测结果
        for xmin, ymin, xmax, ymax, prob, cls in pred_cpu[0]:
            if prob < conf_threshold:  # 过滤低置信度的检测
                break
            cls = int(cls)  # 类别索引

            label = self.labels[cls]  # 生成类别标签
            # 还原边界框坐标到原始图像尺寸
            xmin = float(xmin) * scale
            ymin = float(ymin) * scale
            xmax = float(xmax) * scale
            ymax = float(ymax) * scale
            box = [xmin, ymin, xmax, ymax]
            labels.append(label)
            boxes.append(box)

        # 转换边界框为整数列表
        boxes = [list(map(lambda x: int(x), box)) for box in boxes]
        return labels, boxes

    def predict(self, img):
        """执行预测"""
        try:
            orig_shape = img.shape  # 保存原始图像尺寸
            # 预处理图像
            input_tensor, scale = self.yolo_det_preprocess(img)

            # 确保输入数据连续
            if not input_tensor.flags['C_CONTIGUOUS']:
                input_tensor = np.ascontiguousarray(input_tensor)

            # 运行推理
            outputs = self.session.run(
                self.output_names,
                {self.input_name: input_tensor}
            )

            # 后处理获取结果
            labels, boxes = self.postprocess(outputs[0], scale)
            return labels, boxes

        except Exception as e:
            print(f"推理错误: {str(e)}")
            raise


class V5TorchPredictor:
    """YOLOv5 PyTorch模型预测器"""

    def __init__(self, model_path, labels):
        self.model_path = model_path  # PyTorch模型路径
        self.labels = labels  # 动态标签集

        self.device = 'cpu' if not torch.cuda.is_available() else 'cuda:0'
        # 加载模型权重
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 使用 GPU 或 CPU
        self.model = self._load_model_weights(model_path)
        self.model.eval()  # 设置为推理模式
        self.model.to(self.device)

    def _load_model_weights(self, model_path):
        """加载YOLOv5权重"""
        try:
            # 尝试直接加载权重
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, device=self.device)
            return model
        except Exception as e:
            print(f"加载模型失败: {e}")
            raise

    def yolo_det_preprocess(self, img: np.ndarray) -> tuple[np.ndarray, float]:
        """预处理图像用于YOLO检测"""
        input_hw = (640, 640)  # YOLOv5默认输入大小为 640x640

        # 调整图像大小并保持长宽比
        img, scale = letterbox(img, new_hw=input_hw)

        # 确保图像类型为 float32，且能够进行归一化
        img = img.astype(np.float32) / 255.0  # 归一化到 [0, 1]

        # 检查图像的通道数
        if len(img.shape) == 2:  # 灰度图像
            img = np.expand_dims(img, axis=-1)  # 增加通道维度
            img = np.repeat(img, 3, axis=-1)  # 转为 3 通道
        elif len(img.shape) == 3 and img.shape[2] == 3:  # RGB图像
            img = img[:, :, ::-1]  # 转换为RGB格式（如果是BGR格式）

        # 转换为 NCHW 格式
        img = img.transpose(2, 0, 1)
        return np.ascontiguousarray(img[None, :, :, :]), scale  # 增加batch维度

    def postprocess(self, pred, scale):
        # 预测形状: [batch_size, num_predictions, 13]
        # 每个预测包含: [x1, y1, x2, y2, confidence, class_1, class_2, ..., class_N]

        # 提取边界框 (x1, y1, x2, y2)
        boxes = pred[..., :4]  # 前4个是边界框坐标

        # 提取置信度
        confidence = pred[..., 4]  # 第5个是置信度

        # 提取类别得分，取后面的 80 个类的分数（如果是 YOLOv5，通常有 80 类）
        class_scores = pred[..., 5:]  # 余下的表示每个类别的得分

        # 获取预测的标签（类别索引）
        labels = class_scores.argmax(dim=-1)  # 获取类别得分最大的索引作为标签

        # 根据置信度筛选框，使用一定的阈值进行后处理
        # 例如：去掉低置信度的框（此步骤是可选的，根据你的需求调整）
        threshold = 0.5  # 设置置信度阈值
        mask = confidence > threshold  # 选择置信度高于阈值的框

        boxes = boxes[mask]  # 筛选出对应的边界框
        labels = labels[mask]  # 筛选出对应的标签

        # 根据需要进一步调整尺度 (如果有缩放等情况)
        # 对 boxes 进行尺度变换，确保其尺寸和原图一致（如果 scale 是必要的）
        if scale:
            # 如果有缩放，按比例调整 boxes 的坐标
            boxes /= scale

        return labels, boxes

    def predict(self, img):
        """执行预测"""
        try:
            orig_shape = img.shape  # 保存原始图像尺寸
            # 预处理图像
            input_tensor, scale = self.yolo_det_preprocess(img)

            # 将图像转换为 PyTorch 张量并移动到设备（GPU 或 CPU）
            input_tensor = torch.from_numpy(input_tensor).to(self.device)

            # 执行推理
            with torch.no_grad():  # 在推理时禁用梯度计算
                pred = self.model(input_tensor)
            # print(f"Prediction shape: {pred.shape}")
            # print(f"Prediction content: {pred}")
            # 后处理获取结果
            labels, boxes = self.postprocess(pred, scale)
            return labels, boxes

        except Exception as e:
            print(f"推理错误: {str(e)}")
            raise
