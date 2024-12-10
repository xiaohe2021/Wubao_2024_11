import cv2
import numpy as np  # 导入numpy用于数组操作
# 导入必要的库
import torch
from ultralytics import YOLO

from base import ONNXBaseModel
from local_utils import BBox
from local_utils import letterbox
from postprocess.yolov5_postprocess import postprocess_batch
from postprocess.yolov8_seg_postprocess import yolov8_seg_postprocess_batch
from preprocess.yolo_preprocess import yolo_det_preprocess


class V10OnnxPredictor(ONNXBaseModel):
    """YOLOv10 ONNX模型预测器"""

    def __init__(self, onnx_path, labels=None):
        super().__init__(onnx_path,
                         labels,
                         providers=['CUDAExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider'])
        # 只有在父类初始化后，才可以访问 self.metadata_names
        if labels is None:
            self.labels = self.get_metadata_name()  # 使用元数据中的 names 作为 labels
        print(f"Labels used in V10OnnxPredictor: {self.labels}")

    def infer(self, src_img):
        """执行预测"""
        try:
            input_hw = self.get_input_hw_shape()
            # 预处理图像
            img, scale = yolo_det_preprocess(src_img, input_hw)
            img = np.expand_dims(img, axis=0)
            # 运行推理
            pred = self.session.run(
                [self.get_output_names0()],
                {self.get_input_name(): img}
            )
            # 后处理获取结果
            # 确保预测结果在CPU上
            if isinstance(pred, np.ndarray):
                pred_cpu = pred
            else:
                pred_cpu = pred.cpu().numpy() if hasattr(pred, 'cpu') else pred
            # 处理每个检测结果
            labels = []
            boxes = []
            conf_threshold = 0.6
            for xmin, ymin, xmax, ymax, prob, cls in pred_cpu[0][0]:
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

        except Exception as e:
            print(f"推理错误: {str(e)}")
            raise


class V8SegOnnxPredictor(ONNXBaseModel):
    """YOLOv8 Seg ONNX模型预测器"""

    def __init__(self, onnx_path, labels=None):
        super().__init__(onnx_path,
                         labels,
                         providers=['CUDAExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider'])
        # 只有在父类初始化后，才可以访问 self.metadata_names
        if labels is None:
            self.labels = self.get_metadata_name()  # 使用元数据中的 names 作为 labels
        print(f"Labels used in V8SegOnnxPredictor: {self.labels}")

    def infer(self, src_img):
        """执行预测"""
        try:
            """后处理预测结果"""
            labels = []  # 存储标签
            boxes = []  # 存储边界框
            iou_threshold = 0.5
            conf_threshold = 0.5
            input_hw = self.get_input_hw_shape()
            img, scale = yolo_det_preprocess(src_img, input_hw)
            img = np.expand_dims(img, axis=0)

            # # B C H W
            src_height, src_width = src_img.shape[:2]
            input_wh = src_width, src_height
            # 执行推理
            inputs = {self.get_input_name(): img}
            det_output = self.session.run([self.get_output_names0()], inputs)
            # (1,37,12096)
            pred_det = torch.from_numpy(det_output[0]).to(torch.device('cuda'))
            pred_det = pred_det.cpu().numpy()

            seg_output = self.session.run([self.get_output_names1()], inputs)
            pred_seg = torch.from_numpy(seg_output[0]).to(torch.device('cuda'))
            pred_seg = pred_seg.cpu().numpy()

            first_pred_det = pred_det[0]
            first_pred_seg = pred_seg[0]  # C H W
            for o in yolov8_seg_postprocess_batch(pred_det, iou_threshold, conf_threshold)[0]:
                xmin, ymin, xmax, ymax, cls, prob, i_grid = o

                query_vec = first_pred_det[-32:, i_grid]  # 32
                masked = query_vec @ first_pred_seg.reshape(32, -1)  # (1, 144 * 256)
                masked = masked.reshape(first_pred_seg.shape[1:])  # 恢复成 (144, 256) 的形状
                mask_f32 = cv2.resize(masked, input_wh)  # 缩放掩码
                mask = (mask_f32 > 0.5).astype(np.uint8)  # 二值化处理
                # 找出掩码中的轮廓
                mask = mask[int(ymin):int(ymax), int(xmin):int(xmax)]
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # 找到最大轮廓
                if len(contours):
                    max_contour = max(contours, key=cv2.contourArea)[:, 0].astype(np.float32)
                    max_contour[:, 0] += xmin
                    max_contour[:, 1] += ymin
                    max_contour[:, 0] *= scale
                    max_contour[:, 1] *= scale
                    max_contour = max_contour.astype(np.int32)
                else:
                    max_contour = []

                xmin = float(max(0., xmin)) * scale
                ymin = float(max(0., ymin)) * scale
                xmax = float(min(src_width, xmax)) * scale
                ymax = float(min(src_height, ymax)) * scale
                box = [xmin, ymin, xmax, ymax]
                boxes.append(box)
                labels = self.labels[cls]
                # 转换边界框为整数列表
            boxes = [list(map(lambda x: int(x), box)) for box in boxes]

            return labels, boxes

        except Exception as e:
            print(f"推理错误: {str(e)}")
            raise


class V5OnnxPredictor(ONNXBaseModel):
    """YOLOv5 ONNX模型预测器"""

    def __init__(self, onnx_path, labels=None):
        super().__init__(onnx_path,
                         labels,
                         providers=['CUDAExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider'])
        # 只有在父类初始化后，才可以访问 self.metadata_names
        if labels is None:
            self.labels = self.get_metadata_name()  # 使用元数据中的 names 作为 labels
        print(f"Labels used in V5OnnxPredictor: {self.labels}")

    def infer(self, src_img):
        """执行预测"""
        try:
            """后处理预测结果"""
            labels = []  # 存储标签
            boxes = []  # 存储边界框

            iou_threshold = 0.5
            conf_threshold = 0.5
            input_hw = self.get_input_hw_shape()
            # 保存原始图像尺寸
            src_height, src_width = src_img.shape[:2]
            # 预处理图像
            input_tensor, scale = yolo_det_preprocess(src_img, input_hw)

            # 确保输入数据连续
            if not input_tensor.flags['C_CONTIGUOUS']:
                input_tensor = np.ascontiguousarray(input_tensor)

            # 运行推理
            outputs = self.session.run(
                self.get_output_names0(),
                {self.get_input_name(): input_tensor}
            )

            # 后处理获取结果

            for o in postprocess_batch(outputs,
                                       iou_threshold,
                                       conf_threshold)[0]:
                xmin, ymin, xmax, ymax, cls, prob, i_grid = o
                xmin = max(0., xmin) * scale
                ymin = max(0., ymin) * scale
                xmax = min(float(src_width), xmax) * scale
                ymax = min(float(src_height), ymax) * scale
                boxes = BBox(xmin, ymin, xmax, ymax)
                labels = self.labels[cls]
            return labels, boxes

        except Exception as e:
            print(f"推理错误: {str(e)}")
            raise


class V10TorchPredictor:
    """YOLOv5 PyTorch模型预测器"""

    def __init__(self, model_path, labels):
        self.model_path = model_path  # PyTorch模型路径
        self.labels = labels  # 动态标签集

        self.device = 'cpu' if not torch.cuda.is_available() else 'cuda:0'
        # 加载模型权重
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 使用 GPU 或 CPU
        # 加载模型
        self.model = YOLO(model_path)  # 直接通过 ultralytics 加载 YOLOv8 模型
        self.model.eval()  # 设置为推理模式
        self.model.to(self.device)


    def infer(self, img):
        """执行预测"""
        try:
            orig_shape = img.shape  # 保存原始图像尺寸
            input_hw = (640,640)
            # 预处理图像
            input_tensor, scale = yolo_det_preprocess(img,input_hw)

            # 将图像转换为 PyTorch 张量并移动到设备（GPU 或 CPU）
            input_tensor = torch.from_numpy(input_tensor).to(self.device)

            # 执行推理
            with torch.no_grad():  # 在推理时禁用梯度计算
                pred = self.model(input_tensor)
            print(f"Prediction shape: {pred.shape}")
            print(f"Prediction content: {pred}")
            # 后处理获取结果
            # 确保预测结果在CPU上
            if isinstance(pred, np.ndarray):
                pred_cpu = pred
            else:
                pred_cpu = pred.cpu().numpy() if hasattr(pred, 'cpu') else pred
            # 处理每个检测结果
            labels = []
            boxes = []
            conf_threshold = 0.6
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

        except Exception as e:
            print(f"推理错误: {str(e)}")
            raise


class V8TorchSegPredictor:
    """YOLOv8 Seg PyTorch模型预测器"""

    def __init__(self, model_path, labels):
        self.model_path = model_path  # PyTorch模型路径
        self.labels = labels  # 动态标签集

        self.device = 'cpu' if not torch.cuda.is_available() else 'cuda:0'
        # 加载模型权重
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 使用 GPU 或 CPU
        # 加载模型
        self.model = YOLO(model_path)  # 直接通过 ultralytics 加载 YOLOv8 模型
        self.model.eval()  # 设置为推理模式
        self.model.to(self.device)


    def infer(self, img):
        """执行预测"""
        try:
            orig_shape = img.shape  # 保存原始图像尺寸
            # 预处理图像
            input_tensor, scale = yolo_det_preprocess(img)

            # 将图像转换为 PyTorch 张量并移动到设备（GPU 或 CPU）
            input_tensor = torch.from_numpy(input_tensor).to(self.device)

            # 执行推理
            with torch.no_grad():  # 在推理时禁用梯度计算
                pred = self.model(input_tensor)
            # print(f"Prediction shape: {pred.shape}")
            # print(f"Prediction content: {pred}")
            # 后处理获取结果
            labels, boxes = yolov8_seg_postprocess_batch(pred, 0.5, 0.5)
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

    def infer(self, img):
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
