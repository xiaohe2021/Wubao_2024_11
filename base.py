from abc import abstractmethod, ABC

import numpy as np
import onnx
import onnxruntime as ort


class ONNXBaseModel(ABC):
    def __init__(self, onnx_path, labels, providers=None):
        self.onnx_path = onnx_path
        self.providers = providers if providers else ['CPUExecutionProvider']
        print(f"self.providers = {self.providers}")
        # 设置日志等级

        ort.set_default_logger_severity(3)
        self.session = ort.InferenceSession(self.onnx_path, providers=self.providers,)
        self.labels = labels
        # 缓存模型相关属性
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.input_hw_shape = self.session.get_inputs()[0].shape[2:]
        self.output_shapes = [i.shape for i in self.session.get_outputs()]
        self.input_types = [i.type for i in self.session.get_inputs()]  # 获取输入类型
        self.output_types = [output.type for output in self.session.get_outputs()]
        self.metadata_names = self._extract_metadata_names()
        if not self.metadata_names:
            print("No metadata names found in model.")
        print(f"Extracted 'names' metadata: {self.metadata_names}")

        # 打印缓存的属性信息
        print(f"Model initialized with input_name: {self.input_name}")
        print(f"Output names: {self.output_names}")
        print(f"Input HW shape: {self.input_hw_shape}")
        print(f"Output shapes: {self.output_shapes}")
        print(f"Input types: {self.input_types}")  # 打印输入类型
        print(f"Output types: {self.output_types}")  # 打印输出类型

    def _extract_metadata_names(self):
        """私有方法：从模型元数据中解析 'names' """
        try:
            # 加载 ONNX 模型文件
            model = onnx.load(self.onnx_path)
            metadata_props = model.metadata_props
            for prop in metadata_props:
                if prop.key == 'names':
                    return eval(prop.value)  # 转换字符串为字典或其他数据结构
        except Exception as e:
            print(f"Error extracting 'names' metadata: {e}")
        return None

    def get_input_name(self):
        return self.input_name

    def get_output_names0(self):
        return self.output_names[0]

    def get_output_names1(self):
        return self.output_names[1]

    def get_input_hw_shape(self):
        return self.input_hw_shape

    def get_input_wh_shape(self):
        return self.input_hw_shape[::-1]

    def get_output_shape(self):
        return self.output_shapes

    def get_metadata_name(self):
        return self.metadata_names

    @abstractmethod
    def infer(self, srcimg: np.ndarray):
        """执行推理"""
        raise NotImplementedError("子类必须实现此方法来推理")
