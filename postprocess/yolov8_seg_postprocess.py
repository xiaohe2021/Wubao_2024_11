import numpy as np
from typing import List, Tuple

# 定义目标类型
Object = Tuple[float, float, float, float, int, float, int]  # xmin, ymin, xmax, ymax, cls, score, grid_index


def yolov8_seg_postprocess_batch(
        batch_pred: np.ndarray,
        iou_threshold: float,
        conf_threshold: float,
) -> List[List[Object]]:
    """
    对批量预测结果进行后处理。

    参数:
        batch_pred (np.ndarray): 批量预测的结果，形状为 (batch_size, channels, grids)。
        iou_threshold (float): 用于过滤的 IoU 阈值。
        conf_threshold (float): 用于过滤的置信度阈值。

    返回:
        List[List[Object]]: 每个批次的处理结果对象列表。
    """
    batch_size, channels, grids = batch_pred.shape
    num_cls = channels  - 4 - 32  # 假设前4个通道是边界框预测 (cx, cy, cw, ch)
    cls_end_idx = num_cls + 4

    results = []  # 存储每个批次的处理结果
    for pred in batch_pred:  # 遍历每个批次
        # 初始化每个类别的目标缓存
        obj_map_cache = [[] for _ in range(num_cls)]
        filter_out_boxes_ls = [[] for _ in range(num_cls)]

        # 遍历每个网格点
        for i_grid in range(grids):
            max_pred_conf = 0.0  # 最大的类别置信度
            max_pred_conf_idx = 0  # 最大置信度对应的类别索引
            # 遍历所有类别，找到置信度最高的类别
            for i in range(4, cls_end_idx):
                cls_conf = pred[i, i_grid]
                if max_pred_conf < cls_conf:
                    max_pred_conf = cls_conf
                    max_pred_conf_idx = i - 4

            # 如果最高置信度低于阈值，则跳过
            if max_pred_conf < conf_threshold:
                continue

            # 提取边界框信息
            cx = pred[0, i_grid]  # 中心点 x 坐标
            cy = pred[1, i_grid]  # 中心点 y 坐标
            cw = pred[2, i_grid]  # 宽度
            ch = pred[3, i_grid]  # 高度
            half_w = cw * 0.5
            half_h = ch * 0.5
            # 构建目标对象 (xmin, ymin, xmax, ymax, cls, score, grid_index)
            obj: Object = (
                max(cx - half_w, 0.),
                max(cy - half_h, 0.),
                (cx + half_w),
                (cy + half_h),
                max_pred_conf_idx,
                max_pred_conf,
                i_grid,
            )
            obj_map_cache[max_pred_conf_idx].append(obj)

        # 对每个类别的目标进行非极大值抑制 (NMS)
        for cls, boxes in enumerate(obj_map_cache):
            if not boxes:
                continue
            filter_out_boxes = filter_out_boxes_ls[cls]
            # 按置信度从高到低排序
            boxes.sort(reverse=True, key=lambda x_: x_[5])
            idx = list(range(len(boxes)))  # 初始化索引列表

            # NMS 过程
            while idx:
                good_idx = idx[0]
                filter_out_boxes.append(boxes[good_idx])
                good_box = boxes[good_idx]
                good_xmin, good_ymin, good_xmax, good_ymax = good_box[:4]
                good_area = (good_xmax - good_xmin) * (good_ymax - good_ymin)

                tmp_idx = idx.copy()
                idx.clear()
                for i in tmp_idx[1:]:
                    temp_box = boxes[i]
                    temp_xmin, temp_ymin, temp_xmax, temp_ymax = temp_box[:4]
                    temp_area = (temp_xmax - temp_xmin) * (temp_ymax - temp_ymin)

                    # 计算 IoU
                    inter_xmin = max(good_xmin, temp_xmin)
                    inter_ymin = max(good_ymin, temp_ymin)
                    inter_xmax = min(good_xmax, temp_xmax)
                    inter_ymax = min(good_ymax, temp_ymax)
                    inter_w = max(0, inter_xmax - inter_xmin)
                    inter_h = max(0, inter_ymax - inter_ymin)
                    inter_area = inter_w * inter_h
                    iou = inter_area / (good_area + temp_area - inter_area + 1e-5)

                    # 如果 IoU 小于阈值，保留该索引
                    if iou <= iou_threshold:
                        idx.append(i)

        # 合并所有类别的过滤结果
        res = []
        for filter_out_boxes in filter_out_boxes_ls:
            res.extend(filter_out_boxes)
        results.append(res)

    return results