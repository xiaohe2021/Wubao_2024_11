from typing import List, Tuple

import numpy as np

# 定义目标类型
Object = Tuple[float, float, float, float, int, float, int]  # xmin, ymin, xmax, ymax, cls, score, grid_index


def yolov5_detect_postprocess_batch(batch_pred: np.ndarray,  # shape: (batch_size, channels, grids)
                      iou_threshold: float,
                      conf_threshold: float,
                      ) -> List[List[Object]]:
    bs, channels, grids = batch_pred.shape
    num_cls = channels - 5
    cls_end_idx = num_cls + 5
    # print(f"cls_end_idx:", cls_end_idx)
    obj_map_cache: List[List[Object]] = [[] for _ in range(num_cls)]
    filter_out_boxes_ls: List[List[Object]] = [[] for _ in range(num_cls)]
    results = []

    for pred in batch_pred:
        # Clear cache for each prediction
        for cache in obj_map_cache:
            cache.clear()
        for cache in filter_out_boxes_ls:
            cache.clear()

        for i_grid in range(grids):
            obj_prob = pred[4, i_grid]
            if obj_prob < conf_threshold:
                continue

            # Find the class with the maximum confidence
            cls_conf = pred[5:cls_end_idx, i_grid]
            max_pred_conf_idx = np.argmax(cls_conf)
            max_pred_conf = cls_conf[max_pred_conf_idx]

            if max_pred_conf < conf_threshold:
                continue

            cx = pred[0, i_grid]
            cy = pred[1, i_grid]
            cw = pred[2, i_grid]
            ch = pred[3, i_grid]
            half_w = cw * 0.5
            half_h = ch * 0.5
            # xmin, ymin, xmax, ymax, cls, score
            obj: Object = (
                max((cx - half_w), 0.),
                max((cy - half_h), 0.),
                (cx + half_w),
                (cy + half_h),
                max_pred_conf_idx,
                max_pred_conf,
                i_grid,
            )
            obj_map_cache[max_pred_conf_idx].append(obj)

        # Perform Non-Maximum Suppression (NMS)
        for cls in range(num_cls):
            boxes = obj_map_cache[cls]
            if len(boxes) == 0:
                continue

            filter_out_boxes = filter_out_boxes_ls[cls]
            boxes.sort(key=lambda x_: x_[5], reverse=True)  # Sort by confidence

            idx = list(range(len(boxes)))
            while len(idx) > 0:
                good_idx = idx[0]
                filter_out_boxes.append(boxes[good_idx])
                tmp = idx.copy()
                idx.clear()

                good_xmin, good_ymin, good_xmax, good_ymax = boxes[good_idx][:4]
                good_width = good_xmax - good_xmin
                good_height = good_ymax - good_ymin

                for i in range(1, len(tmp)):
                    temp_idx = tmp[i]
                    temp_xmin, temp_ymin, temp_xmax, temp_ymax = boxes[temp_idx][:4]
                    temp_width = temp_xmax - temp_xmin
                    temp_height = temp_ymax - temp_ymin

                    # Calculate IoU
                    inter_x1 = max(good_xmin, temp_xmin)
                    inter_y1 = max(good_ymin, temp_ymin)
                    inter_x2 = min(good_xmax, temp_xmax)
                    inter_y2 = min(good_ymax, temp_ymax)

                    w = max(inter_x2 - inter_x1, 0)
                    h = max(inter_y2 - inter_y1, 0)
                    inter_area = w * h
                    area_1 = good_width * good_height
                    area_2 = temp_width * temp_height
                    o = inter_area / (area_1 + area_2 - inter_area + 1e-5)

                    if o <= iou_threshold:
                        idx.append(temp_idx)

    res = []
    for filter_out_boxes in filter_out_boxes_ls:
        res.extend(filter_out_boxes)
    results.append(res)

    return results


def postprocess_batch_1(
        batch_pred: np.ndarray,  # shape: (batch_size, channels, grids)
        iou_threshold: float,
        conf_threshold: float,
        num_cls: int,
        src_width: float,
        src_height: float,
        scale: float,
        pad_left: int,
        pad_top: int,
) -> List[List[Object]]:
    bs, channels, grids = batch_pred.shape
    cls_end_idx = num_cls + 5
    obj_map_cache: List[List[Object]] = [[] for _ in range(num_cls)]
    filter_out_boxes_ls: List[List[Object]] = [[] for _ in range(num_cls)]
    results = []

    for pred in batch_pred:
        # Clear cache for each prediction
        for cache in obj_map_cache:
            cache.clear()
        for cache in filter_out_boxes_ls:
            cache.clear()

        for i_grid in range(grids):
            obj_prob = pred[4, i_grid]
            if obj_prob < conf_threshold:
                continue

            # Find the class with the maximum confidence
            cls_conf = pred[5:cls_end_idx, i_grid]
            max_pred_conf_idx = np.argmax(cls_conf)
            max_pred_conf = cls_conf[max_pred_conf_idx]

            if max_pred_conf < conf_threshold:
                continue

            cx = pred[0, i_grid]
            cy = pred[1, i_grid]
            cw = pred[2, i_grid]
            ch = pred[3, i_grid]
            half_w = cw * 0.5
            half_h = ch * 0.5

            cx -= pad_left
            cy -= pad_top

            # xmin, ymin, xmax, ymax, cls, score
            obj: Object = (
                max((cx - half_w) * scale, 0.),
                max((cy - half_h) * scale, 0.),
                min((cx + half_w) * scale, src_width),
                min((cy + half_h) * scale, src_height),
                max_pred_conf_idx,
                max_pred_conf,
                i_grid,
            )
            obj_map_cache[max_pred_conf_idx].append(obj)

        # Perform Non-Maximum Suppression (NMS)
        for cls in range(num_cls):
            boxes = obj_map_cache[cls]
            if len(boxes) == 0:
                continue

            filter_out_boxes = filter_out_boxes_ls[cls]
            boxes.sort(key=lambda x_: x_[5], reverse=True)  # Sort by confidence

            idx = list(range(len(boxes)))
            while len(idx) > 0:
                good_idx = idx[0]
                filter_out_boxes.append(boxes[good_idx])
                tmp = idx.copy()
                idx.clear()

                good_xmin, good_ymin, good_xmax, good_ymax = boxes[good_idx][:4]
                good_width = good_xmax - good_xmin
                good_height = good_ymax - good_ymin

                for i in range(1, len(tmp)):
                    temp_idx = tmp[i]
                    temp_xmin, temp_ymin, temp_xmax, temp_ymax = boxes[temp_idx][:4]
                    temp_width = temp_xmax - temp_xmin
                    temp_height = temp_ymax - temp_ymin

                    # Calculate IoU
                    inter_x1 = max(good_xmin, temp_xmin)
                    inter_y1 = max(good_ymin, temp_ymin)
                    inter_x2 = min(good_xmax, temp_xmax)
                    inter_y2 = min(good_ymax, temp_ymax)

                    w = max(inter_x2 - inter_x1, 0)
                    h = max(inter_y2 - inter_y1, 0)
                    inter_area = w * h
                    area_1 = good_width * good_height
                    area_2 = temp_width * temp_height
                    o = inter_area / (area_1 + area_2 - inter_area + 1e-5)

                    if o <= iou_threshold:
                        idx.append(temp_idx)

    res = []
    for filter_out_boxes in filter_out_boxes_ls:
        res.extend(filter_out_boxes)
    results.append(res)

    return results
