import numpy as np

Object = tuple[float, float, float, float, int, float]
Props = list[tuple[int, float]]
ObjectProps = tuple[Object, Props]


def yolov8_detect_props_postprocess_batch(
        batch_pred: np.ndarray[tuple[int, int, int], np.float32],
        iou_threshold: float,
        conf_threshold: float,
        num_cls: int,
        has_property: np.ndarray[int, bool],  # 每个类别是否存在属性
        property_groups: np.ndarray[int, int],
) -> list[list[ObjectProps]]:
    return postprocess_batch(
        batch_pred,
        iou_threshold,
        conf_threshold,
        num_cls,
        has_property,
        property_groups,
    )


def postprocess_batch(
        batch_pred: np.ndarray[tuple[int, int, int], np.float32],
        iou_threshold: float,
        conf_threshold: float,
        num_cls: int,
        has_property: np.ndarray[int, bool],  # 每个类别是否存在属性
        property_groups: np.ndarray[int, int],
        # src_width: float,
        # src_height: float,
        # scale: float,
        # pad_left: int,
        # pad_top: int,
) -> list[list[ObjectProps]]:
    bs, channels, grids = batch_pred.shape
    cls_end_idx = num_cls + 4
    obj_map_cache: list[list[ObjectProps]] = [[((0., 0., 0., 0., 0, 0.), [(0, 0.)])] for _ in range(num_cls)]
    filter_out_boxes_ls: list[list[ObjectProps]] = [[((0., 0., 0., 0., 0, 0.), [(0, 0.)])] for _ in range(num_cls)]
    results = []
    for pred in batch_pred:
        # pred: C, Grids
        [cache.clear() for cache in obj_map_cache]
        [cache.clear() for cache in filter_out_boxes_ls]
        for i_grid in range(grids):
            max_pred_conf = 0.
            max_pred_conf_idx = 0
            for i in range(4, cls_end_idx):
                cls_conf = pred[i, i_grid]
                if max_pred_conf < cls_conf:
                    max_pred_conf = cls_conf
                    max_pred_conf_idx = i - 4
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
                max_pred_conf
            )
            # cls_idx, cls_prob
            props: Props = [(0, 0.)]
            props.clear()
            if has_property[max_pred_conf_idx]:
                base_start = cls_end_idx
                for group_num in property_groups:
                    start = base_start
                    end = start + group_num
                    max_cls = 0xFFFFFFFF
                    max_cls_prob = 0.
                    for cls, i in enumerate(range(start, end)):
                        prob = pred[i, i_grid]
                        if prob > max_cls_prob:
                            max_cls = cls
                            max_cls_prob = prob
                    base_start = end
                    props.append((max_cls, max_cls_prob))
            obj_map_cache[max_pred_conf_idx].append((obj, props))

        # for cls, boxes in enumerate(obj_map_cache):
        for cls in range(num_cls):
            boxes = obj_map_cache[cls]
            if len(boxes) == 0:
                continue
            filter_out_boxes = filter_out_boxes_ls[cls]
            boxes.sort(reverse=True, key=lambda x_: x_[0][5])
            idx = [_i for _i in range(len(boxes))]
            while len(idx) > 0:
                good_idx = idx[0]
                filter_out_boxes.append(boxes[good_idx])
                tmp = idx.copy()
                idx.clear()
                good_xmin = boxes[good_idx][0][0]
                good_ymin = boxes[good_idx][0][1]
                good_xmax = boxes[good_idx][0][2]
                good_ymax = boxes[good_idx][0][3]
                good_width = good_xmax - good_xmin
                good_height = good_ymax - good_ymin
                for i in range(1, len(tmp)):
                    tmp_i = tmp[i]
                    temp_xmin = boxes[tmp_i][0][0]
                    temp_ymin = boxes[tmp_i][0][1]
                    temp_xmax = boxes[tmp_i][0][2]
                    temp_ymax = boxes[tmp_i][0][3]
                    temp_width = temp_xmax - temp_xmin
                    temp_height = temp_ymax - temp_ymin
                    inter_x1 = max(good_xmin, temp_xmin)
                    inter_y1 = max(good_ymin, temp_ymin)
                    inter_x2 = min(good_xmax, temp_xmax)
                    inter_y2 = min(good_ymax, temp_ymax)
                    w = max(inter_x2 - inter_x1, 0)
                    h = max(inter_y2 - inter_y1, 0)
                    inter_area = w * h
                    area_1 = good_width * good_height
                    area_2 = temp_width * temp_height
                    o = inter_area / (area_1 + area_2 - inter_area + 0.00001)
                    if o <= iou_threshold:
                        idx.append(tmp_i)
        res = []
        for filter_out_boxes in filter_out_boxes_ls:
            res.extend(filter_out_boxes)
        results.append(res)
    return results
