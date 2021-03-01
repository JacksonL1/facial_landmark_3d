import numpy as np
import torch
import cv2


def get_crop_box(data, scale, resolution, shape, type):
    h, w, _ = shape
    if type == "landmark":
        left = np.min(data[:, 0])
        right = np.max(data[:, 0])
        top = np.min(data[:, 1])
        bottom = np.max(data[:, 1])
        old_size = (right - left + bottom - top) / 2 * 1.1
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
    else:
        box = [int(i) for i in data[0]]
        x, y, x2, y2 = box
        old_size = (x2 - x + y2 - y) / 2
        center = np.array([x2 - (x2 - x) / 2.0, y2 - (y2 - y) / 2.0 + old_size * 0.12])
    h_e = h - abs(h / 2 - center[1]) * 2
    w_e = w - abs(w / 2 - center[0]) * 2
    small_size = min(h_e, w_e)
    size = min(int(old_size * scale), small_size)
    resize_ratio = size / resolution
    x = int(center[0] - size / 2)
    y = int(center[1] - size / 2)
    x2 = int(center[0] + size / 2)
    y2 = int(center[1] + size / 2)
    return [x, y, x2, y2], resize_ratio


def crop_image(image, box, resolution, device='cuda'):
    image = image[:, :, ::-1] / 255.
    l, t, r, b = box
    image = image[t:b, l:r]
    dst_image = cv2.resize(image, (resolution, resolution))
    dst_image = dst_image.transpose(2, 0, 1)
    return torch.tensor(dst_image).float().to(device)[None, ...]


def resize_para(ori_frame):
    w, h, c = ori_frame.shape
    d = max(w, h)
    scale_to = 640 if d >= 1280 else d / 2
    scale_to = max(64, scale_to)
    input_scale = d / scale_to
    w = int(w / input_scale)
    h = int(h / input_scale)
    image_info = [w, h, input_scale]
    return image_info


def roi_box_from_landmark(landmark):
    bx1 = np.min(landmark[:, 0].astype(np.int))
    by1 = np.min(landmark[:, 1].astype(np.int))
    bx2 = np.max(landmark[:, 0].astype(np.int))
    by2 = np.max(landmark[:, 1].astype(np.int))

    cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
    max_len = max(bx2 - bx1, by2 - by1) / 2

    lt = np.array([cx - max_len, cy - max_len])
    rb = np.array([cx + max_len, cy + max_len])
    length = np.linalg.norm(rb - lt) // 2
    ncx = (lt[0] + rb[0]) // 2
    ncy = (lt[1] + rb[1]) // 2

    roi = [ncx - length, ncy - length, ncx + length, ncy + length]
    return roi


def resize_frame(frame, min_size):
    h, w, c = frame.shape
    d = max(w, h)
    scale_to = min_size
    input_scale = d / scale_to
    r_w = int(w / input_scale)
    r_h = int(h / input_scale)
    frame = cv2.resize(frame, (r_w, r_h))
    return frame


def get_smooth_data(frame, landmark, frame_list, landmark_list, list_size):
    if len(frame_list) < list_size:
        frame_list.extend([frame] * list_size)
        landmark_list.extend([landmark] * list_size)
    else:
        frame_list.pop(0)
        landmark_list.pop(0)
        frame_list.append(frame)
        landmark_list.append(landmark)
    ave_landmark = np.mean(landmark_list, axis=0)
    pre_frame = frame_list[list_size - 1]
    return pre_frame, ave_landmark
