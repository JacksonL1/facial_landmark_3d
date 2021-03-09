import numpy as np
import torch
import cv2
from PIL import Image
import torchvision.transforms as transforms


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


def seg_crop_img(ori_image, rect, cropped_size):
    l, t, r, b = rect
    center_x = r - (r - l) // 2
    center_y = b - (b - t) // 2
    w = (r - l) * 1.2
    h = (b - t) * 1.2
    crop_size = max(w, h)
    if crop_size > cropped_size:
        crop_ly = int(max(0, center_y - crop_size // 2))
        crop_lx = int(max(0, center_x - crop_size // 2))
        crop_ly = int(min(ori_image.shape[0] - crop_size, crop_ly))
        crop_lx = int(min(ori_image.shape[1] - crop_size, crop_lx))
        crop_imgs = ori_image[crop_ly: int(crop_ly + crop_size), crop_lx: int(crop_lx + crop_size), :]
    else:

        crop_ly = int(max(0, center_y - cropped_size // 2))
        crop_lx = int(max(0, center_x - cropped_size // 2))
        crop_ly = int(min(ori_image.shape[0] - cropped_size, crop_ly))
        crop_lx = int(min(ori_image.shape[1] - cropped_size, crop_lx))
        crop_imgs = ori_image[crop_ly: int(crop_ly + cropped_size), crop_lx: int(crop_lx + cropped_size), :]
    # new_rect = [l - crop_lx, t - crop_ly, r - crop_lx, b - crop_ly]
    new_rect = [crop_lx, crop_ly, crop_lx + crop_size, crop_ly + crop_size]
    return crop_imgs, new_rect


def face_seg(img, net):
    h, w, _ = img.shape
    face_area = [1, 2, 3, 4, 5, 6, 10, 11, 12, 13]
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    w_ratio = w / 512
    h_ratio = h / 512
    pil_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    resize_pil_image = pil_image.resize((512, 512), Image.BILINEAR)
    tensor_image = to_tensor(resize_pil_image)
    tensor_image = torch.unsqueeze(tensor_image, 0)
    tensor_image = tensor_image.cuda()
    out = net(tensor_image)[0]
    parsing = out.squeeze(0).cpu().detach().numpy().argmax(0)
    vis_parsing_anno = parsing.copy().astype(np.uint8)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1]))
    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        if pi in face_area:
            index = np.where(vis_parsing_anno == pi)
            vis_parsing_anno_color[index[0], index[1]] = 1
    image_mask = vis_parsing_anno_color[..., None]
    image_mask[np.where(image_mask != 0)] = 1.

    return image_mask, [w_ratio, h_ratio]
