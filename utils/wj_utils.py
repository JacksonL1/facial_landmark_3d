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
    # ori_image = (ori_image.detach().cpu().numpy().transpose(2, 3, 1, 0).squeeze()[:, :, ::-1] * 255).astype(np.uint8)
    l, t, r, b = rect
    r = r - l
    b = b - t
    l = 0
    t = 0
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
    eye_area = [2, 3, 4, 5, 6, 11, 12, 13]
    face_area = [1, 2, 3, 4, 5, 6, 10, 11, 12, 13]
    # face_area = [1]
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    pil_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    resize_pil_image = pil_image.resize((512, 512), Image.BILINEAR)
    tensor_image = to_tensor(resize_pil_image)
    tensor_image = torch.unsqueeze(tensor_image, 0)
    tensor_image = tensor_image.cuda()
    out = net(tensor_image)[0]
    parsing = out.squeeze(0).cpu().detach().numpy().argmax(0)
    vis_parsing_anno = parsing.copy().astype(np.uint8)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1]))
    eye_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1]))
    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        if pi in face_area:
            index = np.where(vis_parsing_anno == pi)
            vis_parsing_anno_color[index[0], index[1]] = 1

    for pi in range(1, num_of_class + 1):
        if pi in eye_area:
            index = np.where(vis_parsing_anno == pi)
            eye_color[index[0], index[1]] = 1
    image_mask = vis_parsing_anno_color[..., None]
    image_mask[np.where(image_mask != 0)] = 1.
    image_mask = cv2.resize(image_mask, (h, w))

    eye_mask = eye_color[..., None]
    eye_mask[np.where(eye_mask != 0)] = 1.
    eye_mask = cv2.resize(eye_mask, (h, w))

    image_mask = image_mask[..., None]
    eye_mask = eye_mask[..., None]

    image_mask = np.concatenate((image_mask, image_mask, image_mask), axis=-1)
    eye_mask = np.concatenate((eye_mask, eye_mask, eye_mask), axis=-1)

    image_mask = torch.from_numpy(image_mask[None, ...].transpose(0, 3, 1, 2)).cuda()
    eye_mask = torch.from_numpy(eye_mask[None, ...].transpose(0, 3, 1, 2)).cuda()
    return image_mask, eye_mask


def color_transfer(ori, img):
    ori = cv2.cvtColor(ori, cv2.COLOR_BGR2LAB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    ori_avg = np.mean(ori, axis=(0, 1))
    img_avg = np.mean(img, axis=(0, 1))

    ori_std = np.std(ori, axis=(0, 1))
    img_std = np.std(img, axis=(0, 1))

    img = (img - img_avg) / img_std * ori_std + ori_avg
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    return img


def tex_completion(ori_texture, face_mask, thresh=80, mean_scalar=150, k=5):
    '''
        补图片中没有信息的头部区域信息（逻辑待优化）
        ori_texture： 通过预测，重建的头部区域信息
        face_mask：人脸部位mask
        thresh： 人脸轮廓边缘向内缩小的像素值
        mean_scalar： 轮廓边缘如果RGB值小于mean_scalar会被内部像素覆盖
        k： 取轮廓内部k个像素向外覆盖
    '''
    new_texture = ori_texture.copy()
    h, w, _ = new_texture.shape
    below_eyeball = 810
    t, b = 10000, 0

    from tqdm import tqdm
    for i in tqdm(range(below_eyeball, h)):
        if np.max(new_texture[i, :]) == 0:
            continue
        edge = np.nonzero(np.mean(new_texture[i, :], axis=1))[0]
        if edge.shape[0] < k:
            new_texture[i] = 0
            if i < h // 2:
                t = i
            else:
                b = i
            break
        left_edge = edge[edge < w // 2]
        right_edge = edge[edge > w // 2]
        if left_edge.shape[0] > 0:
            l_num = np.min(left_edge) + thresh
        else:
            l_num = w // 2
        if right_edge.shape[0] > 0:
            r_num = np.max(right_edge) - thresh
        else:
            r_num = w // 2

        for j in range(l_num, -1, -1):
            area = []
            idx = 1
            while len(area) < k:
                curr = new_texture[i][j + idx]
                if np.mean(curr) < mean_scalar:
                    area = []
                else:
                    area.append(curr)
                idx += 1
                if j + idx >= w:
                    break
            if len(area) < k:
                new_texture[i] = 0
                if i < h // 2:
                    t = i
                else:
                    b = i
                break
            else:
                j = j - k - 1 + idx
                area = np.array(area)
                new_texture[i][:j] = np.concatenate((np.tile(area, (j // k, 1)), area[:j % k]))
                break

        for j in range(r_num, w):
            area = []
            idx = 1
            while len(area) < k:
                curr = new_texture[i][j - idx]
                if np.mean(curr) < mean_scalar:
                    area = []
                else:
                    area.append(curr)
                idx += 1
                if j - idx <= 0:
                    break
            if len(area) < k:
                new_texture[i] = 0
                if i < h // 2:
                    t = i
                else:
                    b = i
                break
            else:
                area = np.array(area)
                j = j - idx + k + 1
                right_j = w - j
                new_texture[i][j:] = np.concatenate((np.tile(area, (right_j // k, 1)), area[:right_j % k]))
                break
        t = min(i, t)
        b = max(i, b)
    t = t + 12
    print(t, b)
    out_face_mask = 1 - face_mask.copy()
    out_face_mask[below_eyeball:, ...] = 0
    top_idx = 0
    btm_idx = 0

    while True:
        top_idx += 1
        if t + top_idx >= h:
            top_idx = h - t - 1
            break

        if np.mean(new_texture[t + top_idx]) >= mean_scalar:
            new_texture[below_eyeball:t + top_idx, ...] = new_texture[t + top_idx + 1, ...]
            break

    out_face_mask = out_face_mask * new_texture[t + top_idx, ...]
    while True:
        btm_idx += 1
        if b - btm_idx <= 0:
            break
        if np.mean(new_texture[b - btm_idx]) >= mean_scalar:
            new_texture[b - btm_idx + 1:, ...] = new_texture[b - btm_idx, ...]
            break
    new_texture += out_face_mask
    return new_texture
