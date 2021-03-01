#!/usr/bin/env python3
# coding: utf-8
import torch
import torchvision.transforms as transforms
from res_flame.models import mobilenet_v1
import numpy as np
import cv2
from utils.ddfa import ToTensorGjz, NormalizeGjz
from utils.inference import (
    crop_img,
    predict_68pts,
)
import socket
import torch.nn as nn
from collections import deque
from utils.wj_utils import resize_para, roi_box_from_landmark, resize_frame
from face_alignment.detection import sfd_detector as detector
from face_alignment.detection import FAN_landmark
import torch.backends.cudnn as cudnn
import torchvision.models as models


def main():
    rect_model_path = "model/s3fd.pth"
    landmark_model_path = "model/2DFAN4-11f355bf06.pth.tar"
    checkpoint_fp = 'model/vdc_mobilenet1.tar'
    model_path = 'model/vdc_resnet_5.901.pth.tar'
    cap = cv2.VideoCapture('video/new1.mp4')

    use_cuda = True
    resize_size = 1280
    ret, frame = cap.read()
    frame = resize_frame(frame, resize_size)
    height, width, _ = frame.shape
    out = cv2.VideoWriter("E:/video/base_original.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (width, height))
    detect_type = "2D"
    w_h_scale = resize_para(frame)
    device = torch.device("cuda")
    face_detect = detector.SFDDetector(device, rect_model_path, w_h_scale)
    face_landmark = FAN_landmark.FANLandmarks(device, landmark_model_path, detect_type)
    thresh = 0.5
    arch = 'mobilenet_1'
    transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])

    if arch == 'mobilenet_1':
        checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
        model = getattr(mobilenet_v1, arch)(num_classes=240)
        model_dict = model.state_dict()
        for k in checkpoint.keys():
            model_dict[k.replace('module.', '')] = checkpoint[k]
        model.load_state_dict(model_dict)
    else:
        model = models.resnet34()
        model.fc = nn.Linear(512, 240)
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)['state_dict']
        model_dict = model.state_dict()
        for k in checkpoint.keys():
            model_dict[k.replace('module.', '')] = checkpoint[k]
        model.load_state_dict(model_dict)
    if use_cuda:
        cudnn.benchmark = True
        model = model.cuda()
    model.eval()
    frame_idx = 0
    vertex_queue = deque()
    frame_queue = deque()
    base_vertex_queue = deque()
    socket_flag = False
    if socket_flag:
        bone_idx = [17, 19, 21, 22, 24, 26, 30, 31, 35,
                    36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                    46, 47, 48, 49, 51, 53, 54, 55, 57, 59]
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((socket.gethostname(), 5066))
        s.listen(5)
        c, addr = s.accept()

    while ret:
        frame = resize_frame(frame, resize_size)

        if frame_idx == 0:
            bbox = face_detect.extract(frame, thresh)
            if len(bbox) > 0:
                frame, landmarks = face_landmark.extract([frame, bbox])
                landmark = landmarks[0]
                roi = roi_box_from_landmark(landmark)
                img = crop_img(frame, roi)
                if arch == 'mobilenet_1':
                    img = cv2.resize(img, dsize=(120, 120), interpolation=cv2.INTER_LINEAR)
                else:
                    img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_LINEAR)
                input_data = transform(img).unsqueeze(0)
                with torch.no_grad():
                    if use_cuda:
                        input_data = input_data.cuda()
                    param = model(input_data)
                    param = param.squeeze().cpu().numpy().flatten().astype(np.float32)
                if socket_flag:
                    base_param = param.copy()
                    base_param[211:] = [0.95447886, -0.728335, 0.1659534, -0.5542762, -0.06099899, 0.04292522,
                                        -0.06127983, 0.57398134, 0.04474127, 0.35590646, 0.02566887, -0.17629004,
                                        0.00880257, -0.02608619, 0.01712574, 0.16425654, 0.00248055, 0.02372667,
                                        0.00698762, -0.14303209, 0.09441634, -0.0962602, -0.02019671, 0.02978132,
                                        -0.05583464, 0.0188359, -0.06895331, -0.00185109, 0.03680567]
                    base_pts = predict_68pts(base_param, roi, img.shape[0])
                    base_vertex_queue.append(base_pts.T.copy())
                    base_vertex_queue.append(base_pts.T.copy())
                pts68 = predict_68pts(param, roi, img.shape[0])

                vertex_queue.append(pts68.T.copy())
                vertex_queue.append(pts68.T.copy())
                frame_queue.append(frame.copy())
                frame_idx += 1
            else:
                print("cannot find face")
                continue
        else:
            roi = roi_box_from_landmark(pre_landmark)
            if roi[0] == roi[2]:
                bbox = face_detect.extract(frame, thresh)
                if len(bbox) > 0:
                    frame, landmarks = face_landmark.extract([frame, bbox])
                    landmark = landmarks[0]
                    roi = roi_box_from_landmark(landmark)
            img = crop_img(frame, roi)
            if arch == 'mobilenet_1':
                img = cv2.resize(img, dsize=(120, 120), interpolation=cv2.INTER_LINEAR)
            else:
                if not img.shape[0] > 0:
                    print("error")
                img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_LINEAR)
            input_data = transform(img).unsqueeze(0)
            with torch.no_grad():
                if use_cuda:
                    input_data = input_data.cuda()
                param = model(input_data)
                param = param.squeeze().cpu().numpy().flatten().astype(np.float32)
                if socket_flag:
                    base_param = param.copy()
                    base_param[211:] = [0.95447886, -0.728335, 0.1659534, -0.5542762, -0.06099899, 0.04292522,
                                        -0.06127983, 0.57398134, 0.04474127, 0.35590646, 0.02566887, -0.17629004,
                                        0.00880257, -0.02608619, 0.01712574, 0.16425654, 0.00248055, 0.02372667,
                                        0.00698762, -0.14303209, 0.09441634, -0.0962602, -0.02019671, 0.02978132,
                                        -0.05583464, 0.0188359, -0.06895331, -0.00185109, 0.03680567]
                    base_pts = predict_68pts(base_param, roi, img.shape[0])
                    base_vertex_queue.append(base_pts.T.copy())

            pts68 = predict_68pts(param, roi, img.shape[0])
            vertex_queue.append(pts68.T.copy())
            frame_queue.append(frame.copy())
            frame_idx += 1
        if frame_idx > 0:
            pre_landmark = pts68.T[:, :2]
            if len(vertex_queue) >= 3:
                vertex_queue.popleft()
                if socket_flag:
                    base_vertex_queue.popleft()
                frame_queue.popleft()
            frame_queue.append(frame.copy())
            vertex_queue.append(pts68.T.copy())

            vertex_ave = np.mean(vertex_queue, axis=0)
            if socket_flag:
                base_vertex_queue.append(base_pts.T.copy())
                base_vertex_ave = np.mean(base_vertex_queue, axis=0)

            pre_frame = frame_queue[1]
            for x, y in vertex_ave[:, :2]:
                cv2.circle(pre_frame, (int(x), int(y)), 1, (255, 255, 0))

            if socket_flag:
                for x, y in base_vertex_ave[:, :2]:
                    cv2.circle(pre_frame, (int(x), int(y)), 1, (0, 255, 255))
                trans = (vertex_ave - base_vertex_ave)

                info = ""
                adjust_ratio = 300
                open_ratio = min(round(trans[62][1] - trans[66][1], 4), 0) / adjust_ratio / 3
                for idx in bone_idx:
                    if idx in [49, 53, 55, 59]:
                        adjust_ratio = adjust_ratio * 2
                    info += ", ".join(str(round(e / adjust_ratio, 4)) for e in trans[idx].tolist())
                    info += ", "
                info += str(open_ratio)
                recv_msg = c.recv(1024)
                print(recv_msg.decode())
                c.send(bytes(info, "utf-8"))

            vertex_queue.popleft()
            frame_queue.popleft()
            if socket_flag:
                base_vertex_queue.popleft()
        cv2.namedWindow("3ddfa", cv2.WINDOW_NORMAL)
        cv2.imshow('3ddfa', pre_frame)
        out.write(pre_frame)
        cv2.waitKey(1)
        ret, frame = cap.read()
    cap.release()
    cv2.destroyAllWindows()

    if socket_flag:
        c.close()


if __name__ == '__main__':
    main()
