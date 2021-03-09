#!/usr/bin/env python3
# coding: utf-8
import torch
import sys
import torchvision.transforms as transforms
from res_flame.models import mobilenet_v1
import numpy as np
import cv2
from utils.ddfa import ToTensorGjz, NormalizeGjz
from utils.inference import (
    crop_img,
    predict_68pts,
)
from res_flame.utils.config import cfg
from utils.wj_utils import roi_box_from_landmark, resize_frame, get_smooth_data
from face_alignment.detection import sfd_detector as detector
from face_alignment.detection import FAN_landmark
import torch.backends.cudnn as cudnn


def get_landmark(path):
    device = torch.device("cuda")
    cap = cv2.VideoCapture(path)

    cudnn.benchmark = True
    smooth_flag = False

    ret, frame = cap.read()
    frame = resize_frame(frame, cfg.resize_size)
    height, width, _ = frame.shape

    face_detect = detector.SFDDetector(device, cfg.rect_model_path)
    face_landmark = FAN_landmark.FANLandmarks(device, cfg.landmark_model_path, cfg.detect_type)
    transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])

    model = mobilenet_v1.mobilenet(cfg.checkpoint_fp, num_classes=240)

    landmark_list = []
    frame_list = []

    while ret:
        frame = resize_frame(frame, cfg.resize_size)
        bbox = face_detect.extract(frame, cfg.thresh)
        if len(bbox) > 0:
            frame, landmarks = face_landmark.extract([frame, bbox])
            roi = roi_box_from_landmark(landmarks[0])
            img = crop_img(frame, roi)
            img = cv2.resize(img, dsize=(cfg.STD_SIZE, cfg.STD_SIZE), interpolation=cv2.INTER_LINEAR)
            input_data = transform(img).unsqueeze(0)
            with torch.no_grad():
                param = model(input_data.cuda())
                param = param.squeeze().cpu().numpy().flatten().astype(np.float32)
            pts68 = predict_68pts(param, roi, img.shape[0])
            if smooth_flag:
                show_frame, landmark_3d = get_smooth_data(frame, pts68.T, frame_list, landmark_list, cfg.list_size)
            else:
                show_frame = frame
                landmark_3d = pts68.T
            for x, y in landmark_3d[:, :2]:
                cv2.circle(show_frame, (int(x), int(y)), 1, (255, 255, 0))
            cv2.imshow('3ddfa', show_frame)
            cv2.waitKey(1)
        else:
            print("cannot find face")
            continue

        ret, frame = cap.read()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    video_path = str(sys.argv[1])
    get_landmark(video_path)
