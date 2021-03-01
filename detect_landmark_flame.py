import cv2
from utils.wj_utils import get_crop_box, crop_image, get_smooth_data, resize_para
from res_flame.deca import DECA
from res_flame.utils.config import cfg
from face_alignment.detection import sfd_detector as detector
from face_alignment.detection import FAN_landmark


def get_landmark(path):
    crop_type = "bbox"  # crop type bbox/landmark
    write_file = "E:/video/demo/landmark3d_smooth.avi"

    deca = DECA(config=cfg, device=cfg.device)

    cap = cv2.VideoCapture(path)
    ret, frame = cap.read()
    w, h, _ = frame.shape
    # out = cv2.VideoWriter(write_file, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 24, (h, w))

    w_h_scale = resize_para(frame)
    face_detect = detector.SFDDetector(cfg.device, cfg.rect_model_path, w_h_scale)
    if crop_type == "landmark":
        face_landmark = FAN_landmark.FANLandmarks(cfg.device, cfg.landmark_model_path, cfg.detect_type)

    frame_list = []
    landmark_list = []
    while ret:
        bbox = face_detect.extract(frame, cfg.thresh)
        if len(bbox) > 0:
            if crop_type == "landmark":
                frame, landmarks = face_landmark.extract([frame, bbox])
                landmark = landmarks[0]
                box, ratio = get_crop_box(landmark, cfg.scale, cfg.dataset.image_size, frame.shape, crop_type)
                images = crop_image(frame, box, cfg.dataset.image_size)
            else:
                box, ratio = get_crop_box(bbox, cfg.scale, cfg.dataset.image_size, frame.shape, crop_type)
                images = crop_image(frame, box, cfg.dataset.image_size)

            code_dict = deca.encode(images)
            landmark3d = deca.get_landmark3d(code_dict, box, ratio)
            show_frame, landmark_3d = get_smooth_data(frame, landmark3d, frame_list, landmark_list, cfg.list_size)
            for x, y in landmark_3d[:, :2]:
                cv2.circle(frame_list[cfg.list_size // 2], (int(x), int(y)), 1, (255, 255, 0))
            cv2.imshow("frame", frame_list[cfg.list_size // 2])
            # out.write(frame_list[list_size // 2])
            cv2.waitKey(1)
        else:
            print("cannot find face......")
        ret, frame = cap.read()


if __name__ == "__main__":
    video_path = "video/base_test.mp4"
    get_landmark(video_path)
