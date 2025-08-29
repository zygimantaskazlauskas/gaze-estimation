import cv2
import logging
import argparse
import warnings
import numpy as np

import torch
import torch.nn.functional as F

from picamera2 import Picamera2
from torchvision import transforms

from config import data_config
from utils.helpers import get_model, draw_bbox_gaze


# Use OpenCV Haar Cascade for face detection (compatible with Raspberry Pi)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Helper function to detect faces (returns bboxes and dummy keypoints)
def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    bboxes = []
    keypoints = []
    for (x, y, w, h) in faces:
        bboxes.append([x, y, x + w, y + h])
        # Haar does not provide keypoints, so use center as dummy
        keypoints.append([[x + w // 2, y + h // 2]])
    return bboxes, keypoints

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(message)s')


def parse_args():
    parser = argparse.ArgumentParser(description="Gaze estimation inference")
    parser.add_argument("--model", type=str, default="resnet34", help="Model name, default `resnet18`")
    parser.add_argument(
        "--weight",
        type=str,
        default="resnet34.pt",
        help="Path to gaze esimation model weights"
    )
    parser.add_argument("--view", action="store_true", default=True, help="Display the inference results")
    parser.add_argument("--source", type=str, default="assets/in_video.mp4",
                        help="Path to source video file or camera index")
    parser.add_argument("--output", type=str, default="output.mp4", help="Path to save output file")
    parser.add_argument("--dataset", type=str, default="gaze360", help="Dataset name to get dataset related configs")
    args = parser.parse_args()

    # Override default values based on selected dataset
    if args.dataset in data_config:
        dataset_config = data_config[args.dataset]
        args.bins = dataset_config["bins"]
        args.binwidth = dataset_config["binwidth"]
        args.angle = dataset_config["angle"]
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}. Available options: {list(data_config.keys())}")

    return args


def pre_process(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = transform(image)
    image_batch = image.unsqueeze(0)
    return image_batch


def main(params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    idx_tensor = torch.arange(params.bins, device=device, dtype=torch.float32)

    # Use Haar Cascade face detector

    try:
        gaze_detector = get_model(params.model, params.bins, inference_mode=True)
        state_dict = torch.load(params.weight, map_location=device)
        gaze_detector.load_state_dict(state_dict)
        logging.info("Gaze Estimation model weights loaded.")
    except Exception as e:
        logging.info(f"Exception occured while loading pre-trained weights of gaze estimation model. Exception: {e}")

    gaze_detector.to(device)
    gaze_detector.eval()


    # Use Picamera2 for Raspberry Pi Camera Module 3 (new camera stack)
    picam2 = Picamera2()
    camera_config = picam2.create_preview_configuration()
    picam2.configure(camera_config)
    picam2.start()
    # Get camera resolution for output video
    width = camera_config["main"]["size"][0]
    height = camera_config["main"]["size"][1]
    if params.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(params.output, fourcc, 30, (width, height))

    with torch.no_grad():
        while True:
            frame = picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if frame is None or frame.size == 0:
                logging.info("Failed to obtain frame or EOF")
                break

            bboxes, keypoints = detect_faces(frame)
            for bbox, keypoint in zip(bboxes, keypoints):
                x_min, y_min, x_max, y_max = map(int, bbox[:4])
                image = frame[y_min:y_max, x_min:x_max]
                if image.size == 0:
                    continue
                image = pre_process(image)
                image = image.to(device)

                # pitch, yaw = torch.zeros((1, 90)), torch.zeros((1, 90)) #gaze_detector(image)
                pitch, yaw = gaze_detector(image)


                pitch_predicted, yaw_predicted = F.softmax(pitch, dim=1), F.softmax(yaw, dim=1)

                # Mapping from binned (0 to 90) to angles (-180 to 180) or (0 to 28) to angles (-42, 42)
                pitch_predicted = torch.sum(pitch_predicted * idx_tensor, dim=1) * params.binwidth - params.angle
                yaw_predicted = torch.sum(yaw_predicted * idx_tensor, dim=1) * params.binwidth - params.angle

                # Degrees to Radians
                pitch_predicted = np.radians(pitch_predicted.cpu())
                yaw_predicted = np.radians(yaw_predicted.cpu())

                # draw box and gaze direction
                draw_bbox_gaze(frame, bbox, pitch_predicted, yaw_predicted)

            if params.output:
                out.write(frame)

            if params.view:
                cv2.imshow('Demo', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    picam2.close()
    if params.output:
        out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()

    if not args.view and not args.output:
        raise Exception("At least one of --view or --ouput must be provided.")

    main(args)
