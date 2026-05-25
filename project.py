import matplotlib.pyplot as plt
import torch
import cv2
import ultralytics
from ultralytics import YOLO
from torch.signal import windows

def mutli_object_tracking(video_path):

    # call yolo26 model
    model = YOLO("yolo26n.pt")
    output_path='output.mp4'
    #use openCV to encode video into discrete frames
    cap=cv2.VideoCapture(video_path)

    #check for videoload error
    if not cap.isOpened():
      raise FileNotFoundError(f"could not open the video file:{video_path}")

    # # get video properties to be used for reconstruction
    fps=int(cap.get(cv2.CAP_PROP_FPS))
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    #create video reconstruct objects
    fourcc=cv2.VideoWriter_fourcc(*'mp4v')
    out=cv2.VideoWriter(output_path,fourcc,fps,(width,height))

    print("Processing video frames with ByteTrack... please wait.")

    # actual processing
    while cap.isOpened():
      sucess,frame=cap.read()
      if sucess:
        result=model.track(frame,tracker="bytetrack.yaml",persist=True)

        ready_frame=result[0].plot()

        out.write(ready_frame)
      else:
        break

    # clear resources
    out.release()
    cap.release()
    return output_path
