import cv2
import numpy as np
import torchvision.transforms as T
from PIL import Image, ImageDraw
from ultralytics import YOLO

class FrameAnnotator():
    def __init__(self, model_path):
        self.frames = []
        self.annotated_frames = []

        self.model = YOLO(model_path)
        
        self.transform = T.Compose([
            T.GaussianBlur(3),
            T.RandomVerticalFlip(1)
        ])
       
        self.frame_rate = None
        
    def video_to_frames(self, filepath):
        """
        Converts a video to individual frames and saves it to self.frames
        filepath (String): the path to the given file
        """
        # takes in the video form a given filepath
        video = cv2.VideoCapture(filepath) 
        self.frames = []
        more_frames, img =  video.read()
        self.frame_rate = video.get(cv2.CAP_PROP_FPS)
        count = 0
        while more_frames and count <= 3:
            count += 1
            self.frames.append(img)
            more_frames, img = video.read()
 
        video.release()
        # cv2.imshow('frame', self.frames[19])
        # cv2.waitKey(0)

    def annotate_frames(self):
        """
        Annotates the frames with the loaded model and saves them to self.annotated_frames
        """
        if len(self.frames) == 0:
            raise Exception("No frames loaded")
        self.annotated_frames = []
        for frame in self.frames:
            # self.annotated_frames.append(self.model.predict(frame))
            output = self.model.predict(conf=0.25, source=frame)
            predictions = output[0].boxes.xyxy
            for pred in predictions:
                x1 = pred[0]
                y1 = pred[1]
                x2 = pred[2]
                y2 = pred[3]
                annotated_frame = Image.fromarray(frame)
                draw = ImageDraw.Draw(annotated_frame)
                draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
                self.annotated_frames.append(np.asarray(self.transform(annotated_frame)))

    def reconstruct_video(self, filename):
        if len(self.annotated_frames) == 0:
            raise Exception("No frames annotated")
        size = (self.frames[0].shape[1], self.frames[0].shape[0])
        writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*"mp4v"), 
                                 self.frame_rate, size)

        for frame in self.annotated_frames:
            writer.write(frame)
        
        writer.release()

    def generate_csv(self):
        if len(self.frames) == 0:
            raise Exception("No frames loaded")


fa = FrameAnnotator('yolov10_starfish_model.pt')

fa.video_to_frames("seafloor_footage.mp4")
fa.annotate_frames()
fa.reconstruct_video("flipped_seafloor.mp4")