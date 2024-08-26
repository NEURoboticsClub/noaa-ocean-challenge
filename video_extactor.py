import cv2
import numpy as np
import torchvision.transforms as T
from PIL import Image, ImageDraw
from ultralytics import YOLO


class FrameAnnotator():
    """
    Class to handle the backend of annotating images. 
    Has hooks to allow for direct command line use.
    model_path (String): the path to the name of the model being used
    """
    def __init__(self, model_path):
        self.frames = []
        self.annotated_frames = []
        self.model = YOLO(model_path)
        self.frame_rate = None

    def video_to_frames(self, filepath):
        """
        Converts a video to individual frames and saves it to self.frames
        filepath (String): the path to the given file
        """
        # takes in the video form a given filepath
        video = cv2.VideoCapture(filepath)
        self.frames = []
        more_frames, img = video.read()
        self.frame_rate = video.get(cv2.CAP_PROP_FPS)
        while more_frames:
            self.frames.append(img)
            more_frames, img = video.read()

        video.release()
        return self.frames

    def annotate_frames(self):
        """
        Annotates the frames with the loaded model and saves them to self.annotated_frames
        """
        if len(self.frames) == 0:
            raise Exception("No frames loaded")
        self.annotated_frames = []
        for frame in self.frames:
            self.annotated_frames.append(self.annotate_single_frame(frame)[0])

    def annotate_single_frame(self, frame):
        """
        Predicts with the model all boundary boxes for a given frame.
        Returns a tuple of the image and the boundary boxes.
        frame (np.ndarray): the frame to operate on.
        """
        # classes 0 is only sea stars
        output = self.model.predict(conf=0.25, source=frame, classes=0)
        predictions = output[0].boxes.xyxy
        annotated_frame = Image.fromarray(frame)
        for pred in predictions:
            x1 = pred[0]
            y1 = pred[1]
            x2 = pred[2]
            y2 = pred[3]
            draw = ImageDraw.Draw(annotated_frame)
            draw.rectangle([x1, y1, x2, y2], outline='red', width=3)

        return (annotated_frame, predictions)

    def reconstruct_video(self, filename, annotated_frames):
        """
        Reconstructs the video from frames and saves it.
        filename (String): the path for the file
        annotated_frames (Optional[List[PIL.Image]]): the frames to save to a video
        """
        if annotated_frames is None:
            annotated_frames = self.annotated_frames
        if len(annotated_frames) == 0:
            raise Exception("No frames annotated")
        size = (self.frames[0].shape[1], self.frames[0].shape[0])
        # the correct fourcc needs to be selected. We are currently set on mp4 files.
        writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*"mp4v"),
                                 self.frame_rate, size)

        for frame in annotated_frames:
            writer.write(np.asarray(frame))

        writer.release()

"""
To uncomment for direct command line use.
"""
# fa = FrameAnnotator('yolov10_starfish_model.pt')

# fa.video_to_frames("seafloor_footage.mp4")
# fa.annotate_frames()
# fa.reconstruct_video("flipped_seafloor.mp4")
