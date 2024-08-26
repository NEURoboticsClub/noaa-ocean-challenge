import PySimpleGUI as sg
from video_extactor import FrameAnnotator
from PIL import ImageTk, Image
import threading
import time
import numpy as np
import cv2
import pandas as pd


class GUIApp:
    def __init__(self):
        menu_def = [['&File', ['&Open', '&Save', '---', 'Properties', 'E&xit']],
                    ['&Edit', ['Paste', ['Special', 'Normal', ], 'Undo'], ],
                    ['&Help', '&About...']]
        font = ("Avenir", 14)
        self.model_dict = {'YOLO Base': FrameAnnotator('yolo_v8_model_base.pt'),
                           'YOLO Multi-Class': FrameAnnotator('yolo_v8_2class_model.pt'),
                           'YOLO Resilient': FrameAnnotator('yolo_v8_resilient_2class.pt')}
        self.image_size = (720, 480)
        # layout setup
        self.layout = [[sg.Menu(menu_def)],
                       [sg.Text('Select input video', font=font)],
                       [sg.Input(key="input_path", font=font),
                        sg.Button("Browse Input Files",
                                  key="browse_input", font=font),
                        sg.Combo(['YOLO Base', 'YOLO Multi-Class', 'YOLO Resilient'], key="selector",
                                 default_value='YOLO Base', font=font),
                        sg.Button('Start Annotating', font=font)],
                       [sg.Image(filename="NUWave_Logo_Resize.png", size=self.image_size,  key="canvas"),
                        sg.ProgressBar(max_value=600, key='progress_bar', orientation='v', size=(53.3, 20))],
                       [sg.Slider(size=(30, 20), range=(0, 100), resolution=1, key="slider", orientation="h",
                                  enable_events=True), sg.T("0", key="counter", size=(10, 1), font=font)],
                       [sg.Button('Next Frame', font=font), sg.Button("Pause/Play", key="Play", font=font),
                        sg.Button('Exit', font=font), sg.Text("Info will show up here", key='info', font=font)],
                       [sg.Button("Browse Ouput Folders", key="browse_output", font=font),
                        sg.Input(key="output_path", font=font)],
                       [sg.Text('Output Filename:         ', font=font),
                        sg.Input(key="output_file", font=font),
                        sg.Button("Save Annotated Video",
                                  key="save_video", font=font),
                        sg.Button("Save XSLX sheet", key="save_sheet", font=font)]]

        self.window = sg.Window('NUWave Sea Star Annotator', self.layout)
        self.breakout = False
        self.fa = self.model_dict['YOLO Base']
        self.frames = []
        self.annotated_frames = []
        self.prediction_boxes = np.empty((0, 4))
        self.pred_lengths = []
        self.touching_slider = False
        self.file_browsing = True

    def update(self):
        event, values = self.window.Read()
        print(event)
        annotation_thr = threading.Thread(
            target=self.annotate_frames_gui, args=(), kwargs={})
        browsing_thr = threading.Thread(target=self.browse, args=(), kwargs={})
        video_saving_thr = threading.Thread(
            target=self.save_annotated_video, args=(), kwargs={})
        sheet_saving_thr = threading.Thread(
            target=self.save_xlsx, args=(), kwargs={})
        if event is None or event == 'Exit':
            self.breakout = True
        if event == "browse_input":
            self.file_browsing = True
            if not browsing_thr.is_alive():
                browsing_thr.start()
        if event == "browse_output":
            self.file_browsing = False
            if not browsing_thr.is_alive():
                browsing_thr.start()
        if event == "Start Annotating":
            if not annotation_thr.is_alive():
                annotation_thr.start()
        if event == "slider":
            self.touching_slider = True
            if int(values["slider"]) < len(self.annotated_frames):
                self.window['canvas'].update(
                    data=self.frame_update_data(int(values["slider"])))
        if event == 'Next Frame':
            if int(values["slider"]) + 1 < len(self.annotated_frames):
                self.window['canvas'].update(
                    data=self.frame_update_data(int(values["slider"]) + 1))
            self.window['slider'].update(value=int(values["slider"]) + 1)
        if event == 'Play':
            self.touching_slider = not self.touching_slider
        if event == 'save_video':
            if not video_saving_thr.is_alive():
                video_saving_thr.start()
        if event == 'save_sheet':
            if not sheet_saving_thr.is_alive():
                sheet_saving_thr.start()

    def annotate_frames_gui(self):
        self.window['info'].update("Loading Frames")
        time.sleep(0.5)
        self.fa = self.model_dict[self.window['selector'].get()]
        try:
            self.frames = self.fa.video_to_frames(
                self.window['input_path'].get())
        except:
            self.window['info'].update("Video Path Invalid")
        finally:
            self.window['info'].update("Annotating Frames")
            self.annotated_frames = []
            self.prediction_boxes = np.empty((0, 4))
            self.window['progress_bar'].update(
                current_count=0, max=len(self.frames))
            for i in range(len(self.frames)):
                frame = self.frames[i]
                curr_frame, curr_pred = self.fa.annotate_single_frame(frame)
                self.annotated_frames.append(curr_frame)
                self.prediction_boxes = np.append(
                    self.prediction_boxes, curr_pred, axis=0)
                self.pred_lengths.append(len(curr_pred))
                self.window['progress_bar'].update(current_count=i+1)
                self.window['slider'].update(
                    range=(0, len(self.annotated_frames)))

                if not self.touching_slider:
                    self.window['canvas'].update(
                        data=self.frame_update_data(i))
                    self.window['slider'].update(value=i+1)

    def save_annotated_video(self):
        self.window['info'].update("Saving Video")
        filename = self.window['output_file'].get()
        if not filename.endswith(".mp4"):
            filename += ".mp4"
        self.fa.reconstruct_video(self.window['output_path'].get()
                                  + "/" + filename,
                                  self.annotated_frames)
        self.window['info'].update("Saving Complete")

    def save_xlsx(self):
        np_pred = self.prediction_boxes
        name_column = np.array([])
        for i in range(len(self.pred_lengths)):
            seconds = i / self.fa.frame_rate % 60
            minutes = int(i / self.fa.frame_rate - seconds)
            padding_seconds = "0" if seconds < 10 else ""
            padding_minutes = "0" if minutes < 10 else ""
            name_column = np.append(name_column, np.array(["Frame: " + str(i) + ", Timestamp: "
                                    + padding_minutes + f"{minutes}:"
                                    + padding_seconds + f"{seconds:.3f}"]
                                    * self.pred_lengths[i]))

        # to swap from YOLONAS output to specified ouput columns
        print(name_column.shape)
        print(np_pred.shape)
        np_pred = np.append(name_column.reshape(
            (len(name_column), 1)), np_pred, axis=1)
        np_pred[:, [3, 2]] = np_pred[:, [2, 3]]
        np_pred[:, [3, 4]] = np_pred[:, [4, 3]]
        labels = np.array(["Current Frame", "X Bound, Left",
                          "X Bound, Right", "Y Bound, Upper", "Y Bound, Lower"])
        df = pd.DataFrame(np_pred, columns=labels)
        self.window['info'].update("Saving Sheet")
        filename = self.window['output_file'].get()
        if not filename.endswith(".xlsx"):
            filename += ".xlsx"
        df.to_excel(self.window['output_path'].get() +
                    "/" + filename, index=False)
        self.window['info'].update("Saving Complete")

    def frame_update_data(self, index):
        return ImageTk.PhotoImage(
            Image.fromarray(cv2.cvtColor(np.array(self.annotated_frames[index]
                                                  .resize(self.image_size, Image.NEAREST)
                                                  .convert('RGB')),
                                         cv2.COLOR_RGB2BGR)))

    def browse(self):
        if self.file_browsing:
            # Open a file dialog and get the file path
            video_path = None
            try:
                video_path = sg.filedialog.askopenfile().name
            except AttributeError:
                self.window['info'].update("No video selected")
            self.window['input_path'].update(video_path)
        else:
            # Open a directory dialog and get the output path
            folder_path = None
            try:
                folder_path = sg.filedialog.askdirectory()
            except AttributeError:
                self.window['info'].update("No folder selected")
            self.window['output_path'].update(folder_path)


if __name__ == "__main__":
    ga = GUIApp()
    while not ga.breakout:
        ga.update()
