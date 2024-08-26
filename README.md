# noaa-ocean-challenge
NUWave Repo for the NOAA Ocean Exploration Challenge
[Link to Challenge Explanation](https://20693798.fs1.hubspotusercontent-na1.net/hubfs/20693798/2024%20OER%20MATE%20ROV%20Computer%20Coding%20Challenge.docx.pdf)
To download the video and put it in the folder you're working in after git cloning. The ```seafloor_footage.mp4``` file is in the ```.gitignore``` because it is too large to upload to github.

Credit to the roboflow YOLOv8 Object Detection model developed by Ultralytics
https://colab.research.google.com/github/roboflow/notebooks/blob/main/notebooks/train-yolov8-object-detection-on-custom-dataset.ipynb#scrollTo=HI4nADCCj3F5 

## Clone this repo:
Open Terminal (Mac/Linux/WSL) or Git Bash (Windows) and type this:
>```
> git clone git@github.com:NEURoboticsClub/noaa-ocean-challenge.git
>```

## Install dependencies:
Open terminal and navigate to the repo you just cloned:
>```
> cd ~/noaa-ocean-challenge
>```
Then to install all needed dependencies run:
>```
> pip install -r requirements.txt
>```

## Running the GUI
Once in the repo, in the terminal run the following command:
>```
> python3 application.py
>```
