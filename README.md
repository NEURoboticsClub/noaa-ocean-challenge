# noaa-ocean-challenge
NUWave Repo for the NOAA Ocean Exploration Challenge
[Link to Challenge Explanation](https://20693798.fs1.hubspotusercontent-na1.net/hubfs/20693798/2024%20OER%20MATE%20ROV%20Computer%20Coding%20Challenge.docx.pdf)
You need to download the video and put it in the folder you're working in after git cloning. The ```seafloor_footage.mp4``` file is in the ```.gitignore``` because it is too large to upload to github.

Models to try
1. FasterRCNN
2. YoloNet v whatever is the highest
3. YoloNas
4. (Reach) transformers


## Clone this repo:
Open Terminal (Mac/Linux/WSL) or Git Bash (Windows) and type this:
>```
> git clone git@github.com:NEURoboticsClub/noaa-ocean-challenge.git
>```

## Install dependencies:
Open terminal and navigate to the repo you just cloned (`cd ~/noaa-ocean-challenge`).If you don't want to use a virtual environment, run `pip install -r requirements.txt` to install all needed dependencies.

## Running the GUI
Once in the repo, in the terminal run the following command:
>```
> python3 application.py
>```
