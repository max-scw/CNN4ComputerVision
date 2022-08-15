# CNN4ComputerVision

Benchmark for Computer Vision image classification for different well-known network structures. Examines the pretrained networks (usually on ImageNet) as well as training from scratch.

## Installation
This project relies on Python 3.9 and assumes [Python](https://www.python.org/downloads) to be installed on your system.

It is recommended to create a virtual environment before installing the required packages listed in _requirements.txt_.
``
pip install -r requirements.txt
``

## Structure
In the folder structure you may create a _Data_ folder where all images are placed in three folders _Trn_ (training), _Val_ (validation), and _Tst_ (test) for the different data sets. Within each folder, the images must be sorted into one folder for every class.

CNN4ComputerVision
|- Data
    |- Trn
        |- Class1
        |- ...
    |- Tst
        |- Class1
        |- ...
    |- Val
        |- Class1
        |- ...
|- TrainModels.py

## Usage
...

## Support
Please contact the authors directly or suggest a new GitHub issue.

## Authors
SCW

## Project status
active.

