# YOLOv11 Implementation on Dental Radiograph

This repository contains the implementation of [YOLOv11 from Ultralytics](https://docs.ultralytics.com/models/yolo11/#what-are-the-key-improvements-in-ultralytics-yolo11-compared-to-previous-versions). In this repository, research and comparison between each model variant of YOLOv11 will be conducted. This model was chosen to explore and understand one of the latest iterations of YOLO, version 11. In this version, YOLO introduces improvements in real-time object detection, accuracy, speed, and efficiency. These enhancements are achieved through changes in the backbone and overall architecture to boost efficiency.

The training process is carried out using a Dental X-Ray dataset focusing on dental diseases and oral health. The use of object detection technology in this context aims to assist dentists in performing more efficient examinations. 

#### Object Detection Classes
- `cavity`
- `Fillings`
- `Impacted Tooth`
- `Implant`

## Installation

### Step 1: Install all of the Requirement Dependencies 
Install all of the dependencies in the `requirements.txt`:

    
    pip install -r requirements.txt
    
### Step 2: Set the correct path
Set the correct path for training and testing.

1. For Training:
Open the `main.py` file and set the path for model and the dataset.
2. For Testing:
Open the `test.py` file and set the path for the pre-trained model and the path for input photo.

## Training: 
Run this command in terminal:
    
    python main.py
    
## Testing:
Run this command in terminal:
    
    python test.py
    
    
## Directories Information
- `dataset/`: Contains the dataset used for training, validation, and testing. Subdirectories include `train/`, `val/`, and `test/` directories.
- `output/`: Stores output logs or messages generated during each training iteration.
- `pre-trained`: Holds pre-trained models used as a starting point for further training on the dataset.
- `runs/`: Includes all results and artifacts generated from the training process, such as model checkpoints and metrics.

## License
This repository follows the license guidelines of the original YOLOv11 project. For more details, refer to the [YOLOv11 License](https://docs.ultralytics.com/models/yolo11/#what-are-the-key-improvements-in-ultralytics-yolo11-compared-to-previous-versions)
    