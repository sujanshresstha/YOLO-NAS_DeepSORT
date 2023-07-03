# Object Tracking using YOLO-NAS and DeepSort

This repository contains code for object tracking in videos using the YOLO-NAS object detection model and the DeepSORT algorithm. The code processes each frame of a video, performs object detection using YOLO-NAS, and tracks the detected objects across frames using DeepSort. It also draws bounding boxes and labels on the objects and calculates the frames per second (FPS) of the processed video.

## Demo of Object Tracker
<p align="center"><img src="data/helpers/demo.gif"\></p>

## Prerequisites
- Python 3.10
- OpenCV
- PyTorch
- deep-sort-realtime
- super-gradients

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/sujanshresstha/YOLO-NAS_DeepSORT.git
   cd YOLO-NAS_DeepSORT
   ```
   
2. Create new environment using conda
   ```
   conda env create -f conda.yml
   conda activate yolonas-deepsort
   ```


## Usage
1. Prepare the video file:
   - Place the video file in the desired location.
   - Update the `video` flag in the code to the path of the video file or set it to `0` to use the webcam as the input.
2. Configure the YOLO-NAS model:
   - Update the `model` flag in the code to select the YOLO-NAS model variant (`yolo_nas_l`, `yolo_nas_m`, or `yolo_nas_s`).
   - Make sure the corresponding model weights are available.
3. Configure the output video:
   - Update the `output` flag in the code to specify the path and filename of the output video file.
4. Set the confidence threshold:
   - Adjust the `conf` flag in the code to set the confidence threshold for object detection. Objects with confidence below this threshold will be filtered out.

5. Run the code:
   ```
   # Run object tracking using YOLO-NAS and DeepSort on a video
   python object_tracking.py --video ./data/video/test.mp4 --output ./output/output.mp4 --model yolo_nas_l

   # Run object tracking using YOLO-NAS and DeepSort on webcam (set video flag to 0)
   python object_tracking.py --video 0 --output ./output/webcam.mp4 --model yolo_nas_l
   ```
   
6. During runtime:
   - The processed video frames will be displayed in a window.
   - The frames per second (FPS) will be printed in the console.
   - Press the 'q' key to stop the processing and close the window.
7. After completion:
   - The output video file will be saved in the specified location.

## Model and Data
- The YOLO-NAS model used in this code is pre-trained on the COCO dataset.
- The COCO class labels are loaded from a file named `coco.names` located in the specified path.
- The DeepSort algorithm is used for tracking the detected objects. It associates unique track IDs with each object and maintains their states across frames.

## License
This project is licensed under the [MIT License](LICENSE).

## Acknowledgements
- This code is built upon the YOLO-NAS model and the DeepSort algorithm.
- Credits to the authors and contributors of the respective repositories used in this project.

## References
- [YOLO-NAS: Towards Real-time Multi-object Detection with Neural Architecture Search](https://github.com/Deci-AI/super-gradients/blob/master/YOLONAS.md)
- [Simple Online and Realtime Tracking with a Deep Association Metric](https://arxiv.org/abs/1703.07402)
