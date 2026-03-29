# Sign Language To Text and Speech Conversion

A real-time American Sign Language (ASL) to text and speech converter built with CNN, MediaPipe, OpenCV, and Tkinter.

## About

This project captures hand gestures from a webcam, recognizes ASL alphabet signs, converts them into text, and speaks the detected output aloud. It combines MediaPipe-based hand landmark extraction with a CNN model trained on hand skeleton images for robust recognition.

## Features

- Real-time hand gesture detection using a webcam
- Recognition of ASL alphabet gestures
- Text output for detected characters and words
- Text-to-speech conversion using `pyttsx3`
- Word suggestion support for better sentence completion
- Landmark-based preprocessing for improved performance across different backgrounds
- Reported accuracy of 97%, reaching 99% in clean background and good lighting conditions

## Tech Stack

- Python 3.11
- TensorFlow / Keras
- MediaPipe
- OpenCV
- cvzone
- pyttsx3
- Tkinter
- Pillow
- pyspellchecker

## How It Works

1. The webcam captures live video frames.
2. MediaPipe detects hand landmarks from the visible hand.
3. The landmarks are drawn on a plain white canvas.
4. The CNN model classifies the gesture into a gesture group and then into a final letter.
5. The recognized output is shown on screen and can be spoken aloud.

## Gesture Groups

The CNN is trained on grouped classes before the final rule-based letter refinement step.

| Class | Letters |
| --- | --- |
| 0 | A, E, M, N, S, T |
| 1 | B, D, F, I, K, R, U, V, W |
| 2 | C, O |
| 3 | G, H |
| 4 | L |
| 5 | P, Q, Z |
| 6 | X |
| 7 | Y, J |

## Project Files

- `final_pred.py` - main GUI application for live prediction
- `prediction_wo_gui.py` - prediction pipeline without GUI
- `data_collection_binary.py` - binary image data collection script
- `data_collection_final.py` - final data collection script
- `cnn8grps_rad1_model.h5` - trained CNN model

## Setup Instructions

### Requirements

- Python 3.11
- Webcam
- Windows is recommended for the current GUI and speech setup

### Clone the Repository

```bash
git clone https://github.com/osaid22/Sign-Language-To-Text-Speech.git
cd Sign-Language-To-Text-Speech
```

### Create and Activate a Virtual Environment

```powershell
python -m venv venv
venv\Scripts\activate
```

### Install Dependencies

```powershell
pip install opencv-python numpy keras tensorflow mediapipe==0.10.30 cvzone==1.5.6 pyttsx3 pyspellchecker Pillow
```

### Run the Application

```powershell
python final_pred.py
```

## System Flow

```text
Webcam -> Hand Detection -> Landmark Extraction -> White Background Drawing
-> CNN Classification -> Letter Prediction -> Text Output -> Speech Output
```

## Notes

- Good lighting improves recognition quality.
- A clear view of a single hand gives the best results.
- If you face MediaPipe compatibility issues, use Python 3.11 in a fresh virtual environment.

## Future Improvements

- Add word-level and sentence-level prediction
- Improve support for dynamic gestures
- Add model retraining scripts and dataset documentation
- Build a web-based or desktop-packaged version

## License

This project is for educational and academic use unless you choose to add a specific open-source license.
