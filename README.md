# Car Drowsiness Detector

![Car Drowsiness Detector](images/banner.png)

The Car Drowsiness Detector is an innovative solution that helps prevent accidents caused by drowsy driving. By using computer vision and deep learning techniques, the system monitors the driver's eyes in real-time and alerts them if they start to close or show signs of drowsiness.

## Features

- Real-time detection of driver's eye status
- Alert system when driver's eyes are closed or drowsiness is detected
- Works with a standard webcam
- Easy to use and integrate into existing car systems
- Can be deployed to systems with low 

## How It Works

1. The system uses a Haar cascade classifier to detect the face and eyes of the driver in the captured video frames.
2. The captured eye region is preprocessed and fed into a pre-trained CNN model.
3. The model predicts the eye status (open or closed) based on the input image.
4. If the eyes are closed or drowsiness is detected for a certain duration, an alarm is triggered to alert the driver.

## Getting Started

### Prerequisites

- Python 3.x
- OpenCV
- Keras
- Pygame

### Installation

1. Clone the repository:
git clone https://github.com/muneebelahimalik/Car-Drowsiness-Detector.git

2. Install the required dependencies:
pip install -r requirements.txt

### Usage

1. Run the main script:
python main.py

2. The system will open a video feed from the webcam and start monitoring the driver's eye status in real-time.
3. If drowsiness is detected, an alarm will be triggered to alert the driver.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any bug fixes, enhancements, or new features you would like to contribute.

## Acknowledgements

- Special thanks to the creators of the Haar cascade classifiers and the Keras library.
- Credits to the authors of the dataset used for training the CNN model.

## Contact

For any questions or inquiries, please contact: 

**Muneeb Elahi Malik**
Email: muneebellahi2001@gmail.com
LinkedIn: [Your LinkedIn Profile](https://www.linkedin.com/in/muneeb-elahi-malik)
