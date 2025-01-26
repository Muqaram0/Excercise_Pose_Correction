# ExercisePoseCorrection 🔥 🏋🏻‍♂️
This project implements a real-time fitness tracking and posture correction system using computer vision and deep learning techniques. It focuses on detecting and classifying three key exercises—push-ups, squats, and bicep curls—while providing real-time feedback on exercise form.

  
![Image](https://github.com/user-attachments/assets/e62fb81f-44cf-483b-812c-6f014d3dce2a)



 

## 🛠️ Methodology
#### 1. Data Collection & Model Training

   Videos of excercises were recorded at 60fps, the frames were then extracted using a simple python script, finally annotation and data augmentation was done with the help of roboflow
    
![image](https://github.com/user-attachments/assets/d0c44cb2-5ad9-462f-9224-b89a302045c7)

Dataset was split into 70% training, 20% validation and 10% Test sets and was then trained using YOLOv8 for robust and fast excercise detection 

#### 2. Model Training Results

  We were able to achieve a Precision of 99%, Recall 89% and mAP50-95 score of 92.7%
       

#### 3. Pose Estimation & Form Analysis

  MediaPipe Pose was used to identify the body landmarks, upoun which depending on the type of excercise classified a logic was using to determine if the posture is correct or not
        •Push-Ups: Detect up/down phases and evaluate back alignment.
        •Squats: Analyze knee alignment, shoulder-to-knee posture, and back angle.
        •Bicep Curls: Assess elbow movement and shoulder stability.

#### 4. Deployment
  Streamlit was used to deploy the model with three modes of input (Webcam, DroidCam and Video Uploads) to allow real time feedback 

#### 5. Demo

[https://github.com/user-attachments/assets/6ffbccd0-181b-4bba-a95c-4d07c03a34fd](https://github.com/user-attachments/assets/e1b2ff2e-53da-4fdb-9661-9931df81af89)



🖥️ Installation and Usage
Prerequisites

    Python 3.7+
    GPU (optional but recommended for real-time performance)

Installation

    Clone the repository:
    git clone https://github.com/your-username/workout-posture-correction.git

Install dependencies:

    pip install -r requirements.txt

Run the Streamlit app:

    streamlit run excercise_pose_correction.py

Input Options

    Webcam: Use a PC or external webcam.
    DroidCam USB: Install DroidCam app on your smartphone and connect via USB.
    Video Uploads: Upload recorded exercise videos for analysis.


#### Contributors
  [Ali Haider](https://github.com/AliH17)
  [Muqaram Majid](https://github.com/Muqaram0)
  [Syed Afraz](https://github.com/RageRolling)
  
