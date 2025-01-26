# ExercisePoseCorrection
This project implements a real-time fitness tracking and posture correction system using computer vision and deep learning techniques. It focuses on detecting and classifying three key exercises—push-ups, squats, and bicep curls—while providing real-time feedback on exercise form.

  
![Image](https://github.com/user-attachments/assets/e62fb81f-44cf-483b-812c-6f014d3dce2a)


![image](https://github.com/user-attachments/assets/f5409f5d-e545-46f6-87c4-c31eb4b002c5)

  •Multi-Input Support:
        •Webcam Integration
        •DroidCam USB for smartphone usage
        •Pre-recorded video uploads
  •User-Friendly Deployment: Built using Streamlit for an interactive and intuitive interface.

🛠️ Methodology
1. Data Collection
    •Videos of exercises recorded at 60 FPS.
    •Frames extracted and annotated using Roboflow.
    •Data augmentation for diversity (brightness, rotation, flipping).

![image](https://github.com/user-attachments/assets/d0c44cb2-5ad9-462f-9224-b89a302045c7)

   •Dataset split into:
        •Training Set (70%)
        •Validation Set (20%)
        •Test Set (10%)

![image](https://github.com/user-attachments/assets/24efeb91-c1c0-4350-84ea-dc2c6da11886)


2. Model Training
    •Trained using YOLOv8 architecture for robust and fast exercise detection.
    •Optimized on GPU for efficiency.
    •Metrics:
        •Precision: 99%
        •Recall: 89%
        •mAP50-95: 92.7%

3. Pose Estimation & Form Analysis
    •MediaPipe Pose identifies body landmarks.
    •Specific posture criteria ensure correct exercise execution:
        •Push-Ups: Detect up/down phases and evaluate back alignment.
        •Squats: Analyze knee alignment, shoulder-to-knee posture, and back angle.
        •Bicep Curls: Assess elbow movement and shoulder stability.

4. Deployment
    •Streamlit app allows real-time feedback and video analysis.
    •Three modes of input:
        •Webcam for live feedback.
        •DroidCam USB to use a smartphone as a webcam.
        •Video Uploads for offline analysis.

📊 Results

   •High accuracy in detecting and classifying exercises.
   •Comprehensive feedback on form:
        •Real-time corrections using skeletal landmarks.
        •Detailed metrics like joint angles and motion tracking.
   •Loss curves and precision-recall graphs demonstrate strong model performance.

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

    streamlit run app.py

Input Options

    Webcam: Use a PC or external webcam.
    DroidCam USB: Install DroidCam app on your smartphone and connect via USB.
    Video Uploads: Upload recorded exercise videos for analysis.


🧑‍💻 Contributors
  Mohammad Ali Haider 
  Syed Afraz 
  Mufti Muqaram Majid Farooqi 
