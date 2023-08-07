# Table of Contents

- [Description](#description)
- [Features](#features)
- [Installation for normal execution](#installation-for-normal-execution)
- [Installation for jupyter lab execution](#installation-for-jupyter-lab-execution)
- [Usage](#usage)
- [Results](#results)
- [Contact](#contact)

# Social Distancing using Computer Vision and Deep Learning

This project utilizes the OpenVINO (Open Visual Inference and Neural Network Optimization) toolkit to implement social distancing enforcement using computer vision and deep learning techniques.

## Description

In response to the COVID-19 pandemic, social distancing has become a crucial measure to prevent the spread of the virus. This project leverages the power of OpenVINO, an open-source toolkit provided by Intel, to enhance the efficiency and performance of the computer vision and deep learning models used for social distancing enforcement.

By analyzing live video feeds from surveillance cameras or recorded videos, the system detects and tracks individuals in the scene. It then measures the distances between people and identifies whether they are maintaining a safe distance or not.

## Features

- Real-time detection and tracking of individuals in a video stream.
- Calculation of the distance between people in the scene.
- Automated identification of violations where individuals are not maintaining a safe distance.
- Utilization of OpenVINO for accelerated inference and optimization on Intel hardware.
- NNCF optimized model

## Installation for normal execution
### Creating Virtual is Recommended (not complusory)

1. Clone the project repository from GitHub using git cli (*if git command is showing error you have to download git from* [git_download_website](https://git-scm.com/downloads))
    ```
    git clone https://github.com/Praveen880890/Social-Distancing-Using-Deep-Learning.git
    
    ```

2. Now change the Directory to *Main_file/Other_approach_1*.
    
    ```
    cd Main_file/Other_approach_1
    
    ```

3. Install the packages required for this project.

    ```
    pip install -r docs\requirements.txt

    ```

4. Now once again change the directory to *"code"*.

    ```
    cd code

    ```

5. Run the python file and observe result.
    It is suggested to let the video run untill the *source ended*

    ```
    python -m demo.py
    
    ```

6. The result is saved in the **/demo_videos** folder.

## Installation for jupyter lab execution
### Creating Virtual is Recommended (not complusory)

1. Clone the project repository from GitHub using git cli (*if git command is showing error you have to download git from* [git_download_website](https://git-scm.com/downloads))
    ```
    git clone https://github.com/Praveen880890/Social-Distancing-Using-Deep-Learning.git
    
    ```

2. Now change the Directory to *Main_file/Other_approach_1*.
    
    ```
    cd Main_file/Other_approach_1
    
    ```

3. Install the packages required for this project.

    ```
    pip install -r docs\requirements.txt
    pip install jupyterlab

    ```

4. Launch the jupyter lab in the current directory

    ```
    jupyter lab

    or

    python -m jupyterlab

    ```

5. After the lab is launched navigate to code folder and open it.

6. ***Optional*** 
    
    Run all cells of convert.ipynb
    
    To Download the model and datasets
    
    It further optimizes the model using NNCF Post-training Quantization API, first we created a dataset and then we run the nncf.quantize() for getting an optimized model. After that we Serialize OpenVINO IR model using the openvino.runtime.serialize().

    **The model which is in /models folder is already optimized and ready to use so only do this step for creating own model by changing parameters related to Quanization.**

7. Run all the cells in demo.ipynb file 
    - You can customize the video file to your desire and observe results
      
      It is suggested to let the video run untill the *source ended*
   
    By changing
    
    *Start the video feed and run the object detection
    
    run_object_detection(source="../data/pedestrians.mp4", flip=False, use_popup=True,skip_first_frames=0, model=yolov8n_with_preprocess_model, device="AUTO")*
    
    Here change the source to the path where you saved the file or you can save it the **/data** folder 
    
    Run the above line by changing *source*="../data/Your_filename"
## Usage

--> Every parameter is fixed accordingly for acheiving greater accuracy and Fps.

--> If you want to change some parameters, you can try out different parameter values and make sure that not changing much code under the functions.

## Results



https://github.com/Praveen880890/Social-Distancing-Using-Deep-Learning/assets/76040957/b94f4711-302d-4386-99bd-33c16f2020f4



## Contact 
    - Email us --> s.praveenchowdarysureddy@gmail.com
               --> reddysaikumar931@gmail.com
               --> shivasaisandela002@gmail.com
