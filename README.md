# Diagnosis and Translation Between Video And Audio Recordings of Heart rate: VAHR
This study's objective is to explore the feasibility of translating heart rate information between these modalities using deep learning techniques. Additionally, it also aims to evaluate the efficiency and consistency of the translated video and audio signals in predicting heart rate.

## Important Links

| [Timesheet](https://1sfu-my.sharepoint.com/:x:/g/personal/kabhishe_sfu_ca/Ee20R0_sK8NKodZikXyvJd8BpL5t7OL5Ass7mfhtdIsgWQ?e=IIMpdv) | [Slack channel](https://sfucmpt340fall2023.slack.com/archives/C05TBCRL7GV) | [Project report](https://www.overleaf.com/project/655aafa73f9551bd78d95b11) |
|-----------|---------------|-------------------------|

## Video/demo/GIF
Record a short video (1:40 - 2 minutes maximum) or gif or a simple screen recording or even using PowerPoint with audio or with text, showcasing your work.


## Table of Contents
1. [Demo](#demo)

2. [Installation](#installation)

3. [Reproducing this project](#repro)

4. [Guidance](#guide)


<a name="demo"></a>
## 1. Example demo

[](https://github.com/sfu-cmpt340/project_07/blob/main/minimalDemo.gif)

### What to find where

Here is where you can find important files

```bash
repository
├── src                                                 ## Source code of the package itself 
    ├── amazing                                         ## Utility functions and code to train models
        ├── functions.py                                ## Utility functions for GUI, processing data, and function to get HR from video
        ├── train_predict_heart_rate_from_feature.py    ## Code to train model to predict heart rate from features
        ├── train_predict_heart_rate_from_video.py      ## Code to train model to predict heart rate from video
    ├── pretrained_models                               ## Pre-trained models for use in run.y
    ├── sample_training_data                            ## Sample dataset for training the models
    ├── run.py                                          ## Main run file to start up the GUI
├── sample_videos_to_predict                            ## Sample videos that can be used for prediction in the GUI
├── README.md                                           ## You are here
├── requirements.txt                                    ## Python modules required
```

<a name="installation"></a>

## 2. Installation
Python 3.11

```bash
git clone https://github.com/sfu-cmpt340/project_07
cd project_07
pip install -r .\requirements.txt
python src/run.py
```

<a name="repro"></a>
## 3. Reproduction
To recreate/train the models:

1. Download [TrainingData.zip](https://drive.google.com/file/d/1K85C8IYuDsKvYJrUjjeVZJ5weZY23l2a/view?usp=sharing)
2. Extract TrainingData.zip into project_07/src
3. In the project_07 directory run "python src/amazing/train_predict_heart_rate_from_video.py"
4. In the project_07 directory run "python src/amazing/train_predict_heart_rate_from_feature.py"
5. Close out any produced figures to continue the training

To use the GUI:

1. In the project_07 directory run "python src/run.py" to start up the GUI
2. Choose a desired prediction method
3. Upload a sample video file from the "sample_videos_to_predict" folder or fill out the feature form
4. Click "Predict My Current Heart Rate Range!" to get a prediction 
5. The video prediction result is displayed 

(Note: When using option "Predict Heart rate from a video (without DL), make sure to close the figure to see the result)

<a name="guide"></a>
## 4. Guidance

- Use [git](https://git-scm.com/book/en/v2)
    - Do NOT use history re-editing (rebase)
    - Commit messages should be informative:
        - No: 'this should fix it', 'bump' commit messages
        - Yes: 'Resolve invalid API call in updating X'
    - Do NOT include IDE folders (.idea), or hidden files. Update your .gitignore where needed.
    - Do NOT use the repository to upload data
- Use [VSCode](https://code.visualstudio.com/) or a similarly powerful IDE
- Use [Copilot for free](https://dev.to/twizelissa/how-to-enable-github-copilot-for-free-as-student-4kal)
- Sign up for [GitHub Education](https://education.github.com/) 
