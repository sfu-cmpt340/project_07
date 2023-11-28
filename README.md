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

A minimal example to showcase your work

```python
from amazing import amazingexample
imgs = amazingexample.demo()
for img in imgs:
    view(img)
```

### What to find where

Explain briefly what files are found where

```bash
repository
├── src                          ## source code of the package itself
├── scripts                      ## scripts, if needed
├── docs                         ## If needed, documentation   
├── README.md                    ## You are here
├── requirements.yml             ## If you use conda
```

<a name="installation"></a>

## 2. Installation
Python 3.11

You might need to change some constant variables such as VIDEO_PATH to run the code successfully.

```bash
git clone $THISREPO
cd $THISREPO
pip install -r .\requirements.txt
python src/run.py
```

<a name="repro"></a>
## 3. Reproduction
Demonstrate how your work can be reproduced, e.g. the results in your report.
```bash
mkdir tmp && cd tmp
wget https://yourstorageisourbusiness.com/dataset.zip
unzip dataset.zip
conda activate amazing
python evaluate.py --epochs=10 --data=/in/put/dir
```
Data can be found at ...
Output will be saved in ...

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
