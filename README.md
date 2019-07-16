# Project Details
**Student name**: Andrew Au  
**Student number**: z5020593  
**Project title**: A Computer Model of Electrocardiogram Signals  
**Supervisor**: Socrates Dokos  
**Assessor**: Bruno Gaeta  

**Description**: This project seeks to implement a dynamic computer model of electrocardiogram signals that generates synthetic ECG waveforms. The implementation of such a model will allow user's to upload experimental data and perform associated parameter fitting with a user interface.  

**Abstract**: An electrocardiogram (ECG) is a record of electrical signals produced by the heart when it contracts and relaxes. The function of the electrocardiogram is to diagnose the physiological condition of the human heart.  
A synthetic ECG waveform can be generated dynamically through the use of a computer model, which would have benefits and uses across teaching, research and evaluation of other biomedical signal techniques.
The immediate objective of this research project is to implement a dynamic computer model of an electrocardiogram that simulates the standard characteristics of the human heart.  
The implementation of this model allows user's to upload data and perform parameter fitting to simulate different waveforms.  
In the future this computer model may be distributed over different platforms for user's to upload their own experimental data for testing.


# Setup Instructions
## Requirements
1. Python3.6
2. pip
3. virtualenv

## Setup
1. `git clone` this repository
2. `cd` into the root of the cloned repo
2. `virtualenv venv -p python3.6`
3. `source venv/bin/activate`
4. `pip install -r requirements.txt`

## Making Builds
Run `make build` in the root directory.

## User setup
N/A


# Resources and references
## Project links
[CSE Thesis Database](https://moodle.telt.unsw.edu.au/course/view.php?id=33523)  
[Module A](https://moodle.telt.unsw.edu.au/course/view.php?id=40119)

## Writing Resources
[Plagiarism Quiz](https://student.unsw.edu.au/plagiarism-quiz)  
[Writing in engineering](https://student.unsw.edu.au/writing-engineering-science)  
[Write your research](https://writeyourresearch.wordpress.com/)  
[Writing an abstract](https://services.unimelb.edu.au/__data/assets/pdf_file/0007/471274/Writing_an_Abstract_Update_051112.pdf)  

## Technical Reading
[Electrocardiogram Model](http://web.mit.edu/~gari/www/papers/ieeetbe50p289.pdf)
