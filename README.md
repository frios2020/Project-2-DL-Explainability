# Project 2: Explainable COVID-19 Pneumonia
================================================================

# NJIT Data Science Program
# Course
# CS677 Deep Learning - Fall 2020
Instructor
 
Pantelis Monogioudis, Ph.D Professor of Practice, NJIT & Adjunct NYU

Teaching Assistant

Nitesh Mistry

Students:
<br>Fernando Rios
<br>Hassan Ouanir
<br>Maha Faruque


Description

In this project, the goals are: (1) to explore development of a machine learning algorithm to distinguish chest X-rays of individuals with respiratory illness testing positive for COVID-19 from other X-rays, (2) to promote discovery of patterns in such X-rays via machine learning interpretability algorithms. 

Due to doctors are reluctant to accept black box algorithms such as your deep learning based method - as an AI engineer we need to listen to them and try to satisfy their needs, they are your customer after all. They tell you that your automated diagnostic system that processes the imaging they give you, must be explainable.

We need to follow theses tasks to achieve the goal: They give you the COVID X-ray / CT Imaging dataset and:
1. Find this this implementation of the method called Local Interpretable Model-Agnostic Explanations (i.e. LIME). You also read this article and you get your hands dirty and replicate the results in your colab notebook with GPU enabled kernel.
2. A fellow AI engineer, tells you about another method called SHAP that stands for SHapley Additive exPlanations and she mentions that Shapley was a Nobel prize winner so it must be important. You then find out that Google is using it and wrote a readable white paper about it and your excitement grows. Your manager sees you on the corridor and mentions that your work is needed soon. You are keen to impress her and start writing your 3-5 page summary of the SHAP approach as can be applied to explaining deep learning classifiers such as the ResNet network used in (1).
3. After your presentation, your manager is clearly impressed with the depth of the SHAP approach and asks for some results for explaining the COVID-19 diagnoses via it. You notice that the extremely popular SHAP Github repo already has an example with VGG16 network applied to ImageNet. You think it wont be too difficult to plugin the model you trained in (1) and explain it. 

Part 1
We created the notebook "CS677 Project 2 Explainability". It is prepared to run easily, just select option run step by step and update the config.yml file with the correct paths
 
In its lines this notebook follow these steps.
<br>Step 1.0 Cloning repository covid-cxr
<br>Step 2.0 Installing requirements
<br>Step 3.0 Creating folder Raw_Data to contain all of your raw data. Set the RAW_DATA field in the PATHS 
<br>Step 4.0 Cloning the covid-chestxray-dataset repository inside of the RAW_DATA folder.
<br>Step 5.0 Cloning the Figure1-COVID-chestxray-dataset repository inside of RAW_DATA folder.
<br>Step 6.0 Download and unzip the RSNA Pneumonia Detection Challenge dataset from Kaggle somewhere on your local machine. 
<br>Step 7.0 Execute preprocess.py to create Pandas DataFrames of filenames and labels. Preprocessed DataFrames and corresponding images of the dataset will be saved within data/processed/.
<br>Step 8.0 Execute train.py to train the neural network model. The trained model weights are saved within results/models/, and its filename will resemble the following structure: 	
<br>Step 9.0 Execute lime_explain.py to generate interpretable explanations for the model's predictions on the test set.
	!python src/interpretability/lime_explain.py

To verify the output of each step you can read the documentation.docx file.
To visualize the log file we use TensorBoard (Depicting loss on the training and validation sets versus epochs, roc curve and confusion matrix on test sets)
 
Part 2 (See file "Part 2 - SHAP Approach Summary .docx")

Part 3
Go to the part 3 section, and run the lines. It will show the explanation using SHAP approach.


All contributions are welcome!
