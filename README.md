# UrbanSound8K-DeepLearning-Classification

## Project Description
This project was developed for the Machine Learning II course (2024/2025) at DCC/FCUP.
The main objective is to develop deep learning classifiers for urban sound data.
The goal is to build classifiers capable of assigning unseen sound excerpts to one of 10 specific classes.

## Dataset
The project utilizes the **UrbanSound8K** dataset, which consists of 8732 labeled sound excerpts,
each with a duration of 4 seconds or less.
The sounds are categorized into the following 10 classes:
* Air conditioner 
* Car horn 
* Children playing 
* Dog bark 
* Drilling 
* Engine idling 
* Gun shot 
* Jackhammer 
* Siren 
* Street music 

## Methodology
The implementation requires developing **two** out of the three following classifier types:
1.  **Multilayer Perceptron (MLP)** 
2.  **Convolutional Neural Network (CNN)** 
3.  **Recurrent Neural Network (RNN)** 

For each classifier, the workflow includes the following stages:
* **Data Pre-processing:** Processing raw sound data to uniformize and normalize inputs, potentially using libraries such as `librosa` for feature extraction.
* **Model Architecture Definition:** Defining layers, neurons, and activation functions for MLPs; choosing between 1D (windows) or 2D (MFCCs) inputs for CNNs; or selecting unit types (e.g., LSTM, GRU) for RNNs .
* **Training Strategies:** Determining optimizers, learning hyperparameters, and regularization techniques .
* **Performance Evaluation:** Assessing the model results.

## Evaluation Strategy
The performance of the classifiers is evaluated using a **10-fold cross-validation** scheme:
* **Training:** 8 folds
* **Validation:** 1 fold
* **Test:** 1 fold

Final classification performance is quantified using a cumulative confusion matrix,
average classification accuracy, and the standard deviation across the 10 experiments.

### Dataset
- Salamon, J., Jacoby, C., & Bello, J. P. (2014). A Dataset and Taxonomy for Urban Sound Research. 
  In *22nd ACM International Conference on Multimedia* (pp. 1041-1044). (https://urbansounddataset.weebly.com/urbansound8k.html)

## Contributors
* **Henrique Teixeira**
* **João Ferreira**
* **Leonor Carvalho**
