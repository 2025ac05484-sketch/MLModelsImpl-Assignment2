Mushroom Classification using Machine Learning

a. Problem Statement

The aim of this project is to classify mushrooms as edible or poisonous using machine learning. Five different classification models are trained and compared to find out which models perform best on the given dataset.

b. Dataset Description

The Mushroom Classification dataset contains different characteristics of mushrooms such as cap, gill, stalk and other features.

The target variable indicates whether a mushroom is edible or poisonous. Since most of the features are categorical, they are converted into numerical values before training the models.

c. Github Repository Link

https://github.com/2025ac05484-sketch/MLModelsImpl-Assignment2

The repository contains the dataset, preprocessing code, trained models, test data, requirements file and Streamlit application.

d. Models Used

The following five machine learning models were used:

1. Logistic Regression
2. Decision Tree
3. kNN
4. Naive Bayes
5. Random Forest

Model Comparison

| ML Model Name       | Accuracy |      AUC | Precision |   Recall | F1 Score |      MCC |
| ------------------- | -------: | -------: | --------: | -------: | -------: | -------: |
| Logistic Regression | 1.000000 | 1.000000 |  1.000000 | 1.000000 | 1.000000 | 1.000000 |
| Decision Tree       | 1.000000 | 1.000000 |  1.000000 | 1.000000 | 1.000000 | 1.000000 |
| kNN                 | 0.998769 | 1.000000 |  1.000000 | 0.997446 | 0.998721 | 0.997538 |
| Naive Bayes         | 0.945846 | 0.997185 |  0.990127 | 0.896552 | 0.941019 | 0.894943 |
| Random Forest       | 1.000000 | 1.000000 |  1.000000 | 1.000000 | 1.000000 | 1.000000 |

Observations on Model Performance

| ML Model Name       | Observation about model performance                                                                 |
| ------------------- | --------------------------------------------------------------------------------------------------- |
| Logistic Regression | Performs perfectly on the test data, with all the evaluation metrics equal to 1.                    |
| Decision Tree       | Gives perfect performance across all the evaluation metrics.                                        |
| kNN                 | Performs almost perfectly, with only a very small drop in recall, F1 score and MCC.                 |
| Naive Bayes         | Performs lower than the other models, especially in recall and MCC, but still gives good results.   |
| Random Forest       | Gives perfect performance across all the evaluation metrics and performs very well on this dataset. |

Overall Winner for the Dataset

Logistic Regression, Decision Tree and Random Forest are the top-performing models, with all three achieving perfect scores across the given evaluation metrics.

Among them, Random Forest can be considered the overall winner because it achieves perfect performance while also being an ensemble model that combines multiple decision trees.

Streamlit Application

The project also includes a Streamlit application that allows the trained models to be evaluated and compared using the test dataset.

The application displays the evaluation metrics, confusion matrix, classification report and comparison of all five models.

Streamlit Application:

https://2025ac05484-mushroom-classifier.streamlit.app/
