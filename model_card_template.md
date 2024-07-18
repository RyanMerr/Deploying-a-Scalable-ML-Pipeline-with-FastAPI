# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Our model is a machine learning classifier trained to predict whether an individual's income exceeds $50,000 based on census data attributes. It utilizes a supervised learning approach with a binary classification objective.

## Intended Use
This model is intended to be used for predicting income levels based on demographic and employment-related attributes found in census data. It can assist in applications such as targeted marketing, resource allocation, and socio-economic analysis.

## Training Data
The training data consists of a subset of the 1994 U.S. Census database, obtained from the UCI Machine Learning Repository. It includes demographic features such as age, education level, occupation, marital status, and more. Categorical features were encoded using one-hot encoding, and labels ('salary' >50K or <=50K) were binarized.

## Evaluation Data
The evaluation data is a separate subset of the same census database, used to assess the model's performance on unseen data. It is processed similarly to the training data to ensure consistency in feature encoding and label binarization.

## Metrics
The model's performance is evaluated using the following metrics:

Precision: Indicates the proportion of true positive predictions among all positive predictions.
Recall: Measures the proportion of actual positives that were correctly identified by the model.
F1 Score: Harmonic mean of precision and recall, providing a single metric for evaluating model performance.
Precision: 0.7285
Recall: 0.2699
F1 Score: 0.3939

## Ethical Considerations
### Fairness:
Care was taken to ensure that the model's predictions do not systematically disadvantage any particular demographic group based on the protected attributes provided by the census data.
### Privacy:
The dataset used adheres to privacy policies and regulations. No personally identifiable information (PII) is exposed or used beyond what is publicly available in the census data.
### Transparency:
The model's predictions are interpretable, with feature importance analysis available to understand which factors influence predictions.

## Caveats and Recommendations
### Data Limitations:
The model's performance heavily relies on the quality and representativeness of the census data used for training and evaluation. Changes in demographics or societal factors not captured in the dataset may impact model accuracy.
### Use Recommendations:
Users should be aware of potential biases in the dataset and monitor model performance across different demographic groups to ensure equitable outcomes.