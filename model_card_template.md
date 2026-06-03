# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This is a random forest classifier built with scikit-learn to predict wether a person's annual income is greater than 50k. It uses census deomgraphic and employment data. This model uses 100 decision trees, whose combined predictions improve accuracy and reduce overfitting. A random state of 123 is used to ensure reproducible results. The trained model and preprocessing omponets are saved as serialized joblib files.

## Intended Use
This model is an educational demonstration of deplying a machine learnign pipeline with FastAPI. It can predict whether an idividual's aual icome exeeds 50k using demographic nad employment data from the U.S. Census. It is intended for students, researchers, and data scientists. This model is not suitable for real world decision making.
## Training Data
This model uses the UCI Adult Census Income dataset from the 1994 U.S. Census, which contains 32,562 records with 14 features and one target variable. For training, the data was split using an 80/20 ratio, which results in about 26,050 samples for training and 6,513 samples for testing. The split was performed using scikit-learn's train_test_split function with a random state of 123 to ensure reproducibility. Features include both numberical attributes (such as age and hours worked per week) and categorical attributes (such as occupation and marital status). The target variable indicates whether the income is over or below 50k, with the dataset being imbalanced toward the lower-income class. During preprocessing, categorical variables are one hot encoded with handeling unseen categories, while the target is binarized into 0/1 labels, and numerical features are left unscaled because the random forest models are not sensitive to feature scaling.

## Evaluation Data
The evaluation set contains 6,513 held out samples, which is 20 percent of the dataset. This was used to assess the model performance after training. The same preprocessing steps and fitted encoders from the training data are applied to avoid data leakage and ensure realistic evaluation conditions. The test set preserves the original class imbalance, which was about 75% being less than 50k and the other 25% being over 50k, enabling a reliable estimate of how the model is expected to perform on unseen data from the same distribution.
## Metrics
_Please include the metrics used and your model's performance on those metrics._

This model is evaluated using precision, recall, and f1-score. Accuracy is not the primary metric because a modeli that always predicts over 50k could still achieve about 75% accuracy while failing to dientify high earners. The scores are as follows: Precision: 0.7385, meaning tht the model is correct 74% of the time when predictiong when someone ears over 50k. Recall: 0.6248, means the model successfully identifies 62% of al people who actually earn over 50k.Lastly, F1: 0.6769 combines precision and recall into a single balanced metric, providing an overall assessment of model performance on the minority class.  
## Ethical Considerations

## Caveats and Recommendations
