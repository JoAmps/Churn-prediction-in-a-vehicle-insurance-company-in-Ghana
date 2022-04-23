
# Project Name
Churn prediction in a Vehicle Insurance Company in Ghana

## Project Intro/Objective
Customer churn prediction is an important business activity for any company. The ability to predict this, is very key, so that companies identify such individuals and make the neccesary arrangements to keep them.
Since this insurance company makes its profit from the number of customers that subscribes to its insurance plans, its absolutely crucial, they know which customers are likely to leave them, so they make the neccesary arrangement to keep them, in order not to lose money and keep making profits.

### Methods Used
* Data exploration/descriptive statistics
* Data processing/cleaning
* Inferential Statistics
* Machine Learning
* Data Visualization
* Predictive Modeling
* Testing
* Deployment
* Monitoring

### Technologies
* Python
* Various python libraries for data science and machine learning
* Heroku
* Data Version Control(DVC)
* Visual studio code, jupyter
* Git
* Testing(pytest)
* CI/CD

## Project Description
#### As mentioned in the Project objective, predicting these customers that are likely to churn is crucial, thats why this insurance company, needs to know the customers that would cancel their subscription to the insurance they give.
#### Since this project is just a practice project to demonstrate my skills, i obtained this synthesized data from the insurance company, to use for my purposes. I obtained two data sets from them, The first was data about their customers from october 2021 to febrary 2022, and the second one was data about their customers in march 2022.
### The questions i deemed to explore were:
#### What are the most crucial factors that affect the customer cancelling their subscription with the company.
#### How strong is the relationship between the features(gender,type of plan,types of vehicles used etc), and whether they churned
#### Which model could be beter suited to predicting this, and be can generalize well on new customers, and also be fast enough during inference time
#### Which metric is the most important to evaluate the performance, are there any tradeoffs to consider

#### Univariate and bivariate visualizations were outputted to give more clarity and analysis on the features and how they relate to eachother and the target(churn)
### Some of the challenges faced was:
#### To determine the metric that would be useful to evaluate this,particularly between recall and precision 
#### Which leads to determining the threshold, and the tradeoff between false positive and false negatives


## Getting Started

1. Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).
2. Raw Data is being kept [datasets]
3. Vehicle insurance churn.ipynb contains the visualizations and experiments used for this
4. Data processing/transformation scripts are being kept [functions/data_preprocess.py]
5. Testing script is here [test_model.py]
6. Script for training is here [train.py]
7. Script for monitoring the model and data is here [monitoring.py]
8. Heroku scripts are here [Procfile,Aptfile,app.py]
5. Output contains the output of the code such as model metrics, log files, model

## To use this
1. Run python requirements -r.txt to install the neccesary python libraries
2. Run clean_data.py to run basic cleaning and obtain the datasets
3. Run train.py which runs the training and evaluation script to train the model and score the performance
4. Run test_model.py to run tests on various functions to ensure all parts function well
5. Push code to your own repo to trigger CI/CD
6. To perform inference, go to https://vehicle-company-churn.herokuapp.com/docs



## Results

![Confusion Matrix](https://github.com/JoAmps/Churn-prediction-in-a-vehicle-insurance-company-in-Ghana/blob/main/confusion_matrix.png)
#### From the original dataset of customers which comprised of 8800 customers, after several experimentation, using different models, the xgboost model performed the best, with a recall of 95.9% and precision of 97.7%
#### 15 false positives, which indicates 15 customers who actually churned, were predicted as not churned, and 6 false negatives, which means 6 customers did not churn, and they were predicted as churned was obtained
#### Though this result is good, the performance was improved by hyper parameter tuning
#### Now theres a decision to determine which is more suited to the business, so theres a tradeoff between false positives and false negatives
#### More false positives suggests that there are more customers who churned, that the model did not pick up, in such a case, if the number is high, the company would have to spend money and resources on those customers to get them to not churn, for example more personalized advertissment, more incentives and discounts

![roc_curve](https://github.com/JoAmps/Churn-prediction-in-a-vehicle-insurance-company-in-Ghana/blob/main/roc_auc_curve.png)
#### More false negatives suggests that there are more customers who did not churn, but the model predicted them to churn, in this situation, if this number is high, the company woud be spending time and resources on the wrong customers, 
#### So there is this tradeoff in what is important to the company, which situation they deem most profitable to pursue, using that knowledge, we adjust the threshold to take care of that
#### I used a threshold of 0.59, to reduce the false positives to 9, and increase the false negatives to 10, but this decision is based on the company needs and wants

![Feature Importance](https://github.com/JoAmps/Churn-prediction-in-a-vehicle-insurance-company-in-Ghana/blob/main/feature_importance.png)
## Monitoring
I created a monitoring script, which detects if new data is available(more customers), and if new data is available, performs inference on the data and compares the new recall score to the previous score, if the new score is equal or higher than the previous, it suggests that model drift has not occurred, and if the score is lower, it is a possible indication that model drift has occurred, and further investigation is required, such as model retraining and the likes


