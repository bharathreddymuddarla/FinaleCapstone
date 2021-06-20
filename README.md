Springboard ML Capstone



1.Capstone Title: 
Insurance policy purchase prediction based on customer’s shopping history

2.Problem Statement:
When a customer reaches out to an auto insurance company seeking for a policy, he/she is presented with many coverage options, which confuses the customer and the company may potentially lose the customer.

 
3.Why this is an interesting problem

A well modeled and trained machine learning application could predict the customer needs based on the purchase history and suggest an appropriate insurance policy to the customer. This would decrease the gap and benefit both customers and the company.


4.Data Collection

For this project, we used the data sets provided by All State Insurance company on kaggle:

Kaggle link: https://www.kaggle.com/c/allstate-purchase-prediction-challenge/data

Data set are provided in 3 csv files of total size 9 MB - 
Sample submissions
Test and
Train
Sample submissions file has two fields: customer_ID and plan.
Test and Train files have (25 different fields which define the customer and policy characteristics).
The training and test sets contain transaction history for customers that ended up purchasing a policy. For each customer_ID, you are given their quote history. In the training set you have the entire quote history, the last row of which contains the coverage options they purchased. In the test set, you have only a partial history of the quotes and do not have the purchased coverage options. These are truncated to certain lengths to simulate making predictions with less history (higher uncertainty) or more history (lower uncertainty).

5.Data Analysis

Below are the observations for the test data.
Total 665,249 records
25 columns
customer_ID - A unique identifier for the customer
shopping_pt - Unique identifier for the shopping point of a given customer
record_type - 0=shopping point, 1=purchase point
day - Day of the week (0-6, 0=Monday)
time - Time of day (HH:MM)
state - State where shopping point occurred
location - Location ID where shopping point occurred
group_size - How many people will be covered under the policy (1, 2, 3 or 4)
homeowner - Whether the customer owns a home or not (0=no, 1=yes)
car_age - Age of the customer’s car
car_value - How valuable was the customer’s car when new
risk_factor - An ordinal assessment of how risky the customer is (1, 2, 3, 4)
age_oldest - Age of the oldest person in customer's group
age_youngest - Age of the youngest person in customer’s group
married_couple - Does the customer group contain a married couple (0=no, 1=yes)
C_previous - What the customer formerly had or currently has for product option C (0=nothing, 1, 2, 3,4)
duration_previous -  how long (in years) the customer was covered by their previous issuer
A,B,C,D,E,F,G - the coverage options
cost - cost of the quoted coverage options
Different data types -
a. time is identified as object
b. state and car_value are other columns which are also identified as objects
c. risk_factor, C_previous and duration_previous are identified as float
d. all other columns are identified as int 
Following columns have Null values - car_value, risk_factor, C_previous, duration_previous
column name 'customer_ID' is inconsistent
Missing values:
a. 240,418 records does not have value for risk_factor, which approximates to 36% of total records
b. 1,531 records does not have value for car_value, which approximates to 0.23% of total records
c. 18,711 records does not have value for C_previous, which approximates to 2.81% of total records
d. 18,711 records does not have value for duration_previous, which approximates to 2.81% of total records

6.Approach to solve the problem

Is this a supervised or unsupervised problem?
We have both train and test data sets to evaluate the model. We can use supervised ML models.
If supervised, is it a classification or regression problem?
We need to predict whether a plan would be selected by the customer, this is a classification problem.
What are you trying to predict?
We are trying to predict the record type column value for a given record, which is 1 in case the policy was purchased or otherwise the value is 0.
What will you use as predictors?
We have several columns in the data set which can be used as the features/predictors.


7.Data Observation

Handling missing values:
‘Risk_factor’ column does not have values for 36% of records. This column gives the ordinal assessment of how risky the customer is and has values ranging from 1 to 4. Rounded mean value of this column is 3. Decided to replace null values with rounded mean values on the column.
The other two columns having null values were ‘car_value’ and ‘C_previous’ with 0.23% and 2.81% of total records respectively. As the percentage of records was less, dropped the records having null values in any of the two columns.

Reducing features:
Dropped ‘customer_id’ column as it is a unique identifier. Also dropped the ‘time’ column, as there is no logical relation between time and the purchase of the plan, we dropped the time column.

Data types conversions:
Converted all decimal columns to int type.
Converted 'state', 'location', 'group_size', 'homeowner', 'car_value', 'risk_factor', 'married_couple', 'c_previous', 'a', 'b', 'c', 'd', 'e', 'f' and 'g' columns to categorical, and applied label encoder.

Feature importance:
Below figure depicts the feature importance generated by random forest regressor.


8.Evaluation of the Models

Applied Logistic regression, KNeighborsClassifier, RandomForestClassifier and OneClassSVM on the given train data set and evaluated the model on the test data set. Below are the results of different models.
LogisticRegression

	Confusion_matrix:
	[[4168 1357]
 [ 245  687]]

	classification_report:
	



KNeighborsClassifier

When n_neighbors=6

Confusion_matrix:
[[3726 1799]
 [ 583  349]]

classification_report:






RandomForestClassifier

Confusion_matrix:
[[4803  722]
 [ 480  452]]

classification_report:

OneClassSVM

When kernel='rbf' and C=1.0

Confusion_matrix:
[[2582  155]
 [ 350  142]]

classification_report:


When kernel='rbf' and C=2.0

Confusion_matrix:
[[2737    0]
 [ 492    0]]

classification_report:



When kernel='rbf', cache_size=7000,C=0.5

Confusion_matrix:
[[   0 2737    0]
 [   0    0    0]
 [   0  492    0]]


classification_report:


	When kernel='rbf', cache_size=7000,C=0.8

Confusion_matrix:
[[   0 2737    0]
 [   0    0    0]
 [   0  492    0]]


classification_report:



	When kernel='linear'

	Confusion_matrix:
	[[1370 1367]
 [ 240  252]]

	classification_report:
	



9.Challenges
Data imbalance:
Data set we have is an imbalanced data set, Out of 654,648 records only 96173 records are classified as ‘1’ rest are classified as ‘0’, making it only 15% of the total data set. Model would not have enough samples to identify the records which are classified as ‘1’. To overcome this problem and balance the data set, we have applied the SMOTE technique.
Compute time on local machine:
While running One class SVM model, on a local machine, the model took a very long time to fit and predict the dataset. We have reduced the data set to 10% before fitting the model.
Also while fitting the OneClassSVM model we set cache_size=7000 MB.


10.Deployment Solution Architecture

Model will be deployed in a container service as a REST API. Client will pass the features as the parameters and API would respond if it is of class 1 or 0.



11.Lessons Learnt

Working on this project gave me very good insight on different machine learning models and techniques. Some of the lessons learnt - 

If we include unique identifiers features in the model, it is trying to learn from those features giving high importance to it, but logically it does not have any impact on the successful record.

As we had imbalance data (very less percent of successful records) in the train/test data set, the model could not learn much on the successful records, even though the scores are high, when we look at the confusion matrix, the model has very less prediction rate on the true positive records. After applying SMOTE technique (to balance the successful and unsuccessful records), the model has improved its performance significantly.

Feature importance on the features A,B,C,D,E,F,G (the coverage option features) changes on changing them to categorical values.


12.Code

CapstoneStone.ipynb: Code for Data wrangling, applying models, tuning and evaluation.

FeatureImportance.ipynb: Code for feature importance extraction.

Capstone_DeepLearning.ipynb: Applying Deep learning


13.GitHub Links

Capstone project: https://github.com/bharathreddymuddarla/FinaleCapstone
Test data:https://github.com/bharathreddymuddarla/FinaleCapstone/tree/main/data
