"""
KNN Project
Modified the LUNG Cancer Data Set and use the kNN Algorithm. Ran the algorithm 1000 times for each value of k=1,2,3,...,20. For each value of k find the mean test and training accuracy. Plot these two mean accuracy rates vs k on a single plot. 

reference:
https://www.youtube.com/watch?v=XSoau_q0kz8
https://towardsdatascience.com/building-a-k-nearest-neighbors-k-nn-model-with-scikit-learn-51209555453a

"""

"""
Algorithm:

1) connvert yes and no to 0 and 1 for normalizing data.
2) target variable: LUNG_CANCER.
3) Filter down attributes (can remove attributed that dont nned to be used)
4) Normalise the data (https://www.journaldev.com/45109/normalize-data-in-python)
5) Divide data into traning data and Testing data.
https://stackoverflow.com/questions/24147278/how-do-i-create-test-and-train-samples-from-one-dataframe-with-pandas
6) Create traning data and testing data dataframes.
https://www.marsja.se/how-to-convert-numpy-array-to-pandas-dataframe-examples/
https://www.kite.com/python/answers/how-to-convert-a-2d-list-into-a-numpy-array-in-python
7) Train the model using KNN algorithm https://www.datacamp.com/community/tutorials/k-nearest-neighbor-classification-scikit-learn
8) Run the algorithm 1000 times for each value of k=1,2,3,...,20. For each value of k find the mean test and training accuracy
9)  find the mean test and training accuracy
10)  Plot these two mean accuracy rates vs k on a single plot

"""
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

#importing dataset using pandas
lung_cancer_data = pd.read_csv (r'survey lung cancer.csv')

"""
1) connvert yes and no to 0 and 1 for normalizing data.
2) target variable: LUNG_CANCER.
3) Filter down attributes (can remove attributed that dont nned to be used)

dropping gender bec its categorical.
dropping peer pressure cause not relevant for cancer.
dropping yellow fingers
https://mindmatters.ai/2020/12/yellow-fingers-do-not-cause-lung-cancer/

Dropping Attributes
https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop.html
"""
#getting data
lung_cancer_data = lung_cancer_data.drop(['GENDER', 'PEER_PRESSURE', 'YELLOW_FINGERS'], axis=1)

#gettinng target data
lung_cancer_target_str = lung_cancer_data.iloc[: , -1]

#creating labelEncoder
le = preprocessing.LabelEncoder()

# converting string labels into numbers
lung_cancer_target= le.fit_transform(lung_cancer_target_str)

#updating data
lung_cancer_data = lung_cancer_data.drop(['LUNG_CANCER'], axis=1)

#getting names
names = lung_cancer_data.columns

"""
4) Normalise the data (https://www.journaldev.com/45109/normalize-data-in-python)
"""
d = preprocessing.normalize(lung_cancer_data, axis=0)
lung_cancer_data = pd.DataFrame(d, columns=names)

lung_cancer_target = lung_cancer_target.reshape(1, -1)

"""
5) Divide data into traning data and Testing data.
https://stackoverflow.com/questions/24147278/how-do-i-create-test-and-train-samples-from-one-dataframe-with-pandas
"""
#transpose for same shape of data and target (https://datascience.stackexchange.com/questions/20199/train-test-split-error-found-input-variables-with-inconsistent-numbers-of-sam)
lung_cancer_target = lung_cancer_target.transpose()

normal_lung_cancer_data_train_x, normal_lung_cancer_data_test_x, normal_lung_cancer_data_train_y, normal_lung_cancer_data_test_y = train_test_split(lung_cancer_data, lung_cancer_target, test_size=0.2, random_state=1)
 
"""
6) Run the algorithm 1000 times for each value of k=1,2,3,...,20.
For each value of k find the mean test and training accuracy. 
Plot these two mean accuracy rates vs k on a single plot.(https://swcarpentry.github.io/python-novice-gapminder/09-plotting/index.html)
(https://towardsdatascience.com/building-a-k-nearest-neighbors-k-nn-model-with-scikit-learn-51209555453a)

"""
sum_test_accuracy_score = 0
mean_test_accuracy_score = 0

sum_train_accuracy_score = 0
mean_train_accuracy_score = 0

mean_test_accuracy_score_list = []
mean_train_accuracy_score_list = []

for i in range(0, 1000):
    print(f"---START RUNNING algorithm {i+1} time---")
    for j in range(1, 21):
        
        # Create KNN classifier
        knn = KNeighborsClassifier(n_neighbors = j)
        print("Algorithm for value of k: ", j)

        # Fit the classifier to the data
        #https://stackoverflow.com/questions/34165731/a-column-vector-y-was-passed-when-a-1d-array-was-expected
        knn.fit(normal_lung_cancer_data_train_x ,  np.ravel(normal_lung_cancer_data_train_y,order='C')  )


        #show first 5 model predictions on the test data
        predection = knn.predict(normal_lung_cancer_data_test_x)[0:5]

        #check accuracy of our model on the test data
        test_accuracy_score = knn.score(normal_lung_cancer_data_test_x, normal_lung_cancer_data_test_y)
        train_accuracy_score = knn.score(normal_lung_cancer_data_train_x, normal_lung_cancer_data_train_y)

        #CALCULATING MEAN OF TRANING AND ACCURACY DATA for each value of k
        #when k reaches 20 compute all scores and save to the list of where mean values will be saved.
        #for the case, where k is not 20, keep saving sum to variable
        if j == 20:
            mean_test_accuracy_score = (sum_test_accuracy_score / j)
            mean_train_accuracy_score = (sum_train_accuracy_score / j)

            print("mean_test_accuracy_score: ", mean_test_accuracy_score)
            print("mean_train_accuracy_score: ", mean_train_accuracy_score)

            mean_test_accuracy_score_list.append(mean_test_accuracy_score)
            mean_train_accuracy_score_list.append(mean_train_accuracy_score)

            sum_test_accuracy_score = 0
            sum_train_accuracy_score = 0
            mean_test_accuracy_score = 0
            mean_train_accuracy_score = 0



        else:
            sum_test_accuracy_score = sum_test_accuracy_score + test_accuracy_score
            sum_train_accuracy_score = sum_train_accuracy_score + train_accuracy_score

#Plot these two mean accuracy rates vs k on a single plot.
k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

print("k values: ", len(k_values) )
print("mean_test_accuracy_score_list: ", len(mean_test_accuracy_score_list) )
print("mean_train_accuracy_score_list: ", len(mean_train_accuracy_score_list) )



#https://www.geeksforgeeks.org/validation-curve/

# Setting the range for the parameter (from 1 to 10)
parameter_range = np.arange(1, 1001, 1)

# Plot mean accuracy scores for training and testing scores
plt.plot(parameter_range, mean_train_accuracy_score_list, label = "Training Score", color = 'b')
plt.plot(parameter_range, mean_test_accuracy_score_list, label = "Testing Score", color = 'g')

# Creating the plot
plt.title("Validation Curve with KNN Classifier")
plt.xlabel("k values")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.legend(loc = 'best')
plt.show()

"""
#test plot (https://swcarpentry.github.io/python-novice-gapminder/09-plotting/index.html)
print("test plot")
time = [0, 1, 2, 3]
second = []
for i in range (0, 60):
    second.append(i)
position = [0, 100, 200, 300]


plt.plot(time, position, second)
plt.xlabel('Time (hr)')
plt.ylabel('Position (km), second')
plt.show()
"""