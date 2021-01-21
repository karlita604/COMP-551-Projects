import numpy as np
import pandas as pd
import NaiveBayes as nb
import LogisticRegression as lg
import csv
import seaborn as sis
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

## KFOLD METHOD ##

def k_fold(X, Y, k):
    return np.array_split(X, k), np.array_split(Y, k)

## CLEANING DATA AND BASIC STATS (TASK 1)
##------------------------------------------------------------------------------------------------------------##
## ADULT ##
missing_values= ['?', 'nan', '--', 'non']
col_names=['Age','Workclass', 'fnlwgt' , 
'Education', 'Education-Number', 'Marital-Status', 'Occupation', 
'Relationship', 'Race', 'Sex', 'Capital_Gain', 'Capital_Loss', 'Hours/Week', 'Native Country','Y']
adult = pd.read_csv('data/adult/adult.data', header=None, names=col_names, na_values=missing_values)
feature_cols=['Age','Workclass', 'Education', 'Education-Number', 'Marital-Status', 'Occupation', 
'Relationship', 'Race', 'Sex', 'Hours/Week', 'Native Country']

#Clean the data - Following the procedure outlined in the WriteUp
#1. Removing fnlwgt and Captial_Loss and Capital_Gain
adult = adult.drop(['Capital_Gain', 'Capital_Loss', 'fnlwgt'], axis=1)
#Make Feature and Target vectors
X_ad = adult.iloc[:,:-1] #Features
y_ad = adult.Y #Target Variable

#2. OneHot Encoding
#Making into a binary classifier vector

labelencoder = LabelEncoder()
X_ad["Workclass"] = labelencoder.fit_transform(X_ad["Workclass"])
X_ad["Education"] = labelencoder.fit_transform(X_ad["Education"])
X_ad["Marital-Status"] = labelencoder.fit_transform(X_ad["Marital-Status"])
X_ad["Occupation"] = labelencoder.fit_transform(X_ad["Occupation"])
X_ad["Relationship"] = labelencoder.fit_transform(X_ad["Relationship"])
X_ad["Race"] = labelencoder.fit_transform(X_ad["Race"])
X_ad["Sex"] = labelencoder.fit_transform(X_ad["Sex"])
X_ad["Native Country"] = labelencoder.fit_transform(X_ad["Native Country"])
y_ad = labelencoder.fit_transform(y_ad)

#4. Basic Statistics ----------------------------------------------

#pair plot
#sis.pairplot(X_ad)
#plt.savefig('/Users/karlita/Desktop/COMP551/adultScatter.png')

#sis.set(color_codes=True)

#print("Basic Statistics")
#print("----------------")
#print("MEAN FOR NUMERICAL FEATURES")
#print(adult.mean())
#print("OCCUPATION MEAN and VARIANCE")
#print(statistics.mean(X_ad["Occupation"]))
#print(statistics.variance(X_ad["Occupation"]))

##Give an explanation of what this would mean



#print("SEX MEAN and VARIANCE")
#print(statistics.mean(X_ad["Sex"]))
#print(statistics.variance(X_ad["Sex"]))

##Give an explanation of what this would mean



#print("----------------")
#print("SOME DISTRIBUTIONS")
#print("Occupation")
#sis.distplot(X_ad["Occupation"])
#print("Education")
#sis.distplot(X_ad["Education"])
#print("Sex")
#sis.distplot(X_ad["Sex"])
#print("Race")
#sis.distplot(X_ad["Race"])

#Basic Plots - Seaborn
#sis.pairplot(pd.DataFrame(X_ad))
#plt.savefig('/Users/Daphne/Desktop/adultScatter.png')

## IONOSPHERE ##
c_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17',
           '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32',
           '33', '34', '35']
ionosphere = pd.read_csv('data/ionosphere/ionosphere.data', header=None, names=c_names)
#Dropping the second column because it is just a vector of 0's
ionosphere = ionosphere.drop(['2'], axis=1)
#Basic Plots - Seaborn
#sis.pairplot(ionosphere)
#plt.savefig('/Users/Daphne/Desktop/ionScatter.png')
#Make Features and Target vectors
X_ion = ionosphere.iloc[:,:-1].values
y_ion = ionosphere['35']
#One Hot Encode Target Values
y_ion = labelencoder.fit_transform(y_ion)

#Do some basic statistics
#Very little analysis to do on this data set and thus just included a few distributions of the data
#sis.distplot(X_ion)


## HABERMAN ##
column = ['age','year','num_aux','class']
hab_data = pd.read_csv('data/haberman/haberman.data', header=None, names=column)
feats = ['age','year','num_aux']

X_hab = hab_data[feats]
y_hab = hab_data['class']
#hab_data[hab_data['class'] == 1] = 0
#hab_data[hab_data['class'] == 2] = 1
y_hab[y_hab == 1] = 0
y_hab[y_hab == 2] = 1
y_hab = y_hab.to_numpy()
#Basic Plots - Seaborn
#sis.pairplot(X_hab)
#plt.savefig('/Users/Daphne/Desktop/habScatter.png')

#sis.distplot(X_hab['age'])
#sis.distplot(X_hab['year'])
#sis.distplot(X_hab['num_aux'])

## BANK ##
column = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'Y']
bank_data = pd.read_csv('data/bank/bank.txt', header=None, sep=";", names=column)
feats = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome']

X_bank = bank_data[feats]
# Basic Plots - Seaborn
#sis.pairplot(X_bank)
#plt.savefig('/Users/Daphne/Desktop/bankScatter.png')
y_bank = bank_data['Y']
y_bank = pd.factorize(y_bank)[0] #replace yes with 1, no with 0
bank_data['Y'] = pd.factorize(bank_data['Y'])[0]
#bank has no missing values
#one hot
categorical_bank = X_bank.select_dtypes(include=[object])
non_catagorical_bank = X_bank.select_dtypes(exclude=[object])
encoder = LabelEncoder()
encoded_data_bank = categorical_bank.apply(encoder.fit_transform)
X_bank = np.concatenate((encoded_data_bank, non_catagorical_bank),axis=1)

#Basic Statistics

#pair plot

#sis.pairplot(bank_data[feats])
#plt.savefig('/Users/Daphne/Desktop/bankScatter.png')
#sis.distplot(bank_data[['age']])
#sis.distplot(bank_data[['balance']])
#sis.distplot(bank_data[['day']])
#sis.distplot(bank_data[['duration']])

## KFOLD SCRIPT (TASK 2) ##
##------------------------------------------------------------------------------------------------------------##
k=5

adult_datasets_X_full, adult_datasets_Y_full = k_fold(np.array(X_ad), np.array(y_ad), k)
ion_datasets_X_full, ion_datasets_Y_full = k_fold(np.array(X_ion), np.array(y_ion), k)
hab_datasets_X_full, hab_datasets_Y_full = k_fold(np.array(X_hab), np.array(y_hab), k)
bank_datasets_X_full, bank_datasets_Y_full = k_fold(np.array(X_bank), np.array(y_bank), k)


bayes = nb.NaiveBayes()
logistic = lg.LogisticRegression(0.01, 10000)
#0 = categorical (one hot encoded for bernoulli likelihood), 1 = continuous (gaussian likelihood)
feats_ad = [1,0,0,1,0,0,0,0,0,1,0]
feats_ion = [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
feats_hab = [1,1,1]
feats_bank = [1,0,0,0,0,1,0,0,0,1,0,1,1,1,1,0]

#index of test set in the split sets will be i
lg_acc_ad, lg_acc_ion, lg_acc_hab, lg_acc_bank = 0, 0, 0, 0
nb_acc_ad, nb_acc_ion, nb_acc_hab, nb_acc_bank = 0, 0, 0, 0


for i in range(k):
    #make copy of original split
    adult_datasets_X, adult_datasets_Y = adult_datasets_X_full.copy(), adult_datasets_Y_full.copy()
    ion_datasets_X, ion_datasets_Y = ion_datasets_X_full.copy(), ion_datasets_Y_full.copy()
    hab_datasets_X, hab_datasets_Y = hab_datasets_X_full.copy(), hab_datasets_Y_full.copy()
    bank_datasets_X, bank_datasets_Y = bank_datasets_X_full.copy(), bank_datasets_Y_full.copy()

    #get test sets at i
    test_adult_X = np.array(adult_datasets_X.pop(i))
    test_adult_Y = np.array(adult_datasets_Y.pop(i))
    test_ion_X = np.array(ion_datasets_X.pop(i))
    test_ion_Y = np.array(ion_datasets_Y.pop(i))
    test_hab_X = np.array(hab_datasets_X.pop(i))
    test_hab_Y = np.array(hab_datasets_Y.pop(i))
    test_bank_X = np.array(bank_datasets_X.pop(i))
    test_bank_Y = np.array(bank_datasets_Y.pop(i))

    #get training sets at all other indexes
    train_adult_X = np.concatenate(np.array(adult_datasets_X))
    train_adult_Y = np.concatenate(np.array(adult_datasets_Y))
    train_ion_X = np.concatenate(np.array(ion_datasets_X))
    train_ion_Y = np.concatenate(np.array(ion_datasets_Y))
    train_hab_X = np.concatenate(np.array(hab_datasets_X))
    train_hab_Y = np.concatenate(np.array(hab_datasets_Y))
    train_bank_X = np.concatenate(np.array(bank_datasets_X))
    train_bank_Y = np.concatenate(np.array(bank_datasets_Y))

    #add ones for the training sets of logistic
    ones = np.zeros((len(train_adult_X), 1))
    train_adult_X_ones = np.hstack((ones, train_adult_X))
    ones = np.zeros((len(test_adult_X), 1))
    test_adult_X_ones = np.hstack((ones, test_adult_X))
    ones = np.zeros((len(train_ion_X), 1))
    train_ion_X_ones = np.hstack((ones, train_ion_X))
    ones = np.zeros((len(test_ion_X), 1))
    test_ion_X_ones = np.hstack((ones, test_ion_X))
    ones = np.zeros((len(train_hab_X), 1))
    train_hab_X_ones = np.hstack((ones, train_hab_X))
    ones = np.zeros((len(test_hab_X), 1))
    test_hab_X_ones = np.hstack((ones, test_hab_X))
    ones = np.zeros((len(train_bank_X), 1))
    train_bank_X_ones = np.hstack((ones, train_bank_X))
    ones = np.zeros((len(test_bank_X), 1))
    test_bank_X_ones = np.hstack((ones, test_bank_X))

    #LOGISTIC
    w_ad = logistic.fit(train_adult_X_ones, train_adult_Y)
    w_ion = logistic.fit(train_ion_X_ones, train_ion_Y)
    w_hab = logistic.fit(train_hab_X_ones, train_hab_Y)
    w_bank = logistic.fit(train_bank_X_ones, train_bank_Y)

    p_ad = logistic.predict(test_adult_X_ones, w_ad)
    p_ion = logistic.predict(test_ion_X_ones, w_ion)
    p_hab = logistic.predict(test_hab_X_ones, w_hab)
    p_bank = logistic.predict(test_bank_X_ones, w_bank)

    lg_acc_ad += logistic.eval_acc(p_ad, test_adult_Y)
    lg_acc_ion += logistic.eval_acc(p_ion, test_ion_Y)
    lg_acc_hab += logistic.eval_acc(p_hab, test_hab_Y)
    lg_acc_bank += logistic.eval_acc(p_bank, test_bank_Y)

    #NAIVE BAYES
    priors_ad, means_ad, stdevs_ad, prob_ad = bayes.fit(train_adult_X, train_adult_Y, feats_ad)
    priors_ion, means_ion, stdevs_ion, prob_ion  = bayes.fit(train_ion_X, train_ion_Y, feats_ion)
    priors_hab, means_hab, stdevs_hab, prob_hab = bayes.fit(train_hab_X, train_hab_Y, feats_hab)
    priors_bank, means_bank, stdevs_bank, prob_bank = bayes.fit(train_bank_X, train_bank_Y, feats_bank)

    p_ad = bayes.predict(test_adult_X, means_ad, stdevs_ad, prob_ad, priors_ad, feats_ad)
    p_ion = bayes.predict(test_ion_X, means_ion, stdevs_ion, prob_ion, priors_ion, feats_ion)
    p_hab = bayes.predict(test_hab_X, means_hab, stdevs_hab, prob_hab, priors_hab, feats_hab)
    p_bank = bayes.predict(test_bank_X, means_bank, stdevs_bank, prob_bank, priors_bank, feats_bank)

    nb_acc_ad += bayes.eval_acc(p_ad, test_adult_Y)
    nb_acc_ion += bayes.eval_acc(p_ion, test_ion_Y)
    nb_acc_hab += bayes.eval_acc(p_hab, test_hab_Y)
    nb_acc_bank += bayes.eval_acc(p_bank, test_bank_Y)

#average accuracies
lg_acc_ad /= k
lg_acc_ion /= k
lg_acc_hab /= k
lg_acc_bank /= k
nb_acc_ad /= k
nb_acc_ion /= k
nb_acc_hab /= k
nb_acc_bank /= k

print("k = %s" % k)
print("----------------------------------------------")
print("LOGISTIC REGRESSION")
print("----------------------------------------------")
print("ADULT:")
print(lg_acc_ad)
print("IONOSPHERE:")
print(lg_acc_ion)
print("HABERMAN:")
print(lg_acc_hab)
print("BANK:")
print(lg_acc_bank)

print("\nNAIVE BAYES")
print("----------------------------------------------")
print("ADULT:")
print(nb_acc_ad)
print("IONOSPHERE:")
print(nb_acc_ion)
print("HABERMAN:")
print(nb_acc_hab)
print("BANK:")
print(nb_acc_bank)






    

