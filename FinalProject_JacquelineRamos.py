"""
    Name: Jacqueline Ramos
    Final Project: This program creates a decision tree classifier (using sklearn)
    from the COVIDiSTRESS Global Survey dataset to predict COVID-19 safety compliance levels.
"""

from numpy.ma.extras import average
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree, metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import confusion_matrix, classification_report
from graphviz import Source
import pydotplus
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


# read dataset into pandas dataframe
df = pd.read_csv('COVIDiSTRESS_May_30_cleaned_final.csv', sep=',')

print('\n----- Data Processing of COVIDiSTRESS Dataset -----')
print('\nInitial Dataframe Info:')
print(df.info(verbose=True))


# dataframe with only people who answered all questions
df_yes = df[df.answered_all == 'Yes']


cleaned_columns = [
    'ID', 
    'RecordedDate', 
    'UserLanguage', 
    'Dem_age', 
    'Dem_gender', 
    'Dem_edu', 
    'Dem_employment',
    'Country',
    'Dem_Expat',
    'Dem_maritalstatus',
    'Dem_dependents',
    'Dem_riskgroup',
    'Dem_isolation',
    'Dem_isolation_adults',
    'Dem_isolation_kids',
    'AD_gain', 
    'AD_loss',
    'Scale_PSS10_UCLA_1',
    'Scale_PSS10_UCLA_2', 
    'Scale_PSS10_UCLA_3', 
    'Scale_PSS10_UCLA_4', 
    'Scale_PSS10_UCLA_5', 
    'Scale_PSS10_UCLA_6', 
    'Scale_PSS10_UCLA_7', 
    'Scale_PSS10_UCLA_8', 
    'Scale_PSS10_UCLA_9', 
    'Scale_PSS10_UCLA_10',
    'Scale_SLON_1', 
    'Scale_SLON_2', 
    'Scale_SLON_3',
    'OECD_people_1', 
    'OECD_people_2', 
    'OECD_insititutions_1', 
    'OECD_insititutions_2', 
    'OECD_insititutions_3', 
    'OECD_insititutions_4', 
    'OECD_insititutions_5', 
    'OECD_insititutions_6', 
    'Corona_concerns_1', 
    'Corona_concerns_2', 
    'Corona_concerns_3', 
    'Corona_concerns_4', 
    'Corona_concerns_5', 
    'Trust_countrymeasure', 
    'Compliance_1', 
    'Compliance_2', 
    'Compliance_3', 
    'Compliance_4', 
    'Compliance_5', 
    'Compliance_6',
    'Expl_Distress_1', 
    'Expl_Distress_2', 
    'Expl_Distress_3', 
    'Expl_Distress_4', 
    'Expl_Distress_5', 
    'Expl_Distress_6', 
    'Expl_Distress_7', 
    'Expl_Distress_8', 
    'Expl_Distress_9', 
    'Expl_Distress_10', 
    'Expl_Distress_11', 
    'Expl_Distress_12', 
    'Expl_Distress_13', 
    'Expl_Distress_14', 
    'Expl_Distress_15', 
    'Expl_Distress_16', 
    'Expl_Distress_17', 
    'Expl_Distress_18', 
    'Expl_Distress_19', 
    'Expl_Distress_20', 
    'Expl_Distress_21', 
    'Expl_Distress_22', 
    'Expl_Distress_23', 
    'Expl_Distress_24',
    'SPS_1', 
    'SPS_2', 
    'SPS_3', 
    'SPS_4', 
    'SPS_5', 
    'SPS_6', 
    'SPS_7', 
    'SPS_8', 
    'SPS_9', 
    'SPS_10', 
    'Expl_Coping_1', 
    'Expl_Coping_2', 
    'Expl_Coping_3', 
    'Expl_Coping_4', 
    'Expl_Coping_5', 
    'Expl_Coping_6', 
    'Expl_Coping_7', 
    'Expl_Coping_8', 
    'Expl_Coping_9', 
    'Expl_Coping_10', 
    'Expl_Coping_11', 
    'Expl_Coping_12', 
    'Expl_Coping_13', 
    'Expl_Coping_14', 
    'Expl_Coping_15', 
    'Expl_Coping_16',
    'Expl_media_1', 
    'Expl_media_2', 
    'Expl_media_3', 
    'Expl_media_4', 
    'Expl_media_5', 
    'Expl_media_6',
    'PSS10_avg', 
    'SLON3_avg',
    'SPS_avg'
    ]

# dataframe with only relevant columns
df_clean = df_yes[cleaned_columns]


# encoding with category encoders
encoder=ce.OneHotEncoder(cols=['Country','Dem_gender','Dem_edu','Dem_employment','Dem_Expat','Dem_maritalstatus','Dem_riskgroup','Dem_isolation','AD_gain','AD_loss'],handle_unknown='ignore',return_df=True,use_cat_names=True)

#Fit and transform Data
df_enc = encoder.fit_transform(df_clean)


# compute averages of survey reponses and add to dataframe

df_enc['Comp_avg'] = df_clean[[
    'Compliance_1', 
    'Compliance_2', 
    'Compliance_3', 
    'Compliance_4', 
    'Compliance_5', 
    'Compliance_6'
    ]].mean(axis=1).round()

df_enc['Comp_med'] = df_clean[[
    'Compliance_1', 
    'Compliance_2', 
    'Compliance_3', 
    'Compliance_4', 
    'Compliance_5', 
    'Compliance_6'
    ]].median(axis=1)*2

df_enc['Comp_bucket'] = df_enc['Comp_avg']
df_enc.loc[df_enc['Comp_avg'] == 1, 'Comp_bucket'] = 1
df_enc.loc[df_enc['Comp_avg'] == 2, 'Comp_bucket'] = 1
df_enc.loc[df_enc['Comp_avg'] == 3, 'Comp_bucket'] = 1
df_enc.loc[df_enc['Comp_avg'] == 4, 'Comp_bucket'] = 2
df_enc.loc[df_enc['Comp_avg'] == 5, 'Comp_bucket'] = 3
df_enc.loc[df_enc['Comp_avg'] == 6, 'Comp_bucket'] = 3

df_enc['Comp_thresh'] = df_enc['Comp_avg']
df_enc.loc[df_enc['Comp_avg'] == 1, 'Comp_thresh'] = 1
df_enc.loc[df_enc['Comp_avg'] == 2, 'Comp_thresh'] = 1
df_enc.loc[df_enc['Comp_avg'] == 3, 'Comp_thresh'] = 1
df_enc.loc[df_enc['Comp_avg'] == 4, 'Comp_thresh'] = 2
df_enc.loc[df_enc['Comp_avg'] == 5, 'Comp_thresh'] = 2
df_enc.loc[df_enc['Comp_avg'] == 6, 'Comp_thresh'] = 2


df_enc['OECD_avg'] = df_clean[[
    'OECD_people_1', 
    'OECD_people_2', 
    'OECD_insititutions_1', 
    'OECD_insititutions_2', 
    'OECD_insititutions_3', 
    'OECD_insititutions_4', 
    'OECD_insititutions_5', 
    'OECD_insititutions_6'
    ]].mean(axis=1)

df_enc['Concerns_avg'] = df_clean[[
    'Corona_concerns_1', 
    'Corona_concerns_2', 
    'Corona_concerns_3', 
    'Corona_concerns_4', 
    'Corona_concerns_5'
    ]].mean(axis=1)

df_enc['Distress_avg'] = df_clean[[
    'Expl_Distress_1', 
    'Expl_Distress_2', 
    'Expl_Distress_3', 
    'Expl_Distress_4', 
    'Expl_Distress_5', 
    'Expl_Distress_6', 
    'Expl_Distress_7', 
    'Expl_Distress_8', 
    'Expl_Distress_9', 
    'Expl_Distress_10', 
    'Expl_Distress_11', 
    'Expl_Distress_12', 
    'Expl_Distress_13', 
    'Expl_Distress_14', 
    'Expl_Distress_15', 
    'Expl_Distress_16', 
    'Expl_Distress_17', 
    'Expl_Distress_18', 
    'Expl_Distress_19', 
    'Expl_Distress_20', 
    'Expl_Distress_21', 
    'Expl_Distress_22', 
    'Expl_Distress_23', 
    'Expl_Distress_24'
    ]].replace(99, None).mean(axis=1)

df_enc['Coping_avg'] = df_clean[[
    'Expl_Coping_1', 
    'Expl_Coping_2', 
    'Expl_Coping_3', 
    'Expl_Coping_4', 
    'Expl_Coping_5', 
    'Expl_Coping_6', 
    'Expl_Coping_7', 
    'Expl_Coping_8', 
    'Expl_Coping_9', 
    'Expl_Coping_10', 
    'Expl_Coping_11', 
    'Expl_Coping_12', 
    'Expl_Coping_13', 
    'Expl_Coping_14', 
    'Expl_Coping_15', 
    'Expl_Coping_16'
    ]].mean(axis=1)

df_enc['Media_avg'] = df_clean[[
    'Expl_media_1', 
    'Expl_media_2', 
    'Expl_media_3', 
    'Expl_media_4', 
    'Expl_media_5', 
    'Expl_media_6'
    ]].mean(axis=1)




features = [ 
    'Dem_age',
    'Dem_gender_Male',
    'Dem_gender_Other/would rather not say',
    'Dem_gender_Female',
    'Dem_edu_College degree, bachelor, master',
    'Dem_edu_Some College, short continuing education or equivalent',
    'Dem_edu_Up to 12 years of school',
    'Dem_edu_PhD/Doctorate',
    'Dem_edu_Up to 9 years of school',
    'Dem_edu_None',
    'Dem_edu_Up to 6 years of school',
    'Dem_employment_Retired',
    'Dem_employment_Part time employed',
    'Dem_employment_Not employed',
    'Dem_employment_Full time employed',
    'Dem_employment_Self-employed',
    'Dem_employment_Student',
    'Dem_Expat_no',
    'Dem_Expat_yes',
    'Dem_maritalstatus_Married/cohabiting',
    'Dem_maritalstatus_Other or would rather not say',
    'Dem_maritalstatus_Single',
    'Dem_maritalstatus_Divorced/widowed',
    'Dem_dependents',
    'Dem_riskgroup_No',
    'Dem_riskgroup_Yes',
    'Dem_riskgroup_Not sure',
    'Dem_isolation_Life carries on with minor changes',
    'Dem_isolation_Isolated',
    'Dem_isolation_Life carries on as usual',
    'Dem_isolation_Isolated in medical facility of similar location',
    'Dem_isolation_1',
    'Dem_isolation_adults',
    'Dem_isolation_kids',
    'AD_gain_� If Program A is adopted, 200 people will be saved.',
    'AD_gain_� If Program B is adopted, there is 1/3 probability that 600 people will be saved, and 2/3 probability that no people will be saved',  
    'AD_loss_� If Program D is adopted there is 1/3 probability that nobody will die, and 2/3 probability that 600 people will die.',
    'AD_loss_� If Program C is adopted 400 people will die.',
    'Trust_countrymeasure',
    'PSS10_avg', 
    'SLON3_avg',
    'SPS_avg',
    'OECD_avg',
    'Concerns_avg',
    'Distress_avg',
    'Coping_avg',
    'Media_avg'
    ] + [c for c in df_enc.columns if c.startswith('Country_')]


# label classifier variables
# I interchanged the target variable between Comp_thresh, Comp_bucket, and Comp_avg
# Comp_thresh yielded the highest model accuracy so it is hardcoded below

X = df_enc[features]
y = df_enc.Comp_thresh

# print full list of all feature columns
print('\nList of features columns:')
print([c for c in X.columns])

# print occurrences of each class
print('\nClass Frequencies:')
print(y.value_counts())

# pie chart of comp avg frequencies
df_enc.Comp_avg.value_counts().plot.pie(autopct='%1.1f%%',explode=(0.0,0.0,0.0,0.5,0.35,0.1))
plt.savefig('avg_pie.png')
plt.show()



# about 1/3 of the dataframe will be used for testing, the rest for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

print('\nTraining Data Class Frequencies:')
print(y_train.value_counts())

""" 
    The Undersampling and Oversampling sections below were for tuning the
    classification model. Various parameter values were used.
    The parameters seen below work with the target variable being Comp_thresh.
    They are commented out because the final model does not use them.

"""

# undersampling

# rus = RandomUnderSampler(sampling_strategy={2:4000})
# X_train, y_train = rus.fit_resample(X_train,y_train)

# print('\nTraining Data Class Frequencies after Undersampling:')
# print(y_train.value_counts())

#oversampling

# ros = RandomOverSampler(sampling_strategy="not majority")
# X_train, y_train = ros.fit_resample(X_train,y_train)

# print('\nTraining Data Class Frequencies after Oversampling:')
# print(y_train.value_counts())




# create decision tree instance
clf = DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

# predict test data
y_pred  = clf.predict(X_test)

# accuracy for this classifier/prediction
print("\nAccuracy:",round(metrics.accuracy_score(y_test, y_pred),2))




# visualizing the decision tree
dot_data_exp = tree.export_graphviz(clf, out_file = None,
						 feature_names = features,
                         class_names = ['1','2'],  
                         filled = True, rounded = True,  
                         special_characters = True,
                         max_depth = 4)





# visualizing the tree
graph = Source(dot_data_exp)
graph.render('Compliance')
graph.view()




# creating the confusion/error matrix
cm = confusion_matrix(y_test, y_pred)
print('\nError Matrix:')
print(cm)

# # visualizing the error matrix
plt.imshow(cm, cmap = 'binary', interpolation = 'None')

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('error_matrix.png')
plt.show()


# print classification report
print('\nClassification Report\n')
print(classification_report(y_test, y_pred, target_names=['1', '2']))



print('\n-----END OF DATA PROCESSING-----\n')
