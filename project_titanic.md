<p style="background-color:#90EE90;color:black;font-size:45px;text-align:center;border-radius:9px 9px;font-weight:bold;border:2px black;">Titanic - SVM and RF Parameters and Deep Learning</p>



Project Planning --

1 The first thing we want to do is an inspection of the data

2 Cleaning the data (NaNs, incorrect values, etc)

3 Transform the data and create dummy variables

4 The plan is to run both a RandomForest and SVM model

5 We want the models to be tuned to their best parameters so we will attempt to write a script to do that

6 We will then see the best feature parameters of both the models to use for deep learning

7 The deep model will be handled with Keras. We want to first explore the different optimizers (We will run a model using each one and see the different accuracies)

8 We will then use the best optimizer to create the final fit using K-fold cross-validation over 10 cycles.




<p style="background-color:#90EE90;color:black;font-size:30px;text-align:center;border-radius:12px 10px;border:2px;"> Here is the whole dataset </p>

The dataset contains ten categories/variables with 891
passenger details. Survived is the response variable for the study. We
are also given a seperate "test" set of data that does not give a
survival result 

<table style="border-collapse: collapse;font-size: 15px; width:800px;">
  <tr>
    <th style="background-color:#D3DBDD;">Variable Name </th>
    <th style="width:500px; background-color:#D3DBDD;">Description</th>
    <th style="background-color:#D3DBDD;">Type</th>
  </tr>
  <tr>
    <td>survival</td>
    <td>Did Survive the incident?</td>
    <td>Categoricol</td>
  </tr>
  <tr>
    <td>pclass </td>
    <td>Class of the ticket</td>
    <td>Categoricol</td>
  </tr>
  <tr>
    <td>sex </td>
    <td>Gender </td>
    <td>Categoricol</td>
  </tr>
  <tr>
    <td>Age </td>
    <td>Age of the passenger</td>
    <td>Numeric</td>
  </tr>
  <tr>
    <td>sibsp </td>
    <td>no of siblings / spouses aboard the Titanic </td>
    <td>Numeric</td>
  </tr>
  <tr>
    <td>parch</td>
    <td>no of parents / children aboard the Titanic</td>
    <td>Numeric</td>
  </tr>
  <tr>
    <td>ticket</td>
    <td>Unique ticket number</td>
    <td>Categoricol</td>
  </tr>
  <tr>
    <td>fare</td>
    <td>Passenger fare </td>
    <td>Numeric</td>
  </tr>
  <tr>
    <td>cabin</td>
    <td>cabin number </td>
    <td>Categoricol</td>
  </tr>
  <tr>
    <td>Embarked</td>
    <td>Port of Embarkation</td>
    <td>Categoricol</td>
  </tr>
</table>



<p style="background-color:#90EE90;color:black;font-size:30px;text-align:center;border-radius:12px 10px;border:2px;"> 1. Data Inspection and Exploration </p>

First we are going to do some simple data exploration,
getting a feel for the different values used.


```py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

gender = pd.read_csv('../input/titanic/gender_submission.csv')

train_data = pd.read_csv('../input/titanic/train.csv')
test_data = pd.read_csv('../input/titanic/test.csv')

train_data['train'] = 1 # here we are defining a column so that we can concatenate the test/train data
test_data['train'] = 0
test_data['Survived'] = np.NaN
all_data = pd.concat([train_data,test_data])

%matplotlib inline
all_data.columns # check the columns
```



    Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
           'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'train'],
          dtype='object')



```py
# checking for dtype, column name, and seeing missing values
train_data.info()
```



    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 13 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   PassengerId  891 non-null    int64  
     1   Survived     891 non-null    int64  
     2   Pclass       891 non-null    int64  
     3   Name         891 non-null    object 
     4   Sex          891 non-null    object 
     5   Age          714 non-null    float64
     6   SibSp        891 non-null    int64  
     7   Parch        891 non-null    int64  
     8   Ticket       891 non-null    object 
     9   Fare         891 non-null    float64
     10  Cabin        204 non-null    object 
     11  Embarked     889 non-null    object 
     12  train        891 non-null    int64  
    dtypes: float64(2), int64(6), object(5)
    memory usage: 90.6+ KB


```py
# getting the descriptive statistics
train_data.describe()
```


``` 
       PassengerId    Survived      Pclass         Age       SibSp  \
count   891.000000  891.000000  891.000000  714.000000  891.000000   
mean    446.000000    0.383838    2.308642   29.699118    0.523008   
std     257.353842    0.486592    0.836071   14.526497    1.102743   
min       1.000000    0.000000    1.000000    0.420000    0.000000   
25%     223.500000    0.000000    2.000000   20.125000    0.000000   
50%     446.000000    0.000000    3.000000   28.000000    0.000000   
75%     668.500000    1.000000    3.000000   38.000000    1.000000   
max     891.000000    1.000000    3.000000   80.000000    8.000000   

            Parch        Fare  train  
count  891.000000  891.000000  891.0  
mean     0.381594   32.204208    1.0  
std      0.806057   49.693429    0.0  
min      0.000000    0.000000    1.0  
25%      0.000000    7.910400    1.0  
50%      0.000000   14.454200    1.0  
75%      0.000000   31.000000    1.0  
max      6.000000  512.329200    1.0  
```

```py
#splitting data into numeric and categorical
dfnum = train_data[['Age','SibSp','Parch','Fare','Survived','Pclass']]
dfcat = train_data[['Sex','Ticket','Cabin','Embarked']]
```

```py
dfnum.describe()
```

``` 
              Age       SibSp       Parch        Fare    Survived      Pclass
count  714.000000  891.000000  891.000000  891.000000  891.000000  891.000000
mean    29.699118    0.523008    0.381594   32.204208    0.383838    2.308642
std     14.526497    1.102743    0.806057   49.693429    0.486592    0.836071
min      0.420000    0.000000    0.000000    0.000000    0.000000    1.000000
25%     20.125000    0.000000    0.000000    7.910400    0.000000    2.000000
50%     28.000000    0.000000    0.000000   14.454200    0.000000    3.000000
75%     38.000000    1.000000    0.000000   31.000000    1.000000    3.000000
max     80.000000    8.000000    6.000000  512.329200    1.000000    3.000000
```


``` python
dfcat.describe()
```

``` 
         Sex  Ticket    Cabin Embarked
count    891     891      204      889
unique     2     681      147        3
top     male  347082  B96 B98        S
freq     577       7        4      644
```


``` python
# here is a heatmap of the correlations between the 4 numeric variables
# as we can see there is a relationship between parch+sibsp (num of parents + num of siblings)
print(dfnum.corr())
sns.heatmap(dfnum.corr())
```



``` 
               Age     SibSp     Parch      Fare  Survived    Pclass
Age       1.000000 -0.308247 -0.189119  0.096067 -0.077221 -0.369226
SibSp    -0.308247  1.000000  0.414838  0.159651 -0.035322  0.083081
Parch    -0.189119  0.414838  1.000000  0.216225  0.081629  0.018443
Fare      0.096067  0.159651  0.216225  1.000000  0.257307 -0.549500
Survived -0.077221 -0.035322  0.081629  0.257307  1.000000 -0.338481
Pclass   -0.369226  0.083081  0.018443 -0.549500 -0.338481  1.000000
```


<img src="titanic_figures\vertopal_a321c76ad9204f40b16c3b1b7fe466bd\99260c2ede08e487d30ef4511e7413d8c27ddf7c.png"/>









```py
pd.options.mode.chained_assignment = None  # default='warn'
dfnum.Age = dfnum.Age.fillna(dfnum.Age.mean())
dfnum.Fare = dfnum.Fare.fillna(dfnum.Fare.median())
```

```py
plt.figure(figsize=[16,12])
from scipy import stats


plt.subplot(231)
sns.histplot(all_data['Age'],kde=True)
plt.title('Age')
plt.xlabel(f'Shapiro test: {round(stats.shapiro(dfnum["Age"]).pvalue,6)}')

plt.subplot(232)
sns.histplot(all_data['SibSp'],kde=True)
plt.title('SibSp')
plt.xlabel(f'Shapiro test: {np.round(stats.shapiro(dfnum["SibSp"]).pvalue,6)}')

plt.subplot(233)
sns.histplot(all_data['Parch'],kde=True)
plt.title('Parch')
plt.xlabel(f'Shapiro test: {np.round(stats.shapiro(dfnum["Parch"]).pvalue,6)}')

plt.subplot(234)
sns.histplot(all_data['Survived'],kde=True)
plt.title('Survived')
plt.xlabel(f'Shapiro test: {np.round(stats.shapiro(dfnum["Survived"]).pvalue,6)}')

plt.subplot(235)
sns.histplot(all_data['Pclass'],kde=True)
plt.title('Pclass')
plt.xlabel(f'Shapiro test: {np.round(stats.shapiro(dfnum["Pclass"]).pvalue,6)}')

```

    Text(0.5, 0, 'Shapiro test: 0.0')



<img src="titanic_figures\vertopal_a321c76ad9204f40b16c3b1b7fe466bd\2ac620dd95d5012b295be59543d41987fab72182.png"/>


``` python
# printing out some stacked histograms and some box plots for age and fare

plt.figure(figsize=[16,12])

# the fare data looks to be fairly skewed and we know from the shapiro test that it's not parametric
plt.subplot(231)
plt.boxplot(x=dfnum['Fare'], showmeans = True, meanline = True)
plt.title('Fare Boxplot')
plt.ylabel('Fare ($)')

# the same can be said for the age data - its looking likely that we will use either median or mode to fill in NA values
plt.subplot(232)
plt.boxplot(dfnum['Age'], showmeans = True, meanline = True)
plt.title('Age Boxplot')
plt.ylabel('Age (Years)')

# just a quick look to see if there is any obvious visual correlation between fare and survival
plt.subplot(233)
plt.hist(x = [dfnum[dfnum['Survived']==1]['Fare'], dfnum[dfnum['Survived']==0]['Fare']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.title('Fare Histogram by Survival')
plt.xlabel('Fare ($)')
plt.ylabel('# of Passengers')
plt.legend()

# and doing the same for age
plt.subplot(234)
plt.hist(x = [dfnum[dfnum['Survived']==1]['Age'], dfnum[dfnum['Survived']==0]['Age']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.title('Age Histogram by Survival')
plt.xlabel('Age (Years)')
plt.ylabel('# of Passengers')
plt.legend()

```



    <matplotlib.legend.Legend at 0x7f77190a6590>


<img src="titanic_figures\vertopal_a321c76ad9204f40b16c3b1b7fe466bd\1237a0cfb86ba227c788a8bcf085e3a7eadfdfbb.png"/>

```py
dfcat.columns
```



    Index(['Sex', 'Ticket', 'Cabin', 'Embarked'], dtype='object')


```py
plt.figure(figsize=[16,12])
from scipy import stats


plt.subplot(231)
sns.barplot(x=dfcat['Sex'].value_counts().index,y=dfcat['Sex'].value_counts()).set(title='Sex')
plt.subplot(232)
sns.barplot(x=dfcat['Ticket'].value_counts().index,y=dfcat['Ticket'].value_counts()).set(title='Ticket')
plt.subplot(233)
sns.barplot(x=dfcat['Cabin'].value_counts().index,y=dfcat['Cabin'].value_counts()).set(title='Cabin')
plt.subplot(234)
sns.barplot(x=dfcat['Embarked'].value_counts().index,y=dfcat['Embarked'].value_counts()).set(title='Embarked')

# now looking at some of the categorical data descriptives
```



    [Text(0.5, 1.0, 'Embarked')]


<img src="titanic_figures\vertopal_a321c76ad9204f40b16c3b1b7fe466bd\677648d8210ee93aa5b3f158fbfd887a208e8571.png"/>


``` python
# noticed that some passengers had multiple rooms, thought it was worth exploring separately
train_data['cabin_multiple'] = train_data.Cabin.apply(lambda x: 0 if pd.isna(x) else len(x.split(' ')))
train_data['cabin_multiple'].value_counts()
```



    0    687
    1    180
    2     16
    3      6
    4      2
    Name: cabin_multiple, dtype: int64


``` python
plt.figure(figsize=[16,12])

plt.subplot(231)
plt.hist(x = [train_data[train_data['Survived']==1]['cabin_multiple'], train_data[train_data['Survived']==0]['cabin_multiple']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.title('cabin_multiple Histogram by Survival')
plt.xlabel('# of Cabins')
plt.ylabel('# of Passengers')
plt.legend()


pd.pivot_table(train_data, index = 'Survived', columns = 'cabin_multiple', values = 'Ticket', aggfunc = 'count')

```

    cabin_multiple      0      1    2    3    4
    Survived                                   
    0               481.0   58.0  7.0  3.0  NaN
    1               206.0  122.0  9.0  3.0  2.0



<img src="titanic_figures\vertopal_a321c76ad9204f40b16c3b1b7fe466bd\4feff5e63f052ec0eb0db91987c1813a0569122e.png"/>


``` python
```

<p style="background-color:#90EE90;color:black;font-size:30px;text-align:center;border-radius:12px 10px;border:2px;"> 2. Data Cleaning and Preprocessing </p>

 Now we are getting to the part that will
differ for most people. We've inspected the data and found some
interesting correlations and nuggets of wisdom. However, choosing which
variables to move forward with is an art not a science and every data
scientist will do something slightly differently.I've decided to move forward with the
following variables:

        'Pclass' --- I've separated the classes into separate variables (P1,P2,P3)

        'Sex' --- The sex variable is likely to inform the model a lot based on the correlations

        'Age' --- Age seems to be an important variable but I will normalise it and clear all NAs

        'SibSp' --- The siblings tell us something about traveling with family which may be helpful

        'norm\_sibsp' --- Here I've normalised the sibling values

        'Parch' --- Likewise the parentage tells us that people are in groups

        'Fare' --- Fare tells us the price of the ticket and is therefore indicative of the wealth of the patron

        'norm\_fare' --- I've included the normalised price of ticket with NAs removed

        'cabin\_multiple' --- If the patron had multiple cabins it tells us something about their demographic

        'numeric\_ticket' --- The ticket number may be informative about the location of the cabin

        'Embarked' --- Perhaps the location they embarked from may relate to their cabin location or wealth



``` python

# split up the cabin multiple string using quick expression
all_data['cabin_multiple'] = all_data.Cabin.apply(lambda x: 0 if pd.isna(x) else len(x.split(' ')))

# split the tickets into number tickets and letter tickets
all_data['numeric_ticket'] = all_data.Ticket.apply(lambda x: 1 if x.isnumeric() else 0) # simply binary if they had numbers
all_data['ticket_letters'] = all_data.Ticket.apply(lambda x: ''.join(x.split(' ')[:-1]).replace('.','').replace('/','').lower() if len(x.split(' ')[:-1]) > 0 else 0)

# extract name titles from the beginning of names
all_data.drop(['Name'],axis=1)
#all_data['name_title'] = all_data.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())

# fill in missing data with median = both age and fare were non-parametric and had very skewed responses so median is more appropriate
all_data.Age = all_data.Age.fillna(train_data.Age.median())
all_data.Fare = all_data.Fare.fillna(train_data.Fare.median())

# remove all the empty values from embarked
all_data.dropna(subset=['Embarked'],inplace = False)

# normalising the sibling and fare data with a log transformation
all_data['norm_sibsp'] = np.log(all_data.SibSp+1)
all_data['norm_fare'] = np.log(all_data.Fare+1)

```

``` python
plt.figure(figsize=[16,12])

sp_norm_data = pd.get_dummies(all_data[['norm_sibsp','SibSp']])
plt.subplot(231)
sns.histplot(data=sp_norm_data,multiple="stack",kde=True,shrink=1.2)

fare_norm_data = pd.get_dummies(all_data[['norm_fare','Fare']])
plt.subplot(232)
sns.histplot(data=fare_norm_data,multiple="stack",kde=True)

sns.histplot()
```



    <AxesSubplot:ylabel='Count'>




<img src="titanic_figures\vertopal_a321c76ad9204f40b16c3b1b7fe466bd\a82903f4f9c7dae66ec75fe027a819b3ceb1ca02.png"/>


``` python
# here we are looking at Pclass vs a selection of factors.
fig, (axis1,axis2,axis3,axis4) = plt.subplots(1,4,figsize=(18,14))

sns.boxplot(x='Pclass',y='Fare',hue='Survived',data=all_data,ax=axis1)
axis1.set_title('Pclass vs Fare Survival Comparison')

sns.barplot(x='Pclass',y='Survived',hue='Sex',data=all_data,ax=axis2,orient = "v")
axis2.set_title('Pclass vs Sex Survival Comparison')

sns.boxplot(x= 'Pclass',y='SibSp',hue = 'Survived',data=all_data,ax=axis3)
axis3.set_title('Pclass vs Siblings Survival Comparison')

sns.violinplot(x='Pclass',y='Age',hue='Survived',data=all_data,split=True,ax=axis4)
axis4.set_title('Pclass vs Age Survival Comparison')
```



    Text(0.5, 1.0, 'Pclass vs Age Survival Comparison')


<img src="titanic_figures\vertopal_a321c76ad9204f40b16c3b1b7fe466bd\ff7f4848a6b73243170df887ec0f5536a05c1bd6.png"/>


``` python
#histogram comparison of sex, class, and age by survival
sca = sns.FacetGrid(all_data, row = 'Sex', col = 'Pclass', hue = 'Survived')
sca.map(plt.hist, 'Age', alpha = .55)
sca.add_legend()
```



    <seaborn.axisgrid.FacetGrid at 0x7f7715b301d0>


<img src="titanic_figures\vertopal_a321c76ad9204f40b16c3b1b7fe466bd\6f2bbb6fa7d6ff2d5155839ce5261312804a34c6.png"/>

``` python
#correlation heatmap of dataset
def corr_hm(df):
    _ , axis1 = plt.subplots(figsize =(14, 12))
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    
    _ = sns.heatmap(
        df.corr(method = 'spearman'), 
        cmap = colormap,
        square=True, 
        cbar_kws={'shrink':.9 }, 
        ax=axis1,
        linewidths=0.1,vmax=1.0, linecolor='grey',
        annot=True,
        annot_kws={'fontsize':14 }
    )
    
    plt.title('Spearman Correlation of Features', y=1.05, size=15)

corr_hm(all_data)
```


<img src="titanic_figures\vertopal_a321c76ad9204f40b16c3b1b7fe466bd\9b9b783ecd22fb78345e4b7bf5683bd077829a58.png"/>


<p style="background-color:#90EE90;color:black;font-size:30px;text-align:center;border-radius:12px 10px;border:2px;"> 3. Transforming the Data and Creating Dummies </p>


``` python
# transforming the class into a string
all_data.Pclass = all_data.Pclass.astype(str)

# create dummies for analysis
all_dummies = pd.get_dummies(all_data[['Pclass','Sex','Age','SibSp','Fare','norm_sibsp','norm_fare','Parch','Embarked',
                                     'cabin_multiple','numeric_ticket','train','Survived']])

# separate the data back into training and testing sets
X_train = all_dummies[all_dummies.train == 1].drop(['train','Survived'], axis = 1)
X_test = all_dummies[all_dummies.train == 0].drop(['train'], axis = 1)

y_train = all_data[all_data.train==1].Survived
y_train.shape
```

    (891,)

``` python
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()

# now we are transforming the values into something scaled using StandardScaler from sklearn
all_dummies_scaled = all_dummies.copy()
all_dummies_scaled[['Age','SibSp','Parch','norm_fare']] = scale.fit_transform(all_dummies_scaled[['Age','SibSp','Parch','norm_fare']])

# we are then separating the training and testing sets out with the scaled data
X_train_scaled = all_dummies_scaled[all_dummies_scaled.train == 1].drop(['train','Survived'], axis = 1)
X_test_scaled = all_dummies_scaled[all_dummies_scaled.train == 0].drop(['train'], axis = 1)

y_train = all_data[all_data.train == 1].Survived
```

<p style="background-color:#90EE90;color:black;font-size:30px;text-align:center;border-radius:12px 10px;border:2px;"> 4. Building the Model </p>

``` python
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
```


Model Tuning For the models we are running a Random Forest Classifier and a Support Vector Machine. I have run a GridSearch with different tuning parameters to identify the best setup to use for these classifiers. For Random Forest that was the following:

``` 
bootstrap: True
 
criterion: gini
 
max_depth: 15
 
max_features: 10
 
min_samples_leaf: 3
 
min_samples_split: 2
 
n_estimators: 550
 
```

``` python
rf = RandomForestClassifier(random_state = 1,bootstrap=True,criterion='gini',max_depth=15,max_features=10,min_samples_leaf=3,min_samples_split=2,
                           n_estimators=550)
best_clf_rf = rf.fit(X_train_scaled,y_train)
```


I performed the same GridSearch for the SVM and found the following best parameters:

``` 
C: 1
 
gamma: 0.1
 
kernel: rbf
 
```

``` python
svc = SVC(probability = True,C=1.0,kernel="rbf",gamma=0.1)
best_clf_svc = svc.fit(X_train_scaled,y_train)
```


Here we are extracting the feature importance for the two models. The plan is to take the combined top10 features of both models to use as our features for the deep learning.

``` python

perm_importance = permutation_importance(best_clf_rf, X_train_scaled, y_train)
feat_importances_rf = pd.Series(perm_importance.importances_mean,index=X_train_scaled.columns)

feat_importances_rf_plt = feat_importances_rf.nlargest(10).plot(kind='barh')
```

<img src="titanic_figures\vertopal_a321c76ad9204f40b16c3b1b7fe466bd\11502fcd1e833ae2feff7914cbfb2c61c8f6f6f8.png"/>


``` python
perm_importance = permutation_importance(best_clf_svc, X_train_scaled, y_train)
feat_importances_svm = pd.Series(perm_importance.importances_mean,index=X_train_scaled.columns)

feat_importances_svm_plt = feat_importances_svm.nlargest(10).plot(kind='barh')
```


<img src="titanic_figures\vertopal_a321c76ad9204f40b16c3b1b7fe466bd\e95fbd7b2bb276ec60252e0cc7f79b4a3f7dec69.png"/>


``` python
# find the common features for deep learning
a=feat_importances_svm.nlargest(10).index
b=feat_importances_rf.nlargest(10).index
df=pd.DataFrame([a,b]).transpose()#

df = df[0].append(df[1]).drop_duplicates('first')

X_train_DL = X_train_scaled[df]
X_test_DL = X_test_scaled[df]

y_train_DL = y_train

dense_neuron = X_train_DL.shape[1]
```

``` 
/opt/conda/lib/python3.7/site-packages/ipykernel_launcher.py:6: FutureWarning: In a future version of pandas all arguments of Series.drop_duplicates will be keyword-only
  
```

I ran the deep learning model looping through the 8 different optimizers to find the version with the highest accuracy/lowest loss
 
<table style="border-collapse: collapse;font-size: 15px; width:800px;">
  <tr>
    <th style="background-color:#D3DBDD;">Optimizer </th>
    <th style="background-color:#D3DBDD;">Loss</th>
    <th style="background-color:#D3DBDD;">Accuracy Percentage</th>
  </tr>
  <tr>
    <td>SGD</td>
    <td>0.125</td>
    <td>83.38%</td>
  </tr>
  <tr>
      <td><b>RMSprop</b> </td>
    <td><b>0.0976</b></td>
    <td><b>87.54%</b></td>
  </tr>
  <tr>
    <td>Adam </td>
    <td>0.108 </td>
    <td>86.083%</td>
  </tr>
  <tr>
    <td>Adadelta </td>
    <td>0.264</td>
    <td>37.59%</td>
  </tr>
  <tr>
    <td>Adagrad </td>
    <td>0.133 </td>
    <td>82.37%</td>
  </tr>
  <tr>
    <td>Adamax</td>
    <td>0.110</td>
    <td>85.29%</td>
  </tr>
  <tr>
    <td>Nadam</td>
    <td>0.106</td>
    <td>86.08%</td>
  </tr>
  <tr>
    <td>Ftrl</td>
    <td>0.249 </td>
    <td>61.61%</td>
  </tr>
</table> As we can see, RMRprop gave the best results, so we will use that optimizer moving forward

``` python
# separating the training data into train/test splits for use in our model fitting
XDL_train,XDL_test,yDL_train,yDL_test=train_test_split(X_train_DL,y_train,test_size=0.2,random_state=1234)
```



``` python
# here we are defining some parameters to use in our model

from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import KFold
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tqdm.notebook import tqdm
from sklearn.utils import shuffle
import tensorflow.keras.backend as K

COLS = list(XDL_train.columns) # defining the columns

batch_size=128
epoch = 500
mae = pd.DataFrame
kFold = KFold(n_splits=10,
              shuffle=True) # defining the number of cross-validation kfolds we want to perform

inputs = np.concatenate((XDL_train,
                         XDL_test),
                        axis=0) # combining the test/train data

targets = np.concatenate((yDL_train,
                          yDL_test),
                         axis=0)
```

Brace yourself, this is a wall of text. I could have subdivided this into several functions but it all works and runs well. Perhaps I can clean it up a bit another time. But for now lets consider what this model is achieving:

  First we are defining a table that will output our mean absolute error scores.

  Next we are collecting the results into lists that we can use for printing.

  The model then enters into an outer for loop that will loop over the number of folds that we want to run (defined above).

  Within this loop we will re-define the model on every run, compile the model using our predefined standards and fit the model.

  Before we fit, we are creating several callbacks that can be used during the run:
    
           1. A callback that will stop the current run if certain amount of progression has passed without improvement.
    
           2. A callback that saves the best weights based on improvements across each epoch.

  The data is split manually based on the kFolds function defined above. Each iteration of the outer loop will use a different subsection of the data for test and validation.

  After the data are fit to the model, we want to extract the feature importance much like we have done above.

  For this to work first we must define a baseline score, here calculated by defining the out of function predictions based on all the valid data and then taking the absolute mean of that score divided by the valid list.

  After this we enter into an inner loop which does the same thing for each feature but shuffles the column each time.

  The last part of the function is just dealing with the outputs 

``` python
def DLmodel(inputs,targets,columns,batch_size,epochs,kFold):
    results_table = {}
    acc_per_fold = [] # variable to collect the accuracy for printing and saving
    loss_per_fold = [] # variable to collect the loss for printing and saving
    i = 0
    fold_no = 1
    for fold,(train_idx,test_idx) in enumerate(kFold.split(inputs,targets)):
    
        K.clear_session()
        
        print('-'*15, '>', f'Fold {fold+1}', '<', '-'*15)
        X_train, X_valid = inputs[train_idx], inputs[test_idx]
        y_train, y_valid = targets[train_idx], targets[test_idx]
        checkpoint_filepath = f"folds{fold}.hdf5"
    
        #################################################################
        # creating the keras model
        model = keras.Sequential([
            layers.BatchNormalization(),
            layers.Dense(32,
                         activation = 'relu',
                         input_shape=([dense_neuron])),
            layers.Dropout(0.3),
            layers.BatchNormalization(),
            layers.Dense(32,
                         activation = 'relu'),
            layers.BatchNormalization(),
            layers.Dense(32,
                         activation = 'relu'),
            layers.Dropout(0.3),
            layers.Dense(1,
                         activation = 'sigmoid')
        ])

        model.compile(
            optimizer = 'RMSprop',
            loss = 'mae',
            metrics=['accuracy']
    
        )
        #################################################################
        # Generate a print
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')
    
        #################################################################
        # create the early stopping and saving functionality
        early_stopping = keras.callbacks.EarlyStopping(
            patience = 100,
            min_delta = 0.001,
            restore_best_weights=True,
        )
        save_best = keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=False,
            monitor='val_accuracy',
            mode='max',
            verbose=0,
            save_best_only=True)

        # fit the model
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_valid,
                             y_valid),    
            batch_size=batch_size,
            epochs = epochs,
            callbacks=[early_stopping,
                       save_best],
            verbose = False,
        )
        scores = model.evaluate(X_valid,
                                y_valid,
                                verbose=0)
        print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]};{model.metrics_names[1]} of {scores[1]*100}%')
        print(f'optimizer: RMSprop')
        acc_per_fold.append(scores[1]*100)
        loss_per_fold.append(scores[0])   

        #################################################################
        # Increase fold number

        
        print('Calculating feature importance...')
            
        # compute baseline (no shuffle)
        oof_preds = model.predict(X_valid,
                                  verbose=0).squeeze() 
        baseline_mae = np.mean(np.abs(oof_preds-y_valid ))
    
        for col in columns:
            if not col in results_table.keys():
                results_table[col] = []
                results_table['baseline'] = []
       
        results_table['baseline'].append(baseline_mae)
    
        ################################################################# 
        # add col names before loops
    

        # defining a function to take care of the inner loop calculations
        for k in tqdm(range(len(columns))):
            # shuffle feature 
            save_col = X_valid[:,k].copy()
            np.random.shuffle(X_valid[:,k])
            col_name = columns[k]
            # computer oof mae with feature k shuffled
            oof_preds = model.predict(X_valid,
                                        verbose=0).squeeze()

            results_table[col_name].append(np.mean(np.abs(oof_preds-y_valid)))
            X_valid[:,k] = save_col
    
        
    
        #################################################################
        # display feature importance
        
        df = pd.DataFrame.from_dict(results_table)
        df = df.iloc[fold].sort_values(ascending=True)
        plt.figure(figsize=(5,6))
        plt.barh(np.arange(len(columns)+1),
                 df.values)
        plt.yticks(np.arange(len(columns)+1),
                   df.index)
        plt.title(f'K-Fold {fold+1} Feature Importance',
                  size=16)
        plt.ylim((-1,len(columns)+1))
        plt.plot([baseline_mae,
                  baseline_mae],
                 [-1,len(columns)+1],
                 '--', color='red',
                 label=f'Baseline OOF\nMAE={baseline_mae:.3f}')
        plt.xlabel(f'Fold {fold+1} OOF MAE with feature permuted',
                   size=14)
        plt.ylabel('Feature',
                   size=14)
        plt.legend()
        plt.show()
                               
        # SAVE FEATURE IMPORTANCE
        df.to_csv(f'feature_importance_fold_{fold+1}.csv',
                  index=False)
    
        fold_no = fold_no + 1
        i = i+1
    
    #################################################################

    # == Provide average scores ==
    print('------------------------------------------------------------------------')
    print('Score per fold')
    for i in range(0, len(acc_per_fold)):
      print('------------------------------------------------------------------------')
      print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
    print(f'> Loss: {np.mean(loss_per_fold)}')
    print('------------------------------------------------------------------------')

    history_df = pd.DataFrame(history.history)
    history_df.loc[:, ['loss', 'val_loss','accuracy','val_accuracy']].plot();
    print("Minimum validation loss: {}".format(history_df['val_loss'].min()))
    pd.DataFrame.from_dict(results_table)
    
    return history, results_table, save_best
```

``` python
history,results_table, save_best = DLmodel(inputs,targets,COLS,batch_size,epoch,kFold)
```
For brevity we will skip to the last kFold

    --------------- > Fold 10 < ---------------
    ------------------------------------------------------------------------
    Training for fold 10 ...
    Score for fold 10: loss of 0.1427287459373474;accuracy of 86.51685118675232%
    optimizer: RMSprop
    Calculating feature importance...


<img src="titanic_figures\vertopal_a321c76ad9204f40b16c3b1b7fe466bd\205a78a59a87feb20d4b6bbec345c461dd3ca0fa.png"/>


    ------------------------------------------------------------------------
    Score per fold
    ------------------------------------------------------------------------
    > Fold 1 - Loss: 0.1486242264509201 - Accuracy: 85.5555534362793%
    ------------------------------------------------------------------------
    > Fold 2 - Loss: 0.15655222535133362 - Accuracy: 85.39325594902039%
    ------------------------------------------------------------------------
    > Fold 3 - Loss: 0.2166995406150818 - Accuracy: 78.65168452262878%
    ------------------------------------------------------------------------
    > Fold 4 - Loss: 0.09792616963386536 - Accuracy: 91.01123809814453%
    ------------------------------------------------------------------------
    > Fold 5 - Loss: 0.173434317111969 - Accuracy: 83.14606547355652%
    ------------------------------------------------------------------------
    > Fold 6 - Loss: 0.13285966217517853 - Accuracy: 87.64045238494873%
    ------------------------------------------------------------------------
    > Fold 7 - Loss: 0.11041408777236938 - Accuracy: 88.76404762268066%
    ------------------------------------------------------------------------
    > Fold 8 - Loss: 0.14105932414531708 - Accuracy: 86.51685118675232%
    ------------------------------------------------------------------------
    > Fold 9 - Loss: 0.2530847489833832 - Accuracy: 74.15730357170105%
    ------------------------------------------------------------------------
    > Fold 10 - Loss: 0.1427287459373474 - Accuracy: 86.51685118675232%
    ------------------------------------------------------------------------
    Average scores for all folds:
    > Accuracy: 84.73533034324646 (+- 4.721670677586862)
    > Loss: 0.15733830481767655
    ------------------------------------------------------------------------
    Minimum validation loss: 0.1427287608385086


<img src="titanic_figures\vertopal_a321c76ad9204f40b16c3b1b7fe466bd\e88a4adb39700b2d9a73f13f8509c9703f1760af.png"/>



As we can see from the output, the model is performing pretty well (between 80-90% accuracy). The plan now, which is totally not necessary and is purely for exploration at this point. Is to take the overall best features across all of the folds and take the best 5 and apply it to one more run. Just to see if using less features that are the most informative is able to squeeze out a little more accuracy. Obviously we want to be careful as well to keep an eye on our loss/val\_loss scores to make sure we aren't overfitting/underfitting.

``` python
from matplotlib.pyplot import figure

param_data = pd.DataFrame.from_dict(results_table)
mean_param_data = {}
for i in param_data:
    mean_param_data[i] = []
    mean_param_data[i].append(param_data[i].mean())
figure(figsize=(16, 12), dpi=80)
plt.scatter(*zip(*sorted(mean_param_data.items()))) 
plt.show()
mean_param_data = pd.Series(mean_param_data)
```



<img src="titanic_figures\vertopal_a321c76ad9204f40b16c3b1b7fe466bd\0518421a98388281d184835391b5d8bcd700fa29.png"/>



Here is a simple scatter plot detailing the average feature importance across all the folds. We are going to take the best 5 and use it for the final model.


``` python
top5_params = mean_param_data.sort_values(ascending = False)[0:5]
top5COLS = list(top5_params.index)

inputs = np.concatenate((XDL_train[top5COLS],
                         XDL_test[top5COLS]),
                        axis=0) # combining the test/train data

targets = np.concatenate((yDL_train,
                          yDL_test),
                         axis=0)
```

``` python
best_params,results_table_params, save_best_params = DLmodel(inputs,targets,top5COLS,batch_size,epoch,kFold)
```

    --------------- > Fold 10 < ---------------
    ------------------------------------------------------------------------
    Training for fold 10 ...
    Score for fold 10: loss of 0.22170962393283844;accuracy of 77.52808928489685%
    optimizer: RMSprop
    Calculating feature importance...


<img src="titanic_figures\vertopal_a321c76ad9204f40b16c3b1b7fe466bd\b00c4ebd0214aad5a639dcc236f2f8b873fdaa2c.png"/>



    ------------------------------------------------------------------------
    Score per fold
    ------------------------------------------------------------------------
    > Fold 1 - Loss: 0.1779356151819229 - Accuracy: 82.22222328186035%
    ------------------------------------------------------------------------
    > Fold 2 - Loss: 0.18038129806518555 - Accuracy: 82.02247023582458%
    ------------------------------------------------------------------------
    > Fold 3 - Loss: 0.07055552303791046 - Accuracy: 93.2584285736084%
    ------------------------------------------------------------------------
    > Fold 4 - Loss: 0.20372572541236877 - Accuracy: 79.77527976036072%
    ------------------------------------------------------------------------
    > Fold 5 - Loss: 0.1693388819694519 - Accuracy: 83.14606547355652%
    ------------------------------------------------------------------------
    > Fold 6 - Loss: 0.20285668969154358 - Accuracy: 79.77527976036072%
    ------------------------------------------------------------------------
    > Fold 7 - Loss: 0.1771462857723236 - Accuracy: 82.02247023582458%
    ------------------------------------------------------------------------
    > Fold 8 - Loss: 0.11299433559179306 - Accuracy: 88.76404762268066%
    ------------------------------------------------------------------------
    > Fold 9 - Loss: 0.13493190705776215 - Accuracy: 86.51685118675232%
    ------------------------------------------------------------------------
    > Fold 10 - Loss: 0.22170962393283844 - Accuracy: 77.52808928489685%
    ------------------------------------------------------------------------
    Average scores for all folds:
    > Accuracy: 83.50312054157257 (+- 4.489692528671384)
    > Loss: 0.16515758857131005
    ------------------------------------------------------------------------
    Minimum validation loss: 0.22170962393283844




<img src="titanic_figures\vertopal_a321c76ad9204f40b16c3b1b7fe466bd\e11984fcc485c4cc4a04a856571f2454149a78b7.png"/>



Accuracy seems to be staying put, still around 80-90%. Which is interesting in of itself. Even without the other variables we are still able to predict survival likelihood to a very high percentage.

``` python
Xfinal_train = X_train_DL[top5_params.index] # creating the testing set with the finalised features
Xfinal_test = X_test_DL[top5_params.index]
best_final = {}
for i in range(kFold.n_splits): 
    # this loop extracts the accuracy and loss from each of our folds evaluated on the whole.
    # training set rather than using out of sample kFolds. We will then save this into a library and find the very best weights to use.
    # this isn't strictly best practice, but like I said earlier, this is more just for exploration and seeing what we can squeeze out.
    save_best_params.model.load_weights(f'folds{i}.hdf5')
    best_final[f'folds{i}.hdf5'] = []
    best_final[f'folds{i}.hdf5'].append(save_best_params.model.evaluate(Xfinal_train,y_train_DL,verbose=True)[1])
```

    28/28 [==============================] - 0s 2ms/step - loss: 0.1806 - accuracy: 0.8305
    28/28 [==============================] - 0s 3ms/step - loss: 0.1664 - accuracy: 0.8350
    28/28 [==============================] - 0s 2ms/step - loss: 0.1645 - accuracy: 0.8361
    28/28 [==============================] - 0s 2ms/step - loss: 0.3249 - accuracy: 0.8092
    28/28 [==============================] - 0s 2ms/step - loss: 0.3400 - accuracy: 0.7834
    28/28 [==============================] - 0s 2ms/step - loss: 0.1712 - accuracy: 0.8316
    28/28 [==============================] - 0s 2ms/step - loss: 0.3995 - accuracy: 0.7643
    28/28 [==============================] - 0s 2ms/step - loss: 0.1710 - accuracy: 0.8305
    28/28 [==============================] - 0s 2ms/step - loss: 0.1659 - accuracy: 0.8339
    28/28 [==============================] - 0s 2ms/step - loss: 0.1805 - accuracy: 0.8260


``` python
```


``` python
save_best_params.model.load_weights(max(best_final))
save_best_params.model.evaluate(Xfinal_train,y_train_DL,verbose=True)
```


    28/28 [==============================] - 0s 2ms/step - loss: 0.1805 - accuracy: 0.8260




    [0.1804504543542862, 0.8260381817817688]




<p style="background-color:#90EE90;color:black;font-size:30px;text-align:center;border-radius:12px 10px;border:2px;"> 4. Model Prediction of Survival </p>

So here we are at the moment of truth\! We are going to use the testing data with our modified feature inputs to predict survival and submit to the competition\!

``` python
from keras import models

print('Generate a prediction')
prediction = np.round(save_best_params.model.predict(Xfinal_test)).astype(int)

print('prediction shape:',prediction.shape)
```


    Generate a prediction
    prediction shape: (418, 1)


``` python
final_data = {'PassengerId':test_data.PassengerId,
             'Survived':prediction.transpose(1,0)[0,:]}

submission = pd.DataFrame(final_data)
```


``` python
submission.to_csv('submission.csv',index=False)
```


<p style="background-color:#90EE90;color:black;font-size:30px;text-align:center;border-radius:12px 10px;border:2px;"> 4. Leaderboard Score </p>

<img src="titanic_figures\figure1.png"/>

<img src="titanic_figures\figure2.png"/>

We managed a prediction of 78% and got into the top 15% of submissions for this competition!