# Pract2

Regression Analysis
Least Squares method for fitting a linear relationship (Linear Regression)

import numpy as np

x = [1, 2, 3, 4, 5]
y = [3, 4, 5, 6, 8]

order = 1
f = np.polyfit(x,y,order)
p = np.poly1d(f)

print(p)

 
1.2 x + 1.6

Mean Square Error (MSE)

from sklearn.metrics import mean_squared_error
y = [11,21,19,17.5,10]
y_bar = [12,18,19.5,18,14]

mean_squared_error(y,y_bar)

5.3

Correlation Coefficient

x = ([1,3,4,4])
y = ([2,5,5,8])

z = np.corrcoef(x,y)
print(z)

[[1.        0.8660254]
 [0.8660254 1.       ]]

Rank correlation Coefficient

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Creating dataframe using Pandas
x_sample = pd.DataFrame([(106,7),(100,27),(86,2),(101,50),(99,28),(103,29),
                         (97,20),(113,12), (112,6),(110,17)],
                        columns=["X","Y"])
rank_correlation = x_sample.corr(method="spearman")
print(rank_correlation)

          X         Y
X  1.000000 -0.175758
Y -0.175758  1.000000

#Visualizing the Correlation Coefficient:
fig = plt.figure(figsize = (5,3))
ax = sns.heatmap(rank_correlation, vmin=-1, vmax=1, annot=True)
plt.show()

R-squared and Adjusted R-squared

x_values = [1,2,3,4,5,6]   # Data points
y_values = [1,5,25,30,22,45]
correlation_matrix = np.corrcoef(x_values, y_values)
corelation_x_y = correlation_matrix[0,1]
r_squared = corelation_x_y**2

print("R-Squared ", r_squared)

R-Squared  0.8186273105029377

#Adjusted R^2
adjusted_R2 =  1- (1-r_squared) * (6-1)/(6-1-1)
print("Adjusted R-Squared ", adjusted_R2)

Adjusted R-Squared  0.7732841381286721

Simple Linear Regression

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/income.data.csv")
# r"C:\Users\YourUsername\Downloads\income.data.csv"
df.head()

y = df.happiness
x = df[['income']]

X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=42)

#Visualize the data using scatter plot
plt.figure(figsize =(5,1) )
plt.scatter(x,y, marker = "^", color = 'pink')
plt.title('Happiness Vs Income Plot')
plt.xlabel('Income')
plt.ylabel('Happiness')

The RMSE value:  0.761

The R_Squared value: 71.848 %

Multiple Linear Regression

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("/content/50_Startups.csv")
df.head()

#Data Visualize Using PairPlot

import matplotlib.pyplot as plt
import seaborn as sns
fig = plt.figure(figsize = (5,3))
sns.pairplot(df, kind = "scatter", hue = 'State')
plt.show()

#Create the figure object

plt.figure(figsize = (5,3))
ax = df.groupby(["State"])['Profit'].mean().plot.bar(fontsize = 12)
ax.set_title("Average profit for different states", fontsize =12)
ax.set_xlabel("State")
ax.set_ylabel("Profit")
plt.show()

#Handling Categorical Variables:
df['NewYork_State'] = np.where(df['State']== "New York", 1,0)
df['California_State'] = np.where(df["State"]== "California", 1, 0)
df['Florida_State'] = np.where(df['State'] == "Florida",1,0)

#Drop the original column state from the dataframe
df.drop(columns = ["State"],axis =1, inplace = True)

#Slice the dataset into the predictor variables and target variable
X = df.drop("Profit",axis = 1)
y = df["Profit"]

#Splitting the data traning set and test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=0)

#Applying the mdel
from sklearn.linear_model import LinearRegression

#Creating an object of LinearRegression class
LR = LinearRegression()

#Fitting the traning data

LR.fit(X_train, y_train)


y_pred = LR.predict(X_test)
y_pred

array([103015.20159796, 132582.27760816, 132447.73845174,  71976.09851258,
       178537.48221055, 116161.24230165,  67851.69209676,  98791.73374687,
       113969.43533012, 167921.0656955 ])

#PRedicting the values

pred_df = pd.DataFrame({"Actual Value":y_test, "Predicted Value":y_pred, "Difference":y_test-y_pred})
pred_df.head()

#compare the y_prediction values with the original values
from sklearn.metrics import r2_score
from sklearn.metrics  import mean_squared_error

score = r2_score(y_test, y_pred)
print("r2 Score is: ",score)
print("Mean_Squared_Error is: ", mean_squared_error(y_test, y_pred))
print("Root_Mean_Squared_error is:", np.sqrt(mean_squared_error(y_test,y_pred)))

r2 Score is:  0.9347068473282424
Mean_Squared_Error is:  83502864.03257748
Root_Mean_Squared_error is: 9137.990152794951

#Adjusted R-Squared

Adj_r2 = 1 - (1-score) * (len(y)-1)/(len(y) - X.shape[1] - 1)
print("Adjusted Accuracy: ", round(Adj_r2,3))

Adjusted Accuracy:  0.926

Polynomial Regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("/content/Position_Salaries.csv")
dataset.head()

#Splitting the data into features and labels

X = dataset["Level"]
y = dataset["Salary"]

#Fitting the curve of degree four

order =3
fit = np.polyfit(X,y, order)
poly = np.poly1d(fit)
print(poly)

      3             2
4120 x - 4.855e+04 x + 1.807e+05 x - 1.213e+05

#The polynomial regression results visualization
plt.figure(figsize = (5,3))
model_x = np.linspace(min(X),max(X), 2*len(X))
model_y = poly(model_x)
plt.plot(model_x, model_y, "-*",X,y,"^")
plt.xlabel("Position Level(PL)")
plt.ylabel("Salary of employess")
plt.title("Position Level Vs Salary")
plt.show()

#A new result prediction with polynomial regression

xnew = [[6.5]]
ypred = poly(xnew)
print("The Predicted Value is :", ypred)

The Predicted Value is : [[133259.46969697]]

#Quantative Measure(MSE)

from sklearn.metrics import mean_squared_error
y_true = dataset[['Salary']]
y_pred = poly(dataset[['Level']])
mse_Q1 = mean_squared_error(y_true, y_pred)
print("The MSE value is :", round(mse_Q1, 3))

The MSE value is : 1515662004.662

#R-squared to check the accuracy of the prediction

from sklearn.metrics import r2_score
r_square = r2_score(y_true, y_pred)
print("The R-Sqared value is:", round(r_squared*100,3), "%")

The R-Sqared value is: 81.863 %

#Logistic Regression

import pandas as pd

col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
pima = pd.read_csv('/content/diabetes.csv', header=None, names=col_names)
pima = pima.iloc[1:]
pima.head()

#Selecting Features
#Split dataset in features and target variable

feature_cols = ['pregnant', 'insulin', 'bmi', 'age', 'glucose', 'bp', 'pedigree']
X = pima[feature_cols]
y = pima.label

#Splitting the data

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.25)
scaler = MinMaxScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train),columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test),columns=X_test.columns)


#Model Development and prediction

logreg = LogisticRegression()

#fit the model with data
mod1 = logreg.fit(X_train,y_train)

#test our model
pred1 = logreg.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score (y_true=y_test, y_pred = pred1)

0.7916666666666666

Sampling Techniques
Simple Random Sampling

import numpy as np
import pandas as pd

#Read the data
df=pd.read_csv("/content/Employee_monthly_salary.csv")
df.head()

#Taking 200 units
df.sample(200)

Systematic Sampling

df.iloc[0:1802:10]

Stratified Sampling

df_male = df[df["Gender"]=="M"]
df_male.head()

df_male.sample(200)

Cluster Sampling

import pandas as pd
import numpy as np

#make this example reproducible
np.random.seed(0)

#create DataFrame
df = pd.DataFrame({'tour': np.repeat(np.arange(1,11), 20),
                   'experience': np.random.normal(loc=7, scale=1, size=200)})

df.head()

#randomly choose 4 tour groups out of the 10
clusters = np.random.choice(np.arange(1,11), size=4, replace = False)

#define sample as all members who belong to one of the 4 tour groups
cluster_sample = df[df['tour'].isin(clusters)]

#view first six rows of sample
cluster_sample

clusters #Randomly selected clusters

array([ 1,  6, 10,  8])

#find how many observations came from each tour group
cluster_sample['tour'].value_counts()

1     20
6     20
8     20
10    20
Name: tour, dtype: int64

UnderSampling

#import Libraries
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

df = sns.load_dataset('diamonds')
df.head()

#Visualize the dataset
fig = plt.figure(figsize = (4,4))
sns.countplot(x = df['cut'])
plt.title('Histogram of Cut features')
plt.show()

print(sorted(Counter(df['cut']).items())) #Count unique values

[('Fair', 1610), ('Good', 4906), ('Ideal', 21551), ('Premium', 13791), ('Very Good', 12082)]

X = df         #Define the vairables
Y = df['cut']

X.drop('cut', axis = 1, inplace = True) #Drop 'cut' column from X

#Implementation
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state = 0)
X_resampled, Y_resampled = rus.fit_resample(X,Y)
print(sorted(Counter(Y_resampled).items()),Y_resampled.shape)

[('Fair', 1610), ('Good', 1610), ('Ideal', 1610), ('Premium', 1610), ('Very Good', 1610)] (8050,)

OverSampling

#Import libraries
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

df = sns.load_dataset('diamonds')
df.head()

#Visualize the dataset
fig = plt.figure(figsize = (4,3))
sns.countplot(x = df['cut'])
plt.show()

print(sorted(Counter(df['cut']).items())) #Count unique values

[('Fair', 1610), ('Good', 4906), ('Ideal', 21551), ('Premium', 13791), ('Very Good', 12082)]

X = df         #Define the vairables
Y = df['cut']

#Implementation
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state = 0)
X_resampled, Y_resampled = ros.fit_resample(X, Y)
print(sorted(Counter(Y_resampled).items()), Y_resampled.shape)

[('Fair', 21551), ('Good', 21551), ('Ideal', 21551), ('Premium', 21551), ('Very Good', 21551)] (107755,)


Inferential StatisTics
Statistical Hypothesis

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

Z_Cal = (153.8 - 150) / (2/np.sqrt(4))
Z_Cal

3.8000000000000114

#Area on the left
norm.cdf(-1.96)

0.024997895148220435

#If you dont remember the critical point, then use ppf.
norm.ppf(0.025)

-1.9599639845400545

#Area to the right
norm.sf(1.96)

0.024997895148220435

#Inverse of sf
norm.isf(0.025)

1.9599639845400545

p_value = 2 * norm.sf(3.8)
p_value

0.00014469608785023995

Central Limit Theorem

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Roll a die 1 million times
one_m = np.random.randint(1,7,1000000)

# Visualize the distribution
sns.displot(x = one_m, bins = 6)
plt.show( )

#Mean of the distribution
np.mean(one_m)

3.499392

#Standard deviation of the distribution
np.std(one_m)

1.7075917633720301

#Reshape the one_m distribution
mean4 = one_m.reshape(250000,4).mean(axis = 1)

#Now plot the mean4
sns.displot(x = mean4, bins = 20)
plt.xlim(0,6)
plt.show()

#Find the mean of mean4
np.mean(mean4)

3.499392

#Standard deviation of mean4
np.std(mean4)

0.8547550118811823

One Smaple Z - Test

#Import the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from statsmodels.stats import weightstats

#Read csv file
df = pd.read_csv('Machine1.csv')
df.head()

#Take this mean as a population mean
df.describe()

#Let us visualize the distribution
sns.displot(data = df, x = 'Machine 1')
plt.show()

#Let us try one more, box plot
sns.catplot(data = df, y = 'Machine 1', kind = 'box')
plt.show()

#Now, we want to do one sample Z test
weightstats.ztest(x1 = df['Machine 1'], value = 150, alternative = 'two-sided')

#Using the stats we dont have Z test, we just have t test, using that we get
stats.ttest_1samp(df['Machine 1'], 150)

One sample t test

#Import the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from statsmodels.stats import weightstats

#Define the sample
volume = pd.Series([148.5, 153.4,150.9,151.2])

#Find the summary of the data
volume.describe()

count      4.000000
mean     151.000000
std        2.004994
min      148.500000
25%      150.300000
50%      151.050000
75%      151.750000
max      153.400000
dtype: float64

stats.ttest_1samp(volume, 150)

TtestResult(statistic=0.997509336107632, pvalue=0.3920333832606524, df=3)

One Proportion Test using Python

#Import the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

#Output will be p - value
stats.binom_test(14, 100, p = 0.21, alternative = 'two-sided')

<ipython-input-143-901ffbcaa03c>:2: DeprecationWarning: 'binom_test' is deprecated in favour of 'binomtest' from version 1.7.0 and will be removed in Scipy 1.12.0.
  stats.binom_test(14, 100, p = 0.21, alternative = 'two-sided')

0.10920815720825927

One Variance Test

#Importing required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Chi-square plot
x_ax = np.linspace(0, 100, 101)
y_ax = stats.chi2.pdf(x_ax, df = 50)#df = degrees of freedom
sns.relplot(x = x_ax, y = y_ax, kind = 'line')
plt.show()

#Chi-Square calculated value
chi_sq_cal = (51-1)*(2.35**2)/(2**2)
chi_sq_cal

69.03125000000001

#Critical value of chi-square
stats.chi2.isf(0.1, 50)

63.167121005726315

Two sample Z

#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats import weightstats
import scipy.stats as stats

#Read the data
df = pd.read_csv('Two+Machines.csv')
df.head()

#Statistical Summary
df.describe()

#Visualize the data
sns.catplot(data = df, x = 'Machine', y = 'Volume', kind = 'box')
plt.show()

#Filter that data values
m1 = df[df['Machine'] == 'Machine 1']['Volume']

m2 = df[df['Machine'] == 'Machine 2']['Volume']

#Z clculated
weightstats.ztest(m1, m2)

#Since sample size is smaller, we can use t test as well
stats.ttest_ind(m1, m2, equal_var = True)

Two sample t

#Import libraries
import numpy as np
import scipy.stats as stats

m1 = [150, 152, 154, 152, 151]
m2 = [156, 155, 158, 155, 154]

#t Calculated
statistic, pvalue = stats.ttest_ind(m1, m2, equal_var = False)

#Conclusion
alpha = 0.05
if pvalue > alpha:
  print('No difference')
else:
  print('There is difference')

There is difference

Paired t test

import numpy as np
import pandas as pd
import scipy.stats as stats

#Define
bp_before = [120, 122, 143, 100, 109]
bp_after = [122, 120, 141, 109, 109]

#T calculated value
statistic, pvalue = stats.ttest_rel(bp_before, bp_after)
statistic

-0.6864064729836442

#Conclusion
alpha = 0.05
if pvalue > alpha:
  print('No difference')
else:
  print('There is difference')

No difference

Two Proportions Test

#Follow the same example above:
import statsmodels.stats.proportion as proportion
from statsmodels.stats import proportion

#Calculated values
result = proportion.test_proportions_2indep(30, 200, 10, 100, method = 'score')

pvalue = result[1]
pvalue

0.2305443235633593

 #Conclusion
alpha = 0.1
if pvalue > alpha:
  print('No difference')
else:
  print('There is difference')

No difference

Two Variance Test

#Follow the same example above
import scipy.stats as stats
from scipy.stats import f

#F calculated
F_cal = 11/(1.1**2)
F_cal

9.09090909090909

#Critical value on the right
f.isf(0.05, 4, 7)

4.120311726897633

#Critical value on the left
f.isf(0.95, 4, 7)

0.1640901524729093

####ANOVA test

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.stats.oneway as oneway

m1 = [150, 151, 152, 152, 151, 150]
m2 = [153, 152, 148, 151, 149, 152]
m3 = [156, 154, 155, 156, 157, 155]

#F calculated
statistic, pvalue = stats.f_oneway(m1, m2, m3)
alpha = 0.05

if pvalue > alpha:
  print('No significant difference')
else:
  print('There is significant difference')

There is significant difference

#Or you may follow this method
oneway.anova_oneway((m1, m2, m3), use_var = 'equal')

<class 'statsmodels.stats.base.HolderTuple'>
statistic = 22.264705882352892
pvalue = 3.237408550907782e-05
df = (2.0, 15.0)
df_num = 2.0
df_denom = 15.0
nobs_t = 18.0
n_groups = 3
means = array([151.        , 150.83333333, 155.5       ])
nobs = array([6., 6., 6.])
vars_ = array([0.8       , 3.76666667, 1.1       ])
use_var = 'equal'
welch_correction = True
tuple = (22.264705882352892, 3.237408550907782e-05)

#ANOVA testing using python for some dataset

mpg = sns.load_dataset('mpg') #mpg meaning miles per gallon
mpg.head()

#Using groupby method
mpg.groupby(['origin', 'cylinders'])[['mpg']].mean()

#Lets filter out only having the 4 cylinders
mpg[mpg['cylinders'] == 4]['mpg']

14     24.0
18     27.0
19     26.0
20     25.0
21     24.0
       ... 
393    27.0
394    44.0
395    32.0
396    28.0
397    31.0
Name: mpg, Length: 204, dtype: float64

#Lets perform ANOVA test
EU =  mpg[(mpg['cylinders'] == 4) & (mpg['origin'] == 'europe')]['mpg']
JP =  mpg[(mpg['cylinders'] == 4) & (mpg['origin'] == 'japan')]['mpg']
US =  mpg[(mpg['cylinders'] == 4) & (mpg['origin'] == 'usa')]['mpg']

#F oneway test
stats.f_oneway(EU, JP, US)

F_onewayResult(statistic=9.411845545485601, pvalue=0.00012379894210177303)

#Using statsmodels
oneway.anova_oneway((EU, JP, US), use_var = 'equal')

<class 'statsmodels.stats.base.HolderTuple'>
statistic = 9.411845545485592
pvalue = 0.00012379894210177455
df = (2.0, 201.0)
df_num = 2.0
df_denom = 201.0
nobs_t = 204.0
n_groups = 3
means = array([28.41111111, 31.59565217, 27.84027778])
nobs = array([63., 69., 72.])
vars_ = array([41.50584229, 29.54777494, 20.6984957 ])
use_var = 'equal'
welch_correction = True
tuple = (9.411845545485592, 0.00012379894210177455)

Post Hoc Test(Tukey's HSD Test)

#Import the required libraries
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.stats.oneway as oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

#Load the 'mpg' dataset from seaborn library
mpg = sns.load_dataset('mpg')
mpg.head()

#Perform Tukey's test
result = pairwise_tukeyhsd(endog = mpg['mpg'], groups = mpg['origin'], alpha=0.05)
print(result)

 Multiple Comparison of Means - Tukey HSD, FWER=0.05 
=====================================================
group1 group2 meandiff p-adj   lower    upper  reject
-----------------------------------------------------
europe  japan   2.5592 0.0404   0.0877  5.0307   True
europe    usa  -7.8079    0.0  -9.8448  -5.771   True
 japan    usa -10.3671    0.0 -12.3114 -8.4228   True
-----------------------------------------------------

#Also compare the 'mpg' by visualizing the average mileage
figure = plt.figure(figsize = (3,3))
sns.catplot(data = mpg, x = 'origin', y = 'mpg', kind = 'box')
plt.show()

Goodness of Fit Test(Chi Square)

#Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

#Define expected and observed value
exp = [50, 50]
obs = [40, 60]

#Perform test
statistic, pvalue = stats.chisquare(obs, exp)

#Conclusion
alpha = 0.05
if pvalue >= alpha:
  print('Fail to reject the null hypothesis.')
else:
  print('Reject the null hyptohesis.')

Reject the null hyptohesis.

Colab paid products - Cancel contracts here
Made 1 formatting edit on line 10

