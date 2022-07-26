#Simple Linear Regression for calories data set
#importing the packages
import pandas as pd   
import numpy as np 
import matplotlib.pyplot as plt

#Loading the data set into python
calories_data = pd.read_csv(r"C:\Users\kaval\OneDrive\Desktop\Assignments\Assignment Datasets\Simple Linear Regression\Assignment Datasets\calories_consumed.csv")
calories_data.columns = "Weightgained", "CaloriesConsumed" ##renaming so that no sapces is there otherwise error
calories_data.describe()
calories_data.info()

#Visualization of barplot, box plot and histograms

plt.bar(height=calories_data.Weightgained, x = np.arange(1, 15, 1))
plt.hist(calories_data.Weightgained)
plt.boxplot(calories_data.Weightgained)

plt.bar(height=calories_data.CaloriesConsumed, x = np.arange(1, 15, 1))
plt.hist(calories_data.CaloriesConsumed)
plt.boxplot(calories_data.CaloriesConsumed)

#Scatter Plot
plt.scatter(x = calories_data['Weightgained'], y = calories_data['CaloriesConsumed'], color = 'green') 

# correlation
np.corrcoef(calories_data.Weightgained, calories_data.CaloriesConsumed) 
#Covarience
cov_output = np.cov(calories_data.Weightgained, calories_data.CaloriesConsumed)[0, 1]
cov_output

#Simple linear regression
import statsmodels.formula.api as smf
#Generating the model
model = smf.ols('CaloriesConsumed ~ Weightgained', data = calories_data).fit()
model.summary()

prediction = model.predict(pd.DataFrame(calories_data['Weightgained']))

# Scatter plot and the regression line
plt.scatter(calories_data.Weightgained, calories_data.CaloriesConsumed)
plt.plot(calories_data.Weightgained, prediction, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Calculating the error
result = calories_data.CaloriesConsumed - prediction
res_sqr = result * result
mse = np.mean(res_sqr)
RMSE = np.sqrt(mse)
RMSE

#Applying transformation techniques to prune the model
#Applying logarthemic transformation
plt.scatter(x = np.log(calories_data['Weightgained']), y = calories_data['CaloriesConsumed'], color = 'brown')
#Correlation
np.corrcoef(np.log(calories_data.Weightgained), calories_data.CaloriesConsumed) 

model2 = smf.ols('CaloriesConsumed ~ np.log(Weightgained)', data = calories_data).fit()
model2.summary()

prediction2 = model2.predict(pd.DataFrame(calories_data['Weightgained']))

# Scatter plot and the regression line
plt.scatter(np.log(calories_data.Weightgained), calories_data.CaloriesConsumed)
plt.plot(np.log(calories_data.Weightgained), prediction2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Calculating the error
result2 = calories_data.CaloriesConsumed - prediction2
res_sqr2 = result2 * result2
mse2 = np.mean(res_sqr2)
RMSE2 = np.sqrt(mse2)
RMSE2

#Applying Exponentian Technique
plt.scatter(x = calories_data['Weightgained'], y = np.log(calories_data['CaloriesConsumed']), color = 'orange')
np.corrcoef(calories_data.Weightgained, np.log(calories_data.CaloriesConsumed))

model3 = smf.ols('np.log(CaloriesConsumed) ~ Weightgained', data = calories_data).fit()
model3.summary()

prediction3 = model3.predict(pd.DataFrame(calories_data['Weightgained']))
pred3_at = np.exp(prediction3)
pred3_at

# Scatter plot and the regression line
plt.scatter(calories_data.Weightgained, np.log(calories_data.CaloriesConsumed))
plt.plot(calories_data.Weightgained, prediction3, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

#Calculating the error
result3 = calories_data.CaloriesConsumed - pred3_at
res_sqr3 = result3 * result3
mse3 = np.mean(res_sqr3)
RMSE3 = np.sqrt(mse3)
RMSE3

#Applying polynomial transformation technique
model4 = smf.ols('np.log(CaloriesConsumed) ~ Weightgained + I(Weightgained*Weightgained)', data = calories_data).fit()
model4.summary()

prediction4 = model4.predict(pd.DataFrame(calories_data))
pred4_at = np.exp(prediction4)
pred4_at

#Scatter plot and the regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X = calories_data.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)



plt.scatter(calories_data.Weightgained, np.log(calories_data.CaloriesConsumed))
plt.plot(X, prediction4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()


# Calculating the error
result4 = calories_data.CaloriesConsumed - pred4_at
res_sqr4 = result4 * result4
mse4 = np.mean(res_sqr4)
RMSE4 = np.sqrt(mse4)
RMSE4


# Choosing the best model using RMSE
data = {"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), "RMSE":pd.Series([RMSE, RMSE2, RMSE3, RMSE4])}
table_rmse = pd.DataFrame(data)
table_rmse

#Selecting the best model for the final modeling of the data set
from sklearn.model_selection import train_test_split

train, test = train_test_split(calories_data, test_size = 0.2)

finalmodel = smf.ols('np.log(CaloriesConsumed) ~ Weightgained + I(Weightgained*Weightgained)', data = train).fit()
finalmodel.summary()

# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))
pred_test_CaloriesConsumed = np.exp(test_pred)
pred_test_CaloriesConsumed

# Model Evaluation on Test data
test_res = test.CaloriesConsumed - pred_test_CaloriesConsumed
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse


# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))
pred_train_CaloriesConsumed = np.exp(train_pred)
pred_train_CaloriesConsumed

# Model Evaluation on train data
train_res = train.CaloriesConsumed - pred_train_CaloriesConsumed
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse

# In this dataset the transformation  is not required
 
"""
      MODEL        RMSE
0         SLR  232.833501
1   Log model  253.558040
2   Exp model  272.420712
3  Poly model  240.827776
 
"""