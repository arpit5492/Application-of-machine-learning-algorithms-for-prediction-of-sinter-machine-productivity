#*****************************************************************************
"""
Application of machine learning algorithms for prediction of sinter machine productivity
Python Code By: Dr. Sushant Rath
"""
#******************************************************************************

#IMPORTING LIB
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#import seaborn as sns


#READING DATA
Sample_data=pd.read_csv('SinterProductivityData.csv')  #DATAFRAME

#PRE-PROCESSING DATA
X=Sample_data.drop(["Sample No","Sinter Productivity, T/m2-hr"], axis="columns",inplace=False)
Y=Sample_data["Sinter Productivity, T/m2-hr"]

#SPLITTING DATA & RUNNING MODEL
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.05,random_state=5)
lgr=LinearRegression(fit_intercept=True)
model_lin1=lgr.fit(x_train,y_train)
prediction_lin1=lgr.predict(x_test)
#sns.regplot(x=y_test,y=prediction_lin1,color="green",fit_reg=True,marker="o")
r2_value=model_lin1.score(X,Y)
print("R^2 value=",r2_value)

# PREDICTION OF SINTER PRODUCTVITY OF ALL DATA
newX=X=Sample_data.drop(["Sample No","Sinter Productivity, T/m2-hr"], axis="columns",inplace=False)
predictionY=lgr.predict(newX)
#print(predictionY)

#CALCULATION OF PERFORMANCE PARAMETERS
residual=Y-predictionY
#print(residual)
errorsqr=(Y-predictionY)**2
#print(errorsqr)
meanY=sum(Y)/449
#print(meanY)
DeviationSquareError=(Y-meanY)**2
#print(DeviationSquareError)
AbsoluteError=abs(residual)
#print(AbsoluteError)
AbsolutePercentageError=abs(residual/Y)*100
#print(AbsolutePercentageError)
SumSquareError=sum(errorsqr)
SumSquareTotal=sum(DeviationSquareError)
RootMeanSquareError=(SumSquareError/449)**(0.5)
print("RMSE=",RootMeanSquareError)
MeanAbsoluteError=sum(AbsoluteError)/449
MeanAbsolutePercentageError=sum(AbsolutePercentageError)/449
print("MAE=",MeanAbsoluteError)
print("MAPE=",MeanAbsolutePercentageError)