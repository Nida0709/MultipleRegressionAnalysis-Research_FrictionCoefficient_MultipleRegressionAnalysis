import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#import seaborn as sns
from tkinter import filedialog

#input Data
target_Path = filedialog.askopenfilename(title='参照先データファイルを指定してください。', filetypes=[('Excel Book', '*.xlsx')])
df = pd.read_excel(target_Path, sheet_name='Sheet1')
t_StableTime = df['StableTime'].values  #Separating from DataFrame to target and input
t_slope = df['slope'].values
t_y_intercept = df['y-intercept'].values
x = df.drop(labels=['Method', 'number', 'StableTime', 'slope', 'y-intercept'], axis=1).values
x_columns = df.drop(labels=['Method', 'number', 'StableTime', 'slope', 'y-intercept'], axis=1).columns.values
                        #print(x_columns) = ['Material' 'Weight' 'SlidingSpeed']

###Have to modificate this system###
#adjust to using form
material_index = [[0.0, 'cnf-fr9'], [10.0, 'cnf-fr10'], [20.0, 'cnf-fr11'],\
     [30.0, 'cnf-fr12'], [40.0, 'cnf-fr15']]
for i in range(len(x)):
    for j in range(len(material_index)):
        if x[i, 0] == material_index[j][1]:
            x[i, 0] = material_index[j][0]
            break
        if j == len(material_index)-1:
            x[i, 0] = 'NotIndexInMaterial'
for i in range(len(x)):
    x[i, 1] = float(x[i, 1][ :-1])
    x[i, 2] = float(x[i, 2][ :-3])

#Split Data
x_train, x_test, t_StableTime_train, t_StableTime_test,t_slope_train, t_slope_test,\
    t_y_intercept_train, t_y_intercept_test = train_test_split(x, t_StableTime, t_slope,\
         t_y_intercept, test_size=0.2, random_state=1) #Randam_state is for fixed　outcomes

#Make model
model_1 = LinearRegression()  #Use linear method
model_2 = LinearRegression()  #Use linear method
model_3 = LinearRegression()  #Use linear method

#Studing
model_1.fit(x_train, t_StableTime_train)
model_2.fit(x_train, t_slope_train)
model_3.fit(x_train, t_y_intercept_train)

#Making Fig
def plot(columns, coef_):
    plt.bar(x=columns, height=coef_)
    plt.show()

plot(x_columns, model_1.coef_)
plot(x_columns, model_2.coef_)
plot(x_columns, model_3.coef_)

print(f'train score(StableTime):{model_1.score(x_train, t_StableTime_train)}')
print(f'test score(StableTime):{model_1.score(x_test, t_StableTime_test)}')
print(f'train score(slope):{model_2.score(x_train, t_slope_train)}')
print(f'test score(slope):{model_2.score(x_test, t_slope_test)}')
print(f'train score(y_intercept):{model_3.score(x_train, t_y_intercept_train)}')
print(f'test score(y_intercept):{model_3.score(x_test, t_y_intercept_test)}')
