# coding=utf-8
# 2018-08-02


 

import numpy as np
import pandas as pd
from sklearn import cross_validation

# from sklearn.model_section import model_section
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import grid


def AllRegressionResult ():
    data = pd.read_csv(grid.RESULT_FILE)
    # data = data.query('day >= 20170213')

    dfArray  = []

    
    for gridid in range(51, 101):

        df = ridgeRegression2 (data, gridid)
        # df = ridgeRegression3 (data, gridid)
        dfArray.append(df)
    
     
    dfAll = pd.concat(dfArray)

    dfAll['grid_id'] = dfAll['grid_id'].astype('int')
    dfAll['day'] = dfAll['day'].astype('int')
    dfAll['hour'] = dfAll['hour'].astype('int')

    dfAll = dfAll.sort_values(by=['day', 'hour', 'grid_id'], axis=0, ascending=True)

    dfAll.to_csv(grid.DATA_PATH + 'result_22.csv', index=False)


def showresult():

    grid_id = 92
    # data3 = pd.read_csv(grid.DATA_PATH + 'result_5.csv').query('grid_id == @grid_id')

 

    # print(data)

    data1 = pd.read_csv(grid.DATA_PATH + 'result_b_20180924_mean.csv').query('grid_id == @grid_id')
    data2 = pd.read_csv(grid.DATA_PATH + 'result_b_20180924_ridge.csv').query('grid_id == @grid_id')
    data3 = pd.read_csv(grid.DATA_PATH + 'result_b_20180926_ridge0.1.csv').query('grid_id == @grid_id')

    # data4 = pd.read_csv(grid.DATA_PATH + 'result_20180923_weightedmean.csv').query('grid_id == @grid_id')
 

    plt.plot(data1['car_number'], 'y',  label = '1 mean')
    plt.plot(data2['car_number'], 'b',  label = '2 ridge')
    plt.plot(data3['car_number'], 'r',  label = '3 ridge 2')
    # plt.plot(data4['car_number'], 'g',  label = '4 weighted mean')
    plt.legend(loc = 'upper left')
    plt.show()

 

def ridgeRegression (inDF, iGridID):

    dataGrid = inDF.query('grid_id  == @iGridID')
    a = np.array(dataGrid)

    x = a[:, 2:6]
    y = a[:, 6]
    poly = PolynomialFeatures(6)
    x = poly.fit_transform(x)

    x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size = 0.2,random_state = 0)
    clf = Ridge(alpha = 1.0, fit_intercept = True)
    aa = clf.fit(x_train, y_train)

    print(iGridID, 'score',clf.score(x_test, y_test))

    y_pre = clf.predict(x)

    dfOut = pd.DataFrame(columns = ["grid_id", "day", "hour", "car_number"])

    # 取后14天
    y_result = y_pre[-14 * 24:]
    allIndex  = 0
    resultIndex = 0
 
    for day in range(14):      

        iCurrentDay =   20170313 + day

        for hour in range(24):

            carNum = y_result[allIndex]

            if hour >= 9 and hour <= 22 :

                if carNum < 0:
                    carNum = 0

                dfOut.loc[resultIndex] = {'grid_id': iGridID, 'day': iCurrentDay, 'hour': hour, 'car_number':carNum}
                resultIndex += 1

            allIndex+=1
    
    # print(dfOut)

    # print ('ridgeRegression gid done', iGridID)
    return dfOut



def ridgeRegression2 (inDF, iGridID):

    # 默认6
    DRGREE = 8

    dataGrid = inDF.query('grid_id  == @iGridID')


    a = np.array(dataGrid)


    # dayno,week,weekday,hour
    x = a[:, 2:6]

    #car_number 
    y = a[:, 6]

    poly = PolynomialFeatures(DRGREE)
    x = poly.fit_transform(x)


    x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size = 0.2,random_state = 0)
    clf = Ridge(alpha = 1.0, fit_intercept = True, normalize= False)
    clf.fit(x_train, y_train)
 

    # print(clf.coef_, aa)
    print('grid= %d, score = %f' % (iGridID, clf.score(x_test, y_test) ))    
    y_pre = clf.predict(x)

    dfOut = pd.DataFrame(columns = ["grid_id", "day", "hour", "car_number"])

    # 倒数第1周，和倒数第3周
    y_result =np.append(y_pre[-7 * 24:],y_pre[-3* 7 * 24: -2 * 7 * 24])
    # print(len(y_result))

    allIndex  = 0
    resultIndex = 0
 
    for day in range(14):      

        iCurrentDay =   20170313 + day

        for hour in range(24):

            carNum = y_result[allIndex]

            if hour >= 9 and hour <= 22 :

                if carNum < 0:
                    carNum = 0

                dfOut.loc[resultIndex] = {'grid_id': iGridID, 'day': iCurrentDay, 'hour': hour, 'car_number':carNum}
                resultIndex += 1

            allIndex+=1

    return dfOut


# 直线型 领回归
def ridgeRegression3(inDF, iGridID):
    # 默认6
    DRGREE = 1

    dataGrid = inDF.query('grid_id  == @iGridID')

    a = np.array(dataGrid)

    # dayno,week,weekday,hour
    x = a[:, 2:6]

    # car_number
    y = a[:, 6]

    poly = PolynomialFeatures(DRGREE)
    x = poly.fit_transform(x)

    x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.2, random_state=0)
    clf = Ridge(alpha=0.1, fit_intercept=True, normalize=False)
    clf.fit(x_train, y_train)

    # print(clf.coef_, aa)
    print('grid= %d, score = %f' % (iGridID, clf.score(x_test, y_test)))



    x_result = np.zeros((24 * 14, 4))

    index = 0
    for i in range(14):
        day = i + 70
        week = day // 7
        dayofweek = day % 7

        for j in range(24):
            hour = j

            x_result[index][0] = day
            x_result[index][1] = week
            x_result[index][2] = dayofweek
            x_result[index][3] = hour

            index += 1

    poly2 = PolynomialFeatures(DRGREE)
    x_result_trans = poly2.fit_transform(x_result)


    y_result = clf.predict(x_result_trans)


    dfOut = pd.DataFrame(columns=["grid_id", "day", "hour", "car_number"])


    allIndex = 0
    resultIndex = 0

    for day in range(14):

        iCurrentDay = 20170313 + day

        for hour in range(24):

            carNum = y_result[allIndex]

            if hour >= 9 and hour <= 22:

                if carNum < 0:
                    carNum = 0

                dfOut.loc[resultIndex] = {'grid_id': iGridID, 'day': iCurrentDay, 'hour': hour, 'car_number': carNum}
                resultIndex += 1

            allIndex += 1

    return dfOut



def validAll():

    dataAll = pd.read_csv(grid.RESULT_FILE)
    # dataAll = dataAll.query('day >= 20170206')
    sum = 0
    len = 0

    for i in range(51, 101):
        sum1, len1 = ridgeRegressionGrid(dataAll, i)

        print(i)

        sum += sum1
        len += len1

    se = (sum / len) ** 0.5
    print('se = %f' % se)




 
def ridgeRegressionGrid(dataAll, gridid):

    # 默认6
    DRGREE = 8

    data = dataAll.query('grid_id  == @gridid')

    dataGrid = data
    # dataGrid = data.query('(day < 20170227  & day >= 20170206) | (day < 20170109)')
    dataGrid = data.query(' day < 20170227  ')


    a = np.array(dataGrid)

    # dayno,week,weekday,hour
    x = a[:, 2:6]

    # car_number
    y = a[:, 6]

    poly = PolynomialFeatures(DRGREE)
    x = poly.fit_transform(x)

    x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.2, random_state=0)
    clf = Ridge(alpha=1.0, fit_intercept=True, normalize=False)
    aa = clf.fit(x_train, y_train)


    x_result = np.zeros((14 * 14, 4))

    index = 0
    for i in range(14):
        day = i + 70 - 7 * 4
        week = day // 7
        dayofweek = day % 7

        # for j in range(24):
        for j in range(9,23):
            hour = j

            x_result[index][0] = day
            x_result[index][1] = week
            x_result[index][2] = dayofweek
            x_result[index][3] = hour

            index += 1

    poly2 = PolynomialFeatures(DRGREE)
    x_result_trans = poly2.fit_transform(x_result)

    # print(x_result_trans)

    y_result = clf.predict(x_result_trans)

    half = int(len(y_result) / 2)

    y_result2 = np.append (y_result[half:], y_result[:half])


    dataValid = data.query('day >= 20170227 & hour >= 9 &  hour <= 22')

    y_valid = np.array(dataValid)[:, 6]



    return  (calcSumSqurt(y_valid, y_result2), len(y_result))



def ridgeRegressionValid ():
    data = pd.read_csv(grid.RESULT_FILE)

    # 默认6
    DRGREE = 2

    # # 添加week字段
    # data['week'] = data['dayno'] // 7

    # data = data.reindex(columns=['grid_id', 'day',  'dayno', 'week', 'weekday', 'hour', 'car_number'])


    # data.to_csv(grid.RESULT_FILE + '2', index = False)

    data = data.query('grid_id  == 98')
    # data = data.query('day >= 20170213')

    dataGrid = data.query('grid_id  == 98')

    dataGrid = dataGrid.query('day < 20170227')


    a = np.array(dataGrid)

    # 绘制原始carnum
    # plt.plot(a[:, 6])
    # plt.show()

 
 
   # dayno,week,weekday,hour
    x = a[:, 2:6]

    #car_number 
    y = a[:, 6]

    poly = PolynomialFeatures(DRGREE)
    x = poly.fit_transform(x)

    # print(x)

    # return


    x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size = 0.2,random_state = 0)
    clf = Ridge(alpha = 0.1, fit_intercept = True, normalize= False)
    aa = clf.fit(x_train, y_train)
    # clf.score(x_test, y_test)

    # print(clf.coef_, aa)
    print('score',clf.score(x_test, y_test))

    start = len(y)  - 7*24  * 2
    end = len(y)  
    time = np.arange(start, end)



    y_pre = clf.predict(x)
    plt.plot(time, y[start:end], 'b', label = 'real')
    plt.plot(time, y_pre[start:end], 'r', label = 'predict')
    plt.legend(loc = 'upper left')
    plt.show()

  

    print('ridgeRegression begin')


    x_result = np.zeros ( ( 24 * 14, 4))
 
    index = 0
    for i in range(14):
        day = i + 70 - 7 * 2 
        week = day // 7
        dayofweek = day % 7

        for j in range(24):
            hour = j

            x_result[index][0] = day
            x_result[index][1] = week
            x_result[index][2] = dayofweek
            x_result[index][3] = hour

            index+=1
            
    poly2 = PolynomialFeatures(DRGREE)
    x_result_trans = poly2.fit_transform(x_result)

    # print(x_result_trans)

    y_result = clf.predict(x_result_trans)

    # y_result = y_result * 1.1

    dataValid = data.query('day >= 20170227')
 

    y_valid = np.array(dataValid)[:,6]
 

    # print(y_result)

    time2 = np.arange(len(y_result))

    print('RMSE: ',calcRMSE(y_valid, y_result))

    # ysort = np.sort(y_result)
    #
    # mean1 = ysort[-14]
    # for  i in range(len(y_result)):
    #     if y_result[i] > mean1:
    #         y_result[i] = y_result[i] * 1.01
    #
    # print('RMSE2: ', calcRMSE(y_valid, y_result))


    plt.plot(time2,y_valid, 'b', label = 'real')
    plt.plot(time2, y_result, 'r', label = 'predict')

    plt.show()
    # # print (x_result)

    # # print (a[:, 2:6])

     

# RMSE(均方根误差)
def calcRMSE(x_obs, x_model):
    if len(x_obs) != len(x_model):
        print ('ERROR RMS', len(x_obs), len(x_model))
        return 0
    
    sum = 0
    for i in range(len(x_obs)):
        diff = x_obs[i] - x_model[i]
        sum += (diff * diff)
    return (sum / len(x_obs)) ** 0.5


def calcSumSqurt(x_obs, x_model):
    if len(x_obs) != len(x_model):
        print('ERROR RMS', len(x_obs), len(x_model))
        return 0

    sum = 0
    for i in range(len(x_obs)):
        diff = x_obs[i] - x_model[i]
        sum += (diff * diff)
    return sum


if __name__ == "__main__":

    # validAll()
    # ridgeRegressionShow()
    # AllRegressionResult()
    showresult()