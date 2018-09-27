# coding=utf-8
# 2018-08-02


import pandas as pd
import time

import datetime
import os
import logging
import numpy as np

from sklearn import linear_model
from matplotlib.figure import SubplotParams

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

import multiprocessing
import matplotlib.pyplot as plt

import regression
import weightedmean



# 数据路径
DATA_PATH = 'E:\\work\\bigdata\\bigdata\\'

SAMPLE_ECAR_FILE = DATA_PATH + r'ecar\BOT_data_ecar_20170103_20170103_part0.csv'
GRID_FILE = DATA_PATH + 'grid.csv'
GRID_FILE_2 = DATA_PATH + 'grid2.csv'

RESULT_FILE = DATA_PATH + 'result_all.csv'


# grid table
class GridRangeRecord:

    def __init__(self ):
        pass

    def __str__(self):
        return ('grid_id %d, x(%f, %f), y(%f, %f)' % (self.grid_id, self.x_min, self.x_max, self.y_min, self.y_max))

    grid_id = 0
    x_min = 0.0
    x_max = 0.0
    y_min = 0.0
    y_max = 0.0


class PosRecord:
    def __init__(self ):
        pass

    def __str__(self):
        return ('%s,%s,%f,%f,%d' % (self.car_id, self.date_time, self.x, self.y, self.grid_id))

    car_id = ''
    date_time = ''
    x = 0.0
    y = 0.0
    grid_id = 0






# parse grid table
def getGridRangeByDF():


    fileSrc = GRID_FILE
    df_sample = pd.read_csv(fileSrc )

    # 3
    df_sample['x_min'] = 0

    # 4
    df_sample['x_max'] = 0

    # 5
    df_sample['y_min'] = 0

    # 6
    df_sample['y_max'] = 0


    for i in range(len(df_sample)):

        arr_y = df_sample.iloc[i, 1].split('~')
        df_sample.iloc[i, 5] = float(arr_y[0])
        df_sample.iloc[i, 6] = float(arr_y[1])

        arr_x = df_sample.iloc[i, 2].split('~')
        df_sample.iloc[i, 3] = float(arr_x[0])
        df_sample.iloc[i, 4] = float(arr_x[1])



    df_sample.set_index(['x_min', 'x_max','y_min', 'y_max'], inplace=True )

    print(df_sample)
    return df_sample


# parse grid table
def getGridRangeByStruct(fileSrc):


 
    df_sample = pd.read_csv(fileSrc, dtype= str )

    arr = []

    for i in range(len(df_sample)):

        record = GridRangeRecord()

        record.grid_id = int(df_sample.iloc[i, 0])

        arr_y = df_sample.iloc[i, 1].split('~')
        record.y_min = float(arr_y[0])
        record.y_max = float(arr_y[1])

        arr_x = df_sample.iloc[i, 2].split('~')
        record.x_min = float(arr_x[0])
        record.x_max = float(arr_x[1])

        arr.append(record)
    return arr


def getGridIDByPosD(x, y, dfGird):

    gridid = 0

    dfAccess = dfGird.query('@x > x_min and @x < x_max and @y > y_min and @y < y_max')

    if len(dfAccess) > 0:
        gridid = dfAccess.iloc[0, 0]


    return gridid

def getGridIDByPos(x, y, arrGird):

    gridid = 0
    for record in arrGird:
        if x > record.x_min and x < record.x_max and y > record.y_min and y < record.y_max:
            gridid = record.grid_id
            break

    return gridid



def outputGridMif(arrGrid):

    strMifHead = '''Version 300
Charset "WindowsSimpChinese"
Delimiter ","
CoordSys Earth Projection 1, 104
Columns 2
\tgrid_id Decimal(16, 0)
\tnum Decimal(16, 0)
Data

'''

    fileMif = open(DATA_PATH + 'mif/grid.mif', 'w')
    fileMif.write(strMifHead)

    fileMid = open(DATA_PATH + 'mif/grid.mid', 'w')

    for grid in arrGrid:
        fileMif.write('Pline 5\n')

        fileMif.write('%f %f\n' % (grid.x_min, grid.y_min))
        fileMif.write('%f %f\n' % (grid.x_min, grid.y_max))
        fileMif.write('%f %f\n' % (grid.x_max, grid.y_max))
        fileMif.write('%f %f\n' % (grid.x_max, grid.y_min))
        fileMif.write('%f %f\n' % (grid.x_min, grid.y_min))

        fileMif.write('\n')

        fileMid.write('%d,0\n' % (grid.grid_id) )

    fileMif.close()
    fileMid.close()



def CleanDataOneFile(strSrcFileName, strOutFileName, arrGrid):


    df_sample = pd.read_csv(strSrcFileName, dtype={'car_id': str, 'lat': np.float64, 'lon': np.float64},
                            usecols =['car_id', 'date_time', 'lat', 'lon'] )

    # dataframe 赋值及其的慢!
    # df_sample['grid_id'] = 0

    fileOutCSV = open(strOutFileName, 'w')
    fileOutCSV.write('car_id,date_time,x,y,grid_id\n')


    start = time.time()
    for i in range(len(df_sample)):

        # df_sample.iloc[i, 4] = getGridIDByPos(df_sample.iloc[i, 3], df_sample.iloc[i, 2], arrGrid)

        pos = PosRecord()

        pos.car_id = df_sample.at[i, 'car_id']
        pos.date_time = df_sample.at[i, 'date_time']
        pos.y = df_sample.at[i, 'lat']
        pos.x = df_sample.at[i, 'lon']
        pos.grid_id = getGridIDByPos(pos.x, pos.y, arrGrid)
        # arr.append(pos)

        strLine = str(pos) + '\n'
        fileOutCSV.write(strLine)


        # if i % 1000 == 0 :
        #     print('--- %d, %d, %f, time = %d, %s' % (i, len(df_sample), float(i) / len(df_sample),  time.time() - start, str(pos)))


    fileOutCSV.close()

    print(strSrcFileName, len(df_sample), 'done', time.time() - start)


def getCarPosTable(arrGrid):

    fileSrc = SAMPLE_ECAR_FILE
    df_sample = pd.read_csv(fileSrc, dtype={'car_id': str, 'lat': np.float64, 'lon': np.float64},
                            usecols =['car_id', 'date_time', 'lat', 'lon'] )

    # dataframe 赋值及其的慢!
    # df_sample['grid_id'] = 0

    arr = []

    fileOutCSV = open(SAMPLE_ECAR_FILE + '.csv', 'w')
    fileOutCSV.write('car_id,date_time,x,y,grid_id\n' )


    start = time.time()
    for i in range(len(df_sample)):

        # df_sample.iloc[i, 4] = getGridIDByPos(df_sample.iloc[i, 3], df_sample.iloc[i, 2], arrGrid)

        pos = PosRecord()

        pos.car_id = df_sample.iloc[i, 0]
        pos.date_time = df_sample.iloc[i, 1]
        pos.y = df_sample.iloc[i, 2]
        pos.x = df_sample.iloc[i, 3]
        pos.grid_id = getGridIDByPos(pos.x, pos.y, arrGrid)
        # arr.append(pos)

        strLine = str(pos) + '\n'
        fileOutCSV.write(strLine)


        if i % 1000 == 0 :
            print('--- %d, %d, %f, time = %d, %s' % (i, len(df_sample), float(i) / len(df_sample),  time.time() - start, str(pos)))


    # print (arr)

    fileOutCSV.close()
    print(SAMPLE_ECAR_FILE, len(df_sample), 'done')


# 进程函数
def worker(strInFileName, arrGrids):


    strBaseFile = os.path.basename(strInFileName)
    strDirFile = os.path.dirname(strInFileName).replace('all', 'clean')
    strOutFileName = r'%s\cleaned_%s' % (strDirFile, strBaseFile)

    if not os.path.isdir(strDirFile):
        os.makedirs(strDirFile)

    CleanDataOneFile(strInFileName, strOutFileName, arrGrids)


# 以多进程清洗数据, 计算每个点的所属的grid
def CleanDataAll_MulProcess(arrGrids):
    path = DATA_PATH + 'all'

    fileList = []

    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            fileList.append(os.path.join(root, name))


    pool = multiprocessing.Pool(processes=4)

    for index in range(len(fileList)):
        strInFileName = fileList[index]
        logging.info('CleanDataAll_MulProcess %d / %d : start parsing %s' % (index, len(fileList), strInFileName))

        pool.apply_async(worker, (strInFileName,arrGrids), )  # 维持执行的进程总数为processes，当一个进程执行完毕后会添加新的进程进去


    pool.close()
    pool.join()  # 调用join之前，先调用close函数，否则会出错。执行完close后不会有新的进程加入到pool,join函数等待所有子进程结束
    print ("Sub-process(es) done.")


# 根据日期获取输入文件名
# 20170102 -> ecar\cleaned_BOT_data_ecar_20170102_20170102_part0.csv
#             rcar\cleaned_BOT_data_rcar_20170102_20170102_part0.csv
def getInputFileByDay(strDay):

    fileList = []
    for i in range(3):
        strEcar =  '%sclean\\ecar\\cleaned_BOT_data_ecar_%s_%s_part%d.csv' % (DATA_PATH, strDay, strDay, i)
        strRcar = '%sclean\\rcar\\cleaned_BOT_data_rcar_%s_%s_part%d.csv' % (DATA_PATH, strDay, strDay, i)

        fileList.append(strEcar)
        fileList.append(strRcar)

        print(strEcar)
        print(strRcar)

    return fileList

# '20170102'
def aggreGridByDay(strDay, index):

    print('aggreGridByDay', strDay)

    t0 = time.time()

    listFile = getInputFileByDay(strDay)
    mapHour2Car = dict()

    dt = datetime.datetime.strptime(strDay, "%Y%m%d")
    wd = dt.weekday()



    for file in listFile:
        aggreGridOneFile(file, mapHour2Car)



    strOutFile = '%sresult\\%s.csv' % (DATA_PATH, strDay)
    file = open(strOutFile, 'w')
    file.write('grid_id,day,dayno,weekday,hour,car_number\n')


    for hour in range(0, 24):
        for gid in range(51,101):

            carNum = 0
            key = '%s_%02d_%02d' % (strDay, hour, gid)
            setCar = mapHour2Car.get(key)
            if setCar  is not None:
                carNum = len(setCar)

            # print('%d,%s,%d,%d' % (gid,strDay,hour, carNum))

            file.write('%d,%s,%d,%d,%d,%d\n' % (gid,strDay,index,wd,hour, carNum))

    file.close()

    print('aggreGridByDay %s done, %d' % (strDay, time.time() - t0))


def getDataRangeGrid():

    startTime = datetime.datetime(2017, 1, 2)
    endTime = datetime.datetime(2017, 3, 12)

    currentTime = startTime
    while currentTime <= endTime :

        strDay = '%d%02d%02d' % (currentTime.year, currentTime.month, currentTime.day)

        aggreGridByDay(strDay)


        currentTime  = currentTime + datetime.timedelta(days=1)





def GetDataRangeGrid_MulProcess():

    startTime = datetime.datetime(2017, 1, 2)
    endTime = datetime.datetime(2017, 3, 12)

    currentTime = startTime

    arrDays = []
    while currentTime <= endTime :

        strDay = '%d%02d%02d' % (currentTime.year, currentTime.month, currentTime.day)
        arrDays.append(strDay)
        currentTime  = currentTime + datetime.timedelta(days=1)



    pool = multiprocessing.Pool(processes=4)

    for index in range(len(arrDays)):
        theDay = arrDays[index]
        logging.info('GetDataRangeGrid_MulProcess %d / %d : start parsing %s' % (index, len(arrDays), theDay))
        # 维持执行的进程总数为processes，当一个进程执行完毕后会添加新的进程进去

        pool.apply_async(aggreGridByDay,( theDay, index,),)



    pool.close()
    pool.join()  # 调用join之前，先调用close函数，否则会出错。执行完close后不会有新的进程加入到pool,join函数等待所有子进程结束
    print ("Sub-process(es) done.")




def aggreGridOneFile(strFile, mapHour2Car):

    dfCleaned = pd.read_csv(strFile, dtype={'car_id': str, 'grid_id': np.int },parse_dates=['date_time' ],
                            usecols =['car_id', 'date_time', 'grid_id'])

    # print(dfCleaned.dtypes)

    for i in range(len(dfCleaned)):
        gridid = dfCleaned.at[i, 'grid_id']

        if gridid  != 0 :
            dt = dfCleaned.at[i, 'date_time']
            carid = dfCleaned.at[i, 'car_id']



            key = '%d%02d%02d_%02d_%02d' % (dt.year, dt.month, dt.day , dt.hour, gridid)

            setCar = mapHour2Car.get(key)
            if setCar is None:
                setCar = {carid}
                mapHour2Car[key] = setCar
            else:
                if carid in setCar:
                    pass
                else:
                    setCar.add(carid)
                    mapHour2Car[key] = setCar






def concatResult():

    path = DATA_PATH + 'result'

    fileList = []

    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            fileList.append(os.path.join(root, name))
            print(os.path.join(root, name))

    df  = []
    for strFile in fileList:
        dfCSV = pd.read_csv(strFile)
        df.append(dfCSV)

    dfAll = pd.concat(df)

    dfAll.to_csv(RESULT_FILE, index=False)


# 添加weekday, week 字段
def addWeekdayField():

    dfCSV = pd.read_csv(RESULT_FILE)

    # dfCSV['weekday'] = 0

    # 添加week字段
    dfCSV['week'] = dfCSV['dayno'] // 7

    dfCSV = dfCSV.reindex(columns=['grid_id', 'day',  'dayno', 'week', 'weekday', 'hour', 'car_number'])
    # for i in range(len(dfCSV)):

    #     iDate = dfCSV.at[i, 'day']
    #     dt = datetime.datetime.strptime(str(iDate), "%Y%m%d")
    
    #     dfCSV.at[i, 'weekday'] = dt.weekday()

    dfCSV.to_csv(DATA_PATH + 'result2.csv', index=False)


def showDays(dfAll):

    df = dfAll.query('grid_id == 44 & hour == 9')

    df.to_csv(DATA_PATH+'1_9.csv', index=False)

    TOP = 70

    seqHoliday = [0, 0, 0, 0, 0, TOP, TOP,
                  0, 0, 0, 0, 0, TOP, TOP,
                  0, 0, 0, 0, 0, TOP, TOP,
                  0, 0, 0, 0, 0, TOP, TOP,
                  0, 0, 0, 0, 0, TOP, TOP,
                  0, 0, 0, 0, 0, TOP, TOP,
                  0, 0, 0, 0, 0, TOP, TOP,
                  0, 0, 0, 0, 0, TOP, TOP,
                  0, 0, 0, 0, 0, TOP, TOP,
                  0, 0, 0, 0, 0, TOP, TOP,
                  ]

    print(df)

    seq = range(len(df))


    plt.plot(seq, df['car_number']);
    plt.plot(seq, seqHoliday);
    plt.show()



def showResultDays(dfAll):

    df = dfAll.query('grid_id == 92 & hour == 9')

    df.to_csv(DATA_PATH+'1_9.csv', index=False)

    TOP = 70

    seqHoliday = [0, 0, 0, 0, 0, TOP, TOP,
                  0, 0, 0, 0, 0, TOP, TOP,
                  0, 0, 0, 0, 0, TOP, TOP,
                  0, 0, 0, 0, 0, TOP, TOP,
                  0, 0, 0, 0, 0, TOP, TOP,
                  0, 0, 0, 0, 0, TOP, TOP,
                  0, 0, 0, 0, 0, TOP, TOP,
                  0, 0, 0, 0, 0, TOP, TOP,
                  0, 0, 0, 0, 0, TOP, TOP,
                  0, 0, 0, 0, 0, TOP, TOP,
                  0, 0, 0, 0, 0, TOP, TOP,
                  0, 0, 0, 0, 0, TOP, TOP,

                  ]

    # print(df)

    print(len(seqHoliday),len(df))

 

    seq = range(len(df))


    plt.plot(seq, df['car_number']);
    plt.plot(seq, seqHoliday);
    plt.show()



def showOneDay(dfAll):

    df = dfAll.query('grid_id == 44 & day == 20170310')

    df.to_csv(DATA_PATH+'1_20170102.csv', index=False)




    seq = range(9,23)


    plt.plot(seq, df['car_number']);
    # plt.plot(seq, seqHoliday);
    plt.show()


def meanGression(dfAll, startTime):
    startNumberTime = datetime.datetime(2017, 1, 2)
    iOut = len(dfAll)
    for i in range(7):
        currentDay = startTime + datetime.timedelta(days=i)
        days = getLast4WeekDay(currentDay)
        iCurrentDay = datetime2IntDay(currentDay)

        wd = currentDay.weekday()

        iDayNum = (currentDay - startNumberTime).days


        print(iDayNum, days)

        for iHour in range(9, 23):
            for iGrid in range(1, 51):
                carNum = 0

                for day in days:
                    carNum += getCarNum(dfAll, iGrid, day, iHour)

                carNum /= 4.0

                # print('%d,%d,%d,%f' % (iGrid, iCurrentDay, iHour, carNum))
                dfAll.loc[iOut] = {'grid_id': iGrid, 'day': iCurrentDay, 'dayno': iDayNum, 'weekday': wd, 'hour': iHour, 'car_number':carNum}
                iOut +=1


# 第一周回归，第二周平均
def ridgeGressionAndMeanResult():

    fileName = RESULT_FILE

    dfAll = pd.read_csv(fileName)

    # showOneDay(dfAll)

    # showDays(dfAll)

    # print(dfAll.dtypes )

    # print(dfAll)

    # df2 = dfAll.query('grid_id == 44 & hour == 20 & day > 20170212')
    # df2 = dfAll.query('grid_id == 1 & day == 20170103')
    # print(df)

    # ridgeGressionShow(dfAll, 50, 21, 20170306, 3)
    #
    # return

    # ridgeGression(dfAll, 50, 20, 20170306, 3)

    # 预测最近一周
    for gridid in range(1, 51):
        for hour in range(9, 23):
            ridgeGression(dfAll, gridid, hour,20170306, 3)

        print('grid %d done' % gridid)

    # 预测第2周, 误差太大，不能用回归
    # for gridid in range(1, 51):
    #     for hour in range(9, 23):
    #         ridgeGression(dfAll, gridid, hour, 20170313, 4)

    # 预测第2周, 用平均值
    startTime = datetime.datetime(2017, 3, 20)

    meanGression(dfAll, startTime)



    dfAll['grid_id'] = dfAll['grid_id'].astype('int')
    dfAll['day'] = dfAll['day'].astype('int')
    dfAll['hour'] = dfAll['hour'].astype('int')
    dfAll['dayno'] = dfAll['dayno'].astype('int')
    dfAll['weekday'] = dfAll['weekday'].astype('int')

    dfAll.to_csv(DATA_PATH + 'result_all_ridge.csv', index=False)

    dfResult = dfAll.query('day > 20170312').sort_values(by=['day', 'hour', 'grid_id'], axis=0, ascending=True)

    dfResult.to_csv(DATA_PATH + 'result4.csv', columns=['grid_id','day','hour','car_number'],index=False)


    #
    # dfArr = []
    # for gridid in range(1, 51):
    #     for hour in range(9, 23):
    #
    #         df = ridgeGression(dfAll, gridid, hour)
    #         dfArr.append(df)
    #
    #         # print('-----------', gridid, hour)
    #
    # dfResult = pd.concat(dfArr)
    #
    # dfResult['grid_id'] = dfResult['grid_id'].astype('int')
    # dfResult['day'] = dfResult['day'].astype('int')
    # dfResult['hour'] = dfResult['hour'].astype('int')
    #
    # dfResult = dfResult.sort_values(by=['day', 'hour', 'grid_id'], axis=0, ascending=True)
    #
    # dfResult.to_csv(DATA_PATH + 'result_all_ridge.csv', index=False)

    # print (df)
    #
    # lineGression(df2)






def ridgeGressionShow(dfAll, iGirdID, iHour, iStartDay, iTrainWeek):


    df_grid_hour = dfAll.query('grid_id == @iGirdID & hour == @iHour & day >= @iStartDay')

    # 特征数
    ATTR_COUNT = iTrainWeek + 2

    # 创建空的二维数组
    x_train = np.zeros ( ( len(df_grid_hour), ATTR_COUNT))
    y_train = np.zeros ( ( len(df_grid_hour) ))

    startDayNo = 0

    for i in range(len(df_grid_hour)):

        dayno = df_grid_hour.iloc[i, 2]
        weekday = df_grid_hour.iloc[i, 3]
        x_train[i][0] = dayno  # dayno

        x_train[i][1] = weekday

        startDayNo = dayno

         # last 1 - 4 week
        for j in range(iTrainWeek):
            x_train[i][j+2] = getCarNumByDayNo(dfAll, iGirdID, dayno - 7 * (j+1), iHour)


        y_train[i] = df_grid_hour.iloc[i, 5]

    print('x_train')
    print(x_train)
    print('y_train')
    print(y_train)

    weeklen = 7

    x_test = np.zeros ( ( weeklen, ATTR_COUNT))

    startDayNo += 1

    for i in range(weeklen):
        daynum = startDayNo + i
        x_test[i][0] = daynum
        x_test[i][1] = i

        for j in range(iTrainWeek):
            x_test[i][j+2] = getCarNumByDayNo(dfAll, iGirdID, daynum - 7 * (j +1), iHour)

    print('x_test')
    print(x_test)





    # for i in range(len(dfTest)):


    # x = x.reshape(-1, 1)
    # y = y.reshape(-1, 1)


    # #
    # # print(x[:,1])
    # # print(x[:,0])
    # print(y)


    clf = linear_model.Ridge(alpha=0.1  )
    # clf = linear_model.LinearRegression()

    aa = clf.fit(x_train, y_train)
    print(clf.coef_, aa)
    print('score',clf.score(x_train, y_train))

    # plt.scatter(x[:,0], y, color='navy', s=30, marker='o', label="training points")

    plt.plot(x_train[:,0], y_train, color='blue', linewidth=1, label="training truth")

    y_plot = clf.predict(x_train)

    plt.plot(x_train[:,0], y_plot, color='orange', linewidth=2, label="predict 0")

    y_test1 = clf.predict(x_test)

    plt.plot(x_test[:, 0], y_test1, color='red', linewidth=2, label="predict 1")

    # # week2
    # startDayNo +=  weeklen
    # x_test2 = np.zeros((weeklen, ATTR_COUNT))
    # for i in range(weeklen):
    #     daynum = startDayNo + i
    #     x_test2[i][0] = daynum
    #     x_test2[i][1] = i
    #
    #     x_test2[i][2] = y_test1[i]
    #     x_test2[i][3] = getCarNumByDayNo(dfAll, iGirdID, daynum - 14, iHour)
    #     x_test2[i][4] = getCarNumByDayNo(dfAll, iGirdID, daynum - 21, iHour)
    #
    # y_test2 = clf.predict(x_test2)
    #
    # print(x_test2)
    # print(y_test2)
    #
    # plt.plot(x_test2[:, 0], y_test2, color='green', linewidth=2, label="predict 2")



    plt.show()


def ridgeGression(dfAll, iGirdID, iHour,  iStartDay, iTrainWeek):

    df_grid_hour = dfAll.query('grid_id == @iGirdID & hour == @iHour & day >= @iStartDay')

    # 特征数
    ATTR_COUNT = iTrainWeek + 2

    # 创建空的二维数组
    x_train = np.zeros ( ( len(df_grid_hour), ATTR_COUNT))
    y_train = np.zeros ( ( len(df_grid_hour) ))

    startDayNo = 0

    for i in range(len(df_grid_hour)):

        dayno = df_grid_hour.iloc[i, 2]
        weekday = df_grid_hour.iloc[i, 3]
        x_train[i][0] = dayno  # dayno

        x_train[i][1] = weekday

        startDayNo = dayno

         # last 1 - 4 week
        for j in range(iTrainWeek):
            x_train[i][j+2] = getCarNumByDayNo(dfAll, iGirdID, dayno - 7 * (j+1), iHour)


        y_train[i] = df_grid_hour.iloc[i, 5]

    # print('x_train')
    # print(x_train)
    # print('y_train')
    # print(y_train)

    weeklen = 7

    x_test = np.zeros ( ( weeklen, ATTR_COUNT))

    startDayNo += 1


    for i in range(weeklen):
        daynum = startDayNo + i
        x_test[i][0] = daynum
        x_test[i][1] = i

        for j in range(iTrainWeek):
            x_test[i][j+2] = getCarNumByDayNo(dfAll, iGirdID, daynum - 7 * (j +1), iHour)



    clf = linear_model.Ridge(alpha=0.1)
    clf.fit(x_train, y_train)

    y_test = clf.predict(x_test)

    iOut = len(dfAll)

    startTime = datetime.datetime(2017, 1, 2)
    for i in range(weeklen):

        iDayNum = x_test[i][0]
        currentDay = startTime + datetime.timedelta(days = iDayNum)
        iDay = datetime2IntDay(currentDay)

        y = y_test[i]
        if y < 0:
            y = 0
        # print(iGirdID, iDay, iDayNum, i, iHour, y)

        dfAll.loc[iOut] = {'grid_id': iGirdID, 'day': iDay, 'dayno': iDayNum, 'weekday': i, 'hour': iHour, 'car_number':y}
        iOut +=1





def polynomial_model(degree=1):
    polynomial_features = PolynomialFeatures(degree=degree,
                                             include_bias=False)
    linear_regression = linear_model.LinearRegression(normalize=True)  #normalize=True对数据归一化处理
    pipeline = Pipeline([("polynomial_features", polynomial_features),#添加多项式特征
                         ("linear_regression", linear_regression)])
    return pipeline


def lineGression(df_grid_hour):
    n_dots = 200

    # X = np.linspace(-2 * np.pi, 2 * np.pi, n_dots)
    # Y = np.sin(X) + 0.2 * np.random.rand(n_dots) - 0.1
    # X = X.reshape(-1, 1)
    # Y = Y.reshape(-1, 1)
    #

    train_size = int(len(df_grid_hour) * 3 / 4)

    # 创建空的二维数组
    x = np.zeros ( (train_size) )
    y = np.zeros ( (train_size) )

    valid_size = len(df_grid_hour) - train_size


    x_valid = np.zeros ((valid_size) )
    y_valid = np.zeros ((valid_size))


    for i in range(train_size):

        x[i] = df_grid_hour.iloc[i, 2]
        y[i] = df_grid_hour.iloc[i, 5] # car_number


    for i in range( len(df_grid_hour) - train_size ):


        x_valid[i] = df_grid_hour.iloc[train_size + i, 2]  # dayno
        y_valid[i] = df_grid_hour.iloc[train_size + i, 5]  # car_number



    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    x_valid = x_valid.reshape(-1, 1)
    y_valid = y_valid.reshape(-1, 1)


    degrees = [3,  7, 20, 150]
    results = []
    for d in degrees:
        model = polynomial_model(degree=d)
        model.fit(x, y)
        train_score = model.score(x, y)  #训练集上拟合的怎么样
        mse = mean_squared_error(y, model.predict(x))  #均方误差 cost
        results.append({"model": model, "degree": d, "score": train_score, "mse": mse})

    for r in results:
        print("degree: {}; train score: {}; mean squared error: {}".format(r["degree"], r["score"], r["mse"]))

    plt.figure(figsize=(12, 6), dpi=200, subplotpars=SubplotParams(hspace=0.3))
    for i, r in enumerate(results):
        fig = plt.subplot(3, 2, i + 1)
        # plt.xlim(-8, 8)
        plt.title("degree={},score={}".format(r["degree"], r["score"]))
        plt.scatter(x, y, s=5, c='b', alpha=0.5)
        plt.plot(x, r["model"].predict(x), 'r-')

        # plt.plot(x_valid, r["model"].predict(x_valid) , 'b-' )

    plt.show()




def validResult():

    fileName = DATA_PATH + 'result_all_mean2.csv'

    dfAll = pd.read_csv(fileName)

    # showOneDay(dfAll)

    showResultDays(dfAll)

    # print(dfAll.dtypes )

    # print(dfAll)


    # df2 = dfAll.query('grid_id == 1 & day == 20170103')



def getCarNum(dfAll, iGrid, iDay, iHour):
    df2 = dfAll.query('grid_id == @iGrid & day == @iDay & hour==@iHour')

    # print(df2)

    aa = df2.iloc[0, 6]

    return aa


def getCarNumByDayNo(dfAll, iGrid, iDayNo, iHour):
    df2 = dfAll.query('grid_id == @iGrid & dayno == @iDayNo & hour==@iHour')

    carnum = 0
    if len(df2) > 0:
        carnum = df2.iloc[0, 6]


    return carnum


def datetime2IntDay(dt):
    strDay = '%d%02d%02d' % (dt.year, dt.month, dt.day)
    return int(strDay)

def getLast4WeekDay(dt):

    arrDay = []
    for i in range(1, 5):
        dt -= datetime.timedelta(days=7)
        iDay = datetime2IntDay(dt)
        arrDay.append(iDay)

    return arrDay


# 预测 【20170313,20170326】连续两周内9<=hour<= 22的车流量预测。
# 1.  根据节后4周来预测  20170213 - 20170312
# 2. 暂不考虑3.8的影响
def getResultBy4WeekMean():

    fileName = RESULT_FILE


    dfAll = pd.read_csv(fileName, dtype = {'car_number':  np.float64},)



    startTime = datetime.datetime(2017, 3, 13)

    for i in range(14):
        currentDay = startTime + datetime.timedelta(days=i)
        days = getLast4WeekDay(currentDay)
        iCurrentDay = datetime2IntDay(currentDay)

        dayno = 70 + i

        print(days)

        for iHour in range(9, 23):
            for iGrid in range(51, 101):
                carNum = 0

                for day in days:
                    carNum += getCarNum(dfAll, iGrid, day, iHour)

                carNum /= 4.0

                # print('%d,%d,%d,%f' % (iGrid, iCurrentDay, iHour, carNum))

                # add new line
                dfAll.loc[len(dfAll)] = {'grid_id': iGrid, 'day': iCurrentDay,'dayno': 70 + i, 'week': dayno // 7, 'weekday':currentDay.weekday(),
                 'hour': iHour, 'car_number':carNum}

    dfAll['grid_id'] = dfAll['grid_id'].astype('int')
    dfAll['day'] = dfAll['day'].astype('int')
    dfAll['hour'] = dfAll['hour'].astype('int')

    dfAll.to_csv(DATA_PATH + 'result2.csv', index=False)

    dfAll.query('day > 20170312').to_csv(DATA_PATH + 'result3.csv', index=False, columns=['grid_id','day','hour','car_number'])


# 预处理
def prepareData():
    arrGrids = getGridRangeByStruct(GRID_FILE_2)
    outputGridMif(arrGrids)

    CleanDataAll_MulProcess(arrGrids)

    GetDataRangeGrid_MulProcess()

    concatResult()

    addWeekdayField()


if __name__ == "__main__":

    start = time.time()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename='grid.log',
                        filemode='a')

    #################################################################################################
    # 定义一个StreamHandler，将INFO级别或更高的日志信息打印到标准错误，并将其添加到当前的日志处理对象#
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG )
    formatter = logging.Formatter('%(asctime)s %(levelname)-4s: %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    #################################################################################################

    logging.info('begin')

    # prepareData()

    # arrGrids = getGridRangeByStruct()
    # outputGridMif(arrGrids)

    # getCarPosTable(arrGrids)

    # CleanDataAll_MulProcess(arrGrids)
    # aggreGridByDay('20170310')

    # getDataRangeGrid()
    # print(fl)

    # GetDataRangeGrid_MulProcess()

    # concatResult()
    # testResult()

    # ridgeGressionAndMeanResult()
    # getResultBy4WeekMean()

    # validResult()

    # addWeekdayField()

    regression.ridgeRegressionShow()
    # regression.AllRegressionResult()

    # regression.showresult()
    # weightedmean.weighted_mean_show()
    # weightedmean.getResultWeightMean()



    logging.info ('done %d s' % (time.time() - start))





