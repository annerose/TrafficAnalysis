# coding=utf-8
# 2018-08-02


import pandas as pd
import time
import binascii
import struct
import datetime
import os
import logging
import numpy as np
# import dm
import multiprocessing
import matplotlib.pyplot as plt


# 数据路径
DATA_PATH = r'E:\projects\bigdata\bigdata\\'

SAMPLE_ECAR_FILE = DATA_PATH + r'ecar\BOT_data_ecar_20170103_20170103_part0.csv'
GRID_FILE = DATA_PATH + 'grid.csv'

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
def getGridRangeByStruct():


    fileSrc = GRID_FILE
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

    fileMif = open('tempdata/grid.mif', 'w')
    fileMif.write(strMifHead)

    fileMid = open('tempdata/grid.mid', 'w')

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




def getCarPosTable(arrGrid):

    fileSrc = SAMPLE_ECAR_FILE
    df_sample = pd.read_csv(fileSrc, dtype={'car_id': str, 'lat': np.float64, 'lon': np.float64},
                            usecols =['car_id', 'date_time', 'lat', 'lon'] )

    # dataframe 赋值及其的慢!
    # df_sample['grid_id'] = 0

    arr = []

    fileOutCSV = open(SAMPLE_ECAR_FILE + '.csv', 'wb')
    fileOutCSV.write('car_id,date_time,x,y,grid_id\r\n'.encode())


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

        strLine = str(pos) + '\r\n'
        fileOutCSV.write(strLine.encode())


    #     if i % 1000 == 0 :
    #         print('--- %d, %d, %f, time = %d, %s' % (i, len(df_sample), float(i) / len(df_sample),  time.time() - start, str(pos)))
    #
    #
    # print (arr)

    fileOutCSV.close()
    print(SAMPLE_ECAR_FILE, len(df_sample), 'done')



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

    arrGrids = getGridRangeByStruct()
    outputGridMif(arrGrids)

    # getCarPosTable(arrGrids)

    logging.info ('done %d s' % (time.time() - start))


