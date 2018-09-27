# coding=utf-8
# 2018-09-22

# 尝试加权平均
 

import numpy as np
import pandas as pd
 
import matplotlib.pyplot as plt
from scipy.linalg import solve

from datetime import datetime
 
import grid

def weighted_mean_show():

    data = pd.read_csv(grid.RESULT_FILE)
  
    data = data.query('day >= 20170206')

    dataGrid = data.query('grid_id  == 44 & weekday == 1 & hour == 9')

    c  = np.array(dataGrid['car_number'])

    # 解方程组
    # c0 w0 + c1w1 + c2w2 = c3
    # c1 w0 + c2w1 + c3w2 = c4
    # w0 + w1 + w2 = 1

    # a = np.array([[c[0], c[1], c[2] ], [c[1], c[2], c[3] ], [1, 1, 1]])
    # b = np.array([c[3], c[4], 1])
    # w = solve(a, b)

    # c5 =  c[2] * w[0] + c[3] * w[1] + c[4] * w[2]

    # 解方程组
    # c0 w0 + c1w1 = c3 - c2w2
    # w0 + w1  = 1 - w2

    w3 = 0.5
    a = np.array([[c[0], c[1]  ],  [1, 1]])
    b = np.array( [c[3] - c[2] * w3,  1 - w3 ])
    w = solve(a, b)


    c5 =  c[2] * w[0] + c[3] * w[1] + c[4] * w3
    
    print(w, w3)
    print(c,c5)
    plt.plot(c)
 
    plt.show()

 

def getResultWeightMean():

    fileName = grid.RESULT_FILE
    dfAll = pd.read_csv(fileName, dtype = {'car_number':  np.float64},)

    dfOut = pd.DataFrame(columns = ["grid_id", "day", "hour", "car_number"])

    # 预测第一周，由前三周对应weekday，hour的车辆数做加权平均 [0.5, 0.3, 0.2]
    # 预测第二周，由预测第一周的车辆数 * 0.85, 从全国趋势看，第二周车流量硬小于第一周
    date = 20170313
    for dayno in range(70, 77):
        for hour in range (9, 23):
            for grid_id in range(1, 51):
                carnum = getResultWeightMean_Last3Week(dfAll, grid_id, dayno,  hour)
                # print(grid_id,date, hour, carnum)

                # 第一周
                dfOut.loc[len(dfOut)] = {'grid_id': grid_id, 'day': date,'hour': hour, 'car_number':carnum}

                # 第二周, 同一个月才能直接+
                dfOut.loc[len(dfOut)] = {'grid_id': grid_id, 'day': date + 7 ,'hour': hour, 'car_number':carnum * 0.85}
        
        date+=1

    
    dfOut['grid_id'] = dfOut['grid_id'].astype('int')
    dfOut['day'] = dfOut['day'].astype('int')
    dfOut['hour'] = dfOut['hour'].astype('int')
   

    dfOut = dfOut.sort_values(by=['day', 'hour', 'grid_id'], axis=0, ascending=True)

    dfOut.to_csv(grid.DATA_PATH + 'result_7.csv', index = False)

 





def getCarNumByDayNo(dfAll, gridid, dayno, hour):
    dfGrid = dfAll.query('grid_id == @gridid & dayno == @dayno & hour == @hour')

    # print(dfGrid)
    carnum = 0
    if(len(dfGrid) > 0):
        carnum = dfGrid.iloc[0, 6]
        # print(carnum)
    

    return carnum



def getResultWeightMean_Last3Week(dfAll, gridid, dayno, hour):

    w = [0.5, 0.3, 0.2]

    lastday   = [dayno - 7, dayno - 7 * 2, dayno - 7*3]
    carnum = [0,0,0]

    for i in range(len(lastday)):
        carnum[i] = getCarNumByDayNo(dfAll, gridid, lastday[i], hour)

    # print(carnum)

    car_num_result = carnum[0] * w[0] + carnum[1] * w[1] + carnum[2] * w[2]
    return car_num_result
 
    