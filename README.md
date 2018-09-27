# TrafficAnalysis
traffic analysis for the big city

数据源于2017Q1 SH 100个区块的流量，根据前10周的流量，预测后2周的流量


岭回归
用最后2周的 8次方回归拟合
顺序
se = 4.123875
逆序
se = 4.110844
alpha= 1.0
se = 4.109666


使用1次方预测
se = 6.163653
使用2次方预测
se = 5.827279
使用3次方预测
se = 10.285415
采用倒数第1周和倒数第三周 alpha = 0.1


![ridge](https://github.com/annerose/TrafficAnalysis/blob/master/ridge.jpg)

2018-09-27

