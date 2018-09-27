# TrafficAnalysis
traffic analysis for the big city

数据源于2017Q1 SH 100个区块的流量，根据前10周的流量，预测后2周的流量<br>


岭回归<br>
用最后2周的 8次方回归拟合<br>
顺序<br>
se = 4.123875<br>
逆序<br>
se = 4.110844<br>
alpha= 1.0<br>
se = 4.109666<br>
<br>

使用1次方预测<br>
se = 6.163653<br>
使用2次方预测<br>
se = 5.827279<br>
使用3次方预测<br>
se = 10.285415<br>
采用倒数第1周和倒数第三周 alpha = 0.1


![ridge](https://github.com/annerose/TrafficAnalysis/blob/master/ridge.jpg)

2018-09-27

