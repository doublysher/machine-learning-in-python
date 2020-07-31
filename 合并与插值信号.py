# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 14:32:37 2019

@author: 李贺
"""
#目标：将样本插值为同一长度，以便作为神经网络的输入
from scipy import interpolate#插值用
import scipy.io as sio #读取数据
import numpy as np
#读取数据
data1 = sio.loadmat('NRev1500Load400')  #文件名称
data2 = sio.loadmat('NRev1500Load800')
data3 = sio.loadmat('NRev1500Load1200')
data4 = sio.loadmat('NRev1500Load1600')

dataa1 = data1.get('Data')  #变量名称
dataa2 = data2.get('Data')
dataa3 = data3.get('Data')
dataa4 = data4.get('Data')

#定义全局变量
a = locals()#locals() 函数会以字典类型返回当前位置的全部局部变量
num_cyl = 12 #12缸
num_data = 4  #4组数据
#一共12缸，挨个定义每个缸得到数据
for i in range(1,num_cyl+1):
    #这里'dataa1%s'%(i)与'dataa1'+str(i) 效果一样  两类写法
    #不同load不同缸的数据得到了定义，data11代表load400下A1缸的数据
    a['dataa1%s'%(i)] = dataa1[0,0]['Cylinder'+str(i)]  
    a['dataa2%s'%(i)] = dataa2[0,0]['Cylinder'+str(i)]
    a['dataa3%s'%(i)] = dataa3[0,0]['Cylinder'+str(i)]
    a['dataa4%s'%(i)] = dataa4[0,0]['Cylinder'+str(i)]

#4组数据，12缸 ,确定每个缸的样本数量 ，num11与num112代表load400下A1与B6的样本数量
for j in range(1,num_data+1):
    for k in range(1,num_cyl+1):
        a['num%s%s'%(j,k)] = len(a['dataa'+str(j)+str(k)])
        
#定义插值目标，即把每个样本插值为8192的长度，一般为2的指数倍，且接近大多数据的长度
y = np.zeros(shape = (1,8192))

#下面循环中需要的参数预定义
c = 0 #作用：各个load各个缸样本数量
sum_c = 0 #作用： 各个load下总样本数量
t_num = [] #


#循环挨个插值，从load400的A1到load1600的B6
for j in range(1,num_data+1):
    for k in range(1,num_cyl+1):
        cnum = a['num'+str(j)+str(k)]  #读取了每组数据的样本数量 
        #每个样本都需要插值
        for m in range(cnum):
            #插值开始
            temp1 = a['dataa'+str(j)+str(k)][m][0][0] #读取旧数据纵坐标
            x1 = np.linspace(0,len(temp1),len(temp1)) #读取旧数据横坐标
            xnew = np.linspace(0,len(temp1),8192)     #定义新数据横坐标
            f = interpolate.interp1d(x1,temp1,kind='slinear')#线性插值，旧数据线性化
            ynew = f(xnew) #获得新横坐标下的纵坐标，插值完成
           #插值结束
        
            c = c+1 #计数，记录插值了多少样本
            y = np.vstack([y,ynew])#将数据按行（竖直方向）排列
            
        print('load'+str(j*400)+'_'+str(k)+'缸样本数：',c) 
        sum_c = sum_c+c #累加样本数量，获得一个load下总样本数量 
        t_num.append(c) #以向量的形式表示每组样本数量
        c = 0  #置0到下一循环
    print('load'+str(j*400)+'总样本数：',sum_c)
    t_num.append(sum_c)   #以向量的形式表示每个load样本总量
    sum_c = 0 #置0到下一循环
    y = np.delete(y,0,0)#删除第一行，因为全是0
    #保存数据为MATLAB格式  #格式：'文件名'，{'变量名1'：变量内容, '变量名2'：变量内容}
    sio.savemat('C_NRev1500Load'+str(j*400),{'C_NRev1500Load'+str(j*400):y,'num':t_num})  
    
    #置0到下一循环
    c=0
    t_num=[]
    y=np.zeros(shape=(1,8192))
    
    
print('插值结束ok')
