# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 18:25:16 2020

"""
#这是打乱数据用的，如果下载不了这个包，就别用了，也把下面的shuffle删掉
from sklearn.utils import shuffle

from keras.utils import to_categorical
import numpy as np
import scipy.io as sio #读取数据

data1 = sio.loadmat('C_B3_NRev1500Load400') 
data_1 = data1.get('C_B3_NRev1500Load400')
data_1 = shuffle(data_1)
data_1_num = data1.get('num')
data2 = sio.loadmat('C_B3_FRev1500Load400') 
data_2 = data2.get('C_B3_FRev1500Load400') 
data_2 = shuffle(data_2)
data_2_num = data2.get('num')

#一般数据要7/3分，70%训练train，30%验证test
num = [data_1_num[0][0],data_2_num[0][0]]   #[正常样本数量，故障样本数量]
train_1 = data_1[0:int(num[0]*0.7)]       #正常样本70%
train_2 = data_2[0:int(num[1]*0.7)]       #故障样本70%

train_X = np.vstack((train_1,train_2))        #合并为一个大矩阵。前正常，后故障
train_X = train_X.reshape(-1, 1, 8192,1) #-1表示不确定，1行 8192列 单通道
train_X = train_X.astype('float32')

train_y1 = [0]*int(num[0]*0.7)  #正常数据0表示
train_y1 = to_categorical(train_y1, 2)#将原有的类别向量转换为独热编码(01矩阵)的形式
train_y2 = [1]*int(num[1]*0.7)  #故障数据1表示
train_y2 = to_categorical(train_y2, 2)#将原有的类别向量转换为独热编码(01矩阵)的形式
train_y = np.vstack((train_y1,train_y2))   

#这里用1D比较好，我们是一行数据，当然2D也行。
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dropout, Dense
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta
#卷积池化
model = Sequential()#Sequential方法实例化模型对象
model.add(Conv2D(2, (1,5), activation='relu', input_shape=[1, 8192, 1])) #卷积核2，卷积核太多就慢，大小（1,5）
model.add(MaxPool2D(pool_size=(1,16)))#最大池化层   8192/16 = 512
model.add(Conv2D(2, (1,5), activation='relu'))
model.add(MaxPool2D(pool_size=(1,16)))#最大池化层    512/16 = 32
model.add(Flatten())#数据矩阵展平  一个样本 32（最后池化大小） * 2（卷积核个数） = 64  个特征
#NN
model.add(Dropout(0.3))#dropout操作
model.add(Dense(32, activation='relu'))#全连接层，激活函数relu，节点数最好要小于特征数32<64
model.add(Dropout(0.3))
model.add(Dense(2, activation='softmax')) #softmax分类器  2分类就是2。
# 编译
model.compile(loss=categorical_crossentropy,
             optimizer=Adadelta(),
             metrics=['accuracy'])#指定损失函数、优化器、评判指标
# 
batch_size = 50 #设置批量梯度下降时的batch_size为50
epochs = 50 #设置遍历所有样本的次数epoch为30
#开始训练
model.fit(train_X, train_y,
         batch_size=batch_size,
         epochs=epochs,verbose=2) #调用模型对象的fit方法开始模型训练,verbose=2表示每一个epoch打印一次
# 
#一般数据要7/3分，70%训练train，30%验证test
test_1 = data_1[int(num[0]*0.7):num[0]]       #正常样本30%
test_2 = data_2[int(num[1]*0.7):num[1]]       #故障样本30%

test_X = np.vstack((test_1,test_2))        #合并为一个大矩阵。前正常，后故障
test_X = test_X.reshape(-1, 1, 8192,1) #-1表示不确定，1行 8192列 单通道
test_X = test_X.astype('float32')

test_y1 = [0]*len(test_1)  #正常数据0表示
test_y1 = to_categorical(test_y1, 2)#将原有的类别向量转换为独热编码(01矩阵)的形式
test_y2 = [1]*len(test_2)  #故障数据1表示
test_y2 = to_categorical(test_y2, 2)#将原有的类别向量转换为独热编码(01矩阵)的形式
test_y = np.vstack((test_y1,test_y2))   

loss, accuracy = model.evaluate(test_X, test_y, verbose=1)#使用测试集的数据做模型评估，返回损失函数值和准确率；
print('loss:%.4f accuracy:%.4f' %(loss, accuracy))
#
#import math
#import matplotlib.pyplot as plt
#import random
# 
#def drawDigit3(position, image, title, isTrue):
#    plt.subplot(*position)
#    plt.imshow(image.reshape(-1, 28), cmap='gray_r')
#    plt.axis('off')
#    if not isTrue:
#        plt.title(title, color='red')
#    else:
#        plt.title(title)
#        
#def batchDraw3(batch_size, test_X, test_y):
#    selected_index = random.sample(range(len(test_y)), k=100)
#    images = test_X[selected_index]
#    labels = test_y[selected_index]
#    predict_labels = model.predict(images)
#    image_number = images.shape[0]
#    row_number = math.ceil(image_number ** 0.5)
#    column_number = row_number
#    plt.figure(figsize=(row_number+8, column_number+8))
#    for i in range(row_number):
#        for j in range(column_number):
#            index = i * column_number + j
#            if index < image_number:
#                position = (row_number, column_number, index+1)
#                image = images[index]
#                actual = np.argmax(labels[index])
#                predict = np.argmax(predict_labels[index])
#                isTrue = actual==predict
#                title = 'actual:%d\npredict:%d' %(actual,predict)
#                drawDigit3(position, image, title, isTrue)
# 
#batchDraw3(100, test_X, test_y)
#plt.show()
