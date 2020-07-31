import numpy as np
np.random.seed(1337)  # for reproducibility
 
from keras.datasets import mnist
from keras.models import Model, Sequential #泛型模型
from keras.layers import Dense, Input, Dropout
import matplotlib.pyplot as plt

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
train_X = train_X.reshape(train_X.shape[0],-1) #-1表示不确定
train_X = train_X.astype('float32')

print(train_X.shape)

train_y1 = [0]*int(num[0]*0.7)  #正常数据0表示
train_y1 = to_categorical(train_y1, 2)#将原有的类别向量转换为独热编码(01矩阵)的形式
train_y2 = [1]*int(num[1]*0.7)  #故障数据1表示
train_y2 = to_categorical(train_y2, 2)#将原有的类别向量转换为独热编码(01矩阵)的形式
train_y = np.vstack((train_y1,train_y2))

test_1 = data_1[int(num[0]*0.7):num[0]]       #正常样本30%
test_2 = data_2[int(num[1]*0.7):num[1]]       #故障样本30%

test_X = np.vstack((test_1,test_2))        #合并为一个大矩阵。前正常，后故障
test_X = test_X.reshape(test_X.shape[0],-1) #-1表示不确定
test_X = test_X.astype('float32')

print(test_X.shape)

test_y1 = [0]*len(test_1)  #正常数据0表示
test_y1 = to_categorical(test_y1, 2)#将原有的类别向量转换为独热编码(01矩阵)的形式
test_y2 = [1]*len(test_2)  #故障数据1表示
test_y2 = to_categorical(test_y2, 2)#将原有的类别向量转换为独热编码(01矩阵)的形式
test_y = np.vstack((test_y1,test_y2))   

# 压缩特征维度至512维
encoding_dim = 64
 
# this is our input placeholder
input_data = Input(shape=(8192,))
 
# 编码层
encoded1 = Dense(1024, activation='relu')(input_data)
encoded2 = Dense(256, activation='relu')(encoded1)
encoded3 = Dense(128, activation='relu')(encoded2)
encoder_output = Dense(encoding_dim)(encoded3)

# 构建编码模型
#encoder = Model(inputs=input_data, outputs=encoder_output)

#分类器
fc = Dense(32, activation='relu')(encoder_output)
fc1 = Dense(8, activation='relu')(fc)
softmax = Dense(2, activation='softmax',name='classification')(fc1)
 
# 解码层
decoded1 = Dense(128, activation='relu')(encoder_output)
decoded2 = Dense(256, activation='relu')(decoded1)
decoded3 = Dense(1024, activation='relu')(decoded2)
decoded = Dense(8192, activation='relu',name='autoencoder')(decoded3)
 
# 构建自编码模型
autoencoder = Model(inputs=input_data, outputs=decoded)
autoencoder.compile(optimizer='adam',loss='mse')

#training
autoencoder.fit(train_X,train_X,epochs=5,batch_size=50,verbose=2,shuffle=True)
 
# 构建分类器模型
classification = Model(inputs=input_data,outputs=softmax)
classification.compile(optimizer='adam',
              loss='mse',
              metrics=['accuracy'])
# binary_crossentropy
# training
classification.fit(train_X,train_y,
          epochs=5, batch_size=10,verbose=2)
 
# testing
loss,accuracy = classification.evaluate(test_X, test_y, verbose=1)
print('loss:%.4f accuracy:%.4f' %(loss,accuracy))
