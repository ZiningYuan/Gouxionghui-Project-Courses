#!/usr/bin/env python
# coding: utf-8

# # 学习经典模型 AlexNet
# 

# “AlexNet采用8层的神经网络，5个卷积层和3个全连接层(3个卷积层后面加了最大池化层)”。 这里和下方网络结构中有一点不一样，感觉网络结构详解中第四层应该为 MaxPooling2D（3*3), stride(2)

# ## 网络结构（详解）
# 
# > 输入层: 227×227×3的图片。
# 
# > 第1层：Conv2D(11×11，96), Stride(4), ReLU, Output: 55×55×96。
# 
# > 第2层：MaxPooling2D(3×3), Stride(2), Output: 27×27×96。
# 
# > 第3层：Conv2D(5×5，256），Same，Output：27×27×256。
# 
# > 第4层：Conv2D(3×3,256），Stride(2)，Output：13×13×256。
# 
# > 第5层：Conv2D(3×3,384），Same，Output：13×13×384。
# 
# > 第6层：Conv2D(3×3,384），Same，Output：13×13×384。
# 
# > 第7层：Conv2D（3×3，256），Same，Output：13×13×256。
# 
# > 第8层：MaxPooling2D（3×3），Stride（2），Output：6×6×256。
# 
# > 输出层：Flatten，Dense(4096)，Dropout(0.5)，Dense（4096），Dropout（0.5），Output。 

# ## 代码实现

# In[3]:


import tensorflow as tf
import keras
from keras.layers import Activation,Conv2D, BatchNormalization, Dense
from keras.layers import Dropout, Flatten, Input, MaxPooling2D, ZeroPadding2D
from keras import Model
#ZeroPadding2D "can add rows and columns of zeros at the top, bottom, left and right side of an image tensor."
#但是在这里我并没有看到实际运用？

IMSIZE = 227
input_layer = Input([IMSIZE,IMSIZE,3])
x = input_layer 
x = Conv2D(96,[11,11],strides = [4,4], activation = 'relu')(x) 
#第一个卷积层，通道数96，卷积核大小为 11*11 ，步长为列4行4，padding 为默认的 valid，使用的activation function是ReLu
x = MaxPooling2D([3,3], strides = [2,2])(x)    
#池化层，核大小为3*3， 步长列2行2
x = Conv2D(256,[5,5],padding = "same", activation = 'relu')(x)
#第二个卷积层，通道数256，卷积核大小为 5*5 ， 步长为默认的列1行1，padding 为same，使用的activation function是ReLu
x = MaxPooling2D([3,3], strides = [2,2])(x)
#池化层，核大小为3*3， 步长列2行2
x = Conv2D(384,[3,3],padding = "same", activation = 'relu')(x) 
#第三个卷积层，通道数384，卷积核大小为 3*3 ，步长为默认的列1行1，padding 为same，使用的activation function是ReLu
x = Conv2D(384,[3,3],padding = "same", activation = 'relu')(x) 
#第四个卷积层，通道数384，卷积核大小为 3*3 ，步长为默认的列1行1，padding 为same，使用的activation function是ReLu
x = Conv2D(256,[3,3],padding = "same", activation = 'relu')(x) 
#第五个卷积层，通道数256，卷积核大小为 3*3 ，步长为默认的列1行1，padding 为same，使用的activation function是ReLu
x = MaxPooling2D([3,3], strides = [2,2])(x)
#池化层，核大小为3*3， 步长列2行2
x = Flatten()(x)   
x = Dense(4096,activation = 'relu')(x)
#第一层全联接到4096个节点，activation依旧是relu
x = Dropout(0.5)(x)
x = Dense(4096,activation = 'relu')(x)
#第二层全联接还是到4096个节点，activation依旧是relu
x = Dropout(0.5)(x)
x = Dense(2,activation = 'softmax')(x)
#第三层全联接到2个节点，因为我们需要最后的output是2，activation依旧是relu
output_layer=x
model=Model(input_layer,output_layer)
model.summary()


# # 模仿模型进行性别判断
# 

# ## 数据处理
# 

# In[15]:


import pandas as pd
import numpy as np
from PIL import Image

MasterFile = pd.read_csv("/clubear/Lecture 2.1 - Linear Regression by TensorFlow/data/faces/FaceScore.csv")
FileNames=MasterFile['Filename'] 
N=len(FileNames)
IMSIZE=227
X=np.zeros([N,IMSIZE,IMSIZE,3])
for i in range(N):
    MyFile=FileNames[i]
    Im=Image.open('/clubear/Lecture 2.1 - Linear Regression by TensorFlow/data/faces/images/'+MyFile)
    Im=Im.resize([IMSIZE,IMSIZE])
    Im=np.array(Im)/255
    X[i,]=Im


# In[16]:


list_y = [] #创建一个空的list
for i in range(N):
    MyFile = FileNames[i] #提取Filenames
    if MyFile[0] == "f": 
         list_y.append(0)
    elif MyFile[0] == "m":
         list_y.append(1)
#如果filename 第一个字母是f，添加0在list里；反之则添加1 （用0和1表示性别）
Y = np.asarray(list_y)
Y



# In[17]:


from sklearn.model_selection import train_test_split
X0,X1,Y0,Y1=train_test_split(X,Y,test_size=0.3,random_state=233) #固定seed为233，train：test = 7:3

from keras.utils import to_categorical
YY0 = to_categorical(Y0)
YY1 = to_categorical(Y1)
YY1


# ## AlexNet模仿
# 保证使用8层，5个卷积层和3个全连接层(3个卷积层后面加了最大池化层)，参数略有调整， 保证使用ReLu作为激活函数，使用Dropout
# 

# In[33]:


IMSIZE = 227
input_layer = Input([IMSIZE,IMSIZE,3])
x = input_layer 
x = Conv2D(96,[11,11],strides = [4,4], activation = 'relu')(x) 
#第一个卷积层，通道数96，卷积核大小为 11*11 ，步长为列4行4，padding 为默认的 valid，使用的activation function是ReLu
x = MaxPooling2D([3,3], strides = [2,2])(x)    
#池化层，核大小为3*3， 步长列2行2
x = Conv2D(256,[5,5],padding = "same", activation = 'relu')(x)
#第二个卷积层，通道数256，卷积核大小为 5*5 ， 步长为默认的列1行1，padding 为same，使用的activation function是ReLu
x = MaxPooling2D([3,3], strides = [2,2])(x)
#池化层，核大小为3*3， 步长列2行2
x = Conv2D(384,[3,3],padding = "same", activation = 'relu')(x) 
#第三个卷积层，通道数384，卷积核大小为 3*3 ，步长为默认的列1行1，padding 为same，使用的activation function是ReLu
x = Conv2D(384,[3,3],padding = "same", activation = 'relu')(x) 
#第四个卷积层，通道数384，卷积核大小为 3*3 ，步长为默认的列1行1，padding 为same，使用的activation function是ReLu
x = Conv2D(256,[3,3],padding = "same", activation = 'relu')(x) 
#第五个卷积层，通道数256，卷积核大小为 3*3 ，步长为默认的列1行1，padding 为same，使用的activation function是ReLu
x = MaxPooling2D([3,3], strides = [2,2])(x)
#池化层，核大小为3*3， 步长列2行2
x = Flatten()(x)   
x = Dense(4096,activation = 'relu')(x)
#第一层全联接到4096个节点，activation依旧是relu
x = Dropout(0.5)(x)
x = Dense(4096,activation = 'relu')(x)
#第二层全联接还是到4096个节点，activation依旧是relu
x = Dropout(0.5)(x)
x = Dense(2,activation = 'softmax')(x)
#第三层全联接到2个节点，因为我们需要最后的output是2，activation依旧是relu
output_layer=x
model2=Model(input_layer,output_layer)
model2.summary()


# ## 参数解释

# 第一层input layer，规定input只能是[227,227,3]的矩阵，无参数。 <br>
# 第二层conv2D： （11*11*3+1）* 96 = 364 * 96 = 34,944, 其中11*11是kernel size，3是上一层遗留通道数，97是卷积核个数。 <br>
# 第三层池化使用[3,3]矩形，步长行列都是2，没有学习项，无参数<br>
# 第四层conv2D: （5*5*96+1）* 256 = 2401 * 256 = 614,656, 其中5*5是kernel size，96是上一层遗留通道数，256是卷积核个数。 <br>
# 第五层池化使用[3,3]矩形，步长行列都是2，没有学习项，无参数 <br>
# 第六层conv2D: （3*3*256+1）* 384 = 2305 * 384 = 885,120, 其中3*3是kernel size，256是上一层遗留通道数，384是卷积核个数。 <br>
# 第七层conv2D: （3*3*384+1）* 384 = 3457 * 384 = 1,327,488, 其中3*3是kernel size，384是上一层遗留通道数，384是卷积核个数。 <br>
# 第八层conv2D: （3*3*384+1）* 256 = 3457 * 256 = 884,992, 其中5*5是kernel size，6是上一层遗留通道数，16是卷积核个数。 <br> 
# 第九层池化使用[3,3]矩形，步长行列都是2 ，没有学习项，无参数 <br>
# 第十层压扁，无参数        <br>
# 第十一层Dense：    9216 * 4096 + 4096 = 37748,736 + 4096 = 37,752,832 <br>
# 第十二层Dense：   4096 * 4096 + 4096 = 16,777,216 + 4096 = 16,781,312 <br>
# 第十三层Dense： 4096 * 2 + 2 = 8192 + 2 = 8,194 <br>
# 所以总共有58289538个参数。
# 
# 
# 

# ## Compile

# In[36]:


from keras.optimizers import Adam
model2.compile(optimizer = Adam(0.001),
             loss = "categorical_crossentropy",
             metrics = ["accuracy"])


# In[37]:



model2.fit(X0,YY0, validation_data = (X1,YY1),
         batch_size = 200, 
          epochs = 20)


# 这个accuracy有点没有预料到...

# In[ ]:




