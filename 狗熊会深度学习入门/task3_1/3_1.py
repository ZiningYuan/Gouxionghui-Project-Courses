#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from PIL import Image

MasterFile = pd.read_csv("/clubear/Lecture 2.1 - Linear Regression by TensorFlow/data/faces/FaceScore.csv")


# # 数据准备

# In[2]:


FileNames=MasterFile['Filename'] 
N=len(FileNames)
IMSIZE=128
X=np.zeros([N,IMSIZE,IMSIZE,3])
for i in range(N):
    MyFile=FileNames[i]
    Im=Image.open('/clubear/Lecture 2.1 - Linear Regression by TensorFlow/data/faces/images/'+MyFile)
    Im=Im.resize([IMSIZE,IMSIZE])
    Im=np.array(Im)/255
    X[i,]=Im


# In[3]:


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


# In[4]:


from sklearn.model_selection import train_test_split
X0,X1,Y0,Y1=train_test_split(X,Y,test_size=0.3,random_state=233) #固定seed为233，train：test = 7:3


# In[5]:


from matplotlib import pyplot as plt
plt.figure()
fig,ax=plt.subplots(3,5)
fig.set_figheight(7.5)
fig.set_figwidth(15)
ax=ax.flatten()
for i in range(15):
    ax[i].imshow(X0[i,:,:,:])
    ax[i].set_title(Y0[i])
#查看数据是否consistent ie 0和1的添加是否对应


# In[6]:


# 产生One-Hot型因变量


# In[7]:


from keras.utils import to_categorical
YY0 = to_categorical(Y0)
YY1 = to_categorical(Y1)
YY1


# # CNN模型

# In[15]:


from keras.layers import Dense, Flatten, Input
from keras.layers import BatchNormalization, Conv2D,MaxPooling2D
from keras import Model

input_layer=Input([IMSIZE,IMSIZE,3])
x=input_layer
x=BatchNormalization()(x)           #加快训练速度，提高模型精度
x=Conv2D(10,[2,2], padding = "valid",activation='relu')(x) 
#卷积层，得到新的像素矩阵
#filter = 10 (我看sample size是60000用的filter是64，这个sample没有很复杂所以尝试不是太高的filter) 用了10个卷积核
#kernel size = 4* 4, padding = “valid” 代表用的是valid 卷积
#用ReLu去掉负数值因为我们不需要

x=MaxPooling2D([16,16])(x)
#最大池化层，挑取每个小矩形16*16中的最大值

x=Flatten()(x)
x=Dense(2,activation='softmax')(x)
#softmax也是个逻辑回顾的activati哦你
output_layer=x
model=Model(input_layer,output_layer)
model.summary()


# # 参数解释
# 

# 1) 第一个layer没有任何参数因为没有任何需要学习的内容
# 

# 2）Normalisation layer的参数是自己学习到的，不固定

# 3) 卷积层有 （2*2（卷积核的长宽）*3（从上一个layer留下的3个通道）+ 1（每个新通道的bias截距项）） * 10 （新通道） = （2*2*3 +1）*10 = 130

# 4）池化层没有任何学习项因为只是取了最值

# 5&6） 在dense层 490 * 2 + 2= 982 

# # 运用模型看精确度

# In[16]:


from keras.optimizers import Adam
model.compile(optimizer = Adam(0.05),
             loss = "categorical_crossentropy",
             metrics = ["accuracy"])


# In[17]:


model.fit(X0,YY0, validation_data = (X1,YY1),
         batch_size = 200, 
          epochs = 20)
 #因为有3850个sample在train data里，batch_size不宜很多
#尝试了epochs = 10，发现accuracy还有向上升的空间，于是将epochs定为20


# 可以看到现在的accuracy是0.88，比之前直接套用logistic regression要高出0.04。

# In[ ]:




