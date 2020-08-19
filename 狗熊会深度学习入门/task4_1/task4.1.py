#!/usr/bin/env python
# coding: utf-8

# # 加载数据

# In[7]:


from keras.datasets import mnist
(X0,Y0),(X1,Y1) = mnist.load_data(path="/clubear/datasets/mnist.npz")
print(X0.shape)
from matplotlib import pyplot as plt
plt.figure()
fig,ax = plt.subplots(2,5)
ax=ax.flatten()
for i in range(10):
    Im=X0[Y0==i][0]
    ax[i].imshow(Im)
plt.show()


# # 数据处理

# In[9]:


from keras.utils import np_utils
#X0不能直接被Tensorflow处理因为缺少一个argument：通道数
N0=X0.shape[0];N1=X1.shape[0]
print([N0,N1])
X0 = X0.reshape(N0,28,28,1)/255 
X1 = X1.reshape(N1,28,28,1)/255
#变成one_hot变量
YY0 = np_utils.to_categorical(Y0)
YY1 = np_utils.to_categorical(Y1)
YY1


# # LeNet5 模型搭建

# In[10]:


from keras.layers import Conv2D,Dense,Flatten,Input,MaxPooling2D,Dropout 
from keras import Model

input_layer = Input([28,28,1])
x = input_layer
x = Conv2D(6,[5,5],padding = "same", activation = 'relu')(x) 
x = MaxPooling2D(pool_size = [2,2], strides = [2,2])(x)    
x = Conv2D(16,[5,5],padding = "valid", activation = 'relu')(x) 
x = MaxPooling2D(pool_size = [2,2], strides = [2,2])(x)
x = Flatten()(x)   
x = Dense(120,activation = 'relu')(x)
x = Dense(84,activation = 'relu')(x)
x = Dense(10,activation = 'softmax')(x)
output_layer=x
model=Model(input_layer,output_layer)
model.summary()


# ## 模型和参数解释

#  第一层input layer，规定input只能是[28,28,1]的矩阵，无参数。 <br>
#  第二层conv2D： （5*5*1+1）* 6 = 26 * 6 = 156, 其中5*5是kernel size，1是上一层遗留通道数，6是卷积核个数。 <br>
#  第三层池化使用[2,2]矩形，步长行列都是2 （没有1*1tensor被重复池化），没有学习项，无参数<br>
#  第四层conv2D: （5*5*6+1）* 16 = 129 * 16 = 2416, 其中5*5是kernel size，6是上一层遗留通道数，16是卷积核个数。 <br>
#  第五层池化使用[2,2]矩形，步长行列都是2 （没有1*1tensor被重复池化），没有学习项，无参数 <br>
# 
#  第六层压扁，无参数        <br>
#  
#  第七层Dense：（hidden）    400 * 120 + 120 = 48000 +120 = 48120 <br>
#  第八层Dense： (hidden）  120 * 84 + 84 = 10080 +84 = 10164 <br>
#  第九层Dense：    84 * 10 + 10 = 840 +10 = 850 <br>
#  
#  所以总共有61706个参数。
#             

# # LeNet5 编译运行

# In[11]:


model.compile(loss = 'categorical_crossentropy',optimizer='adam',metrics = ['accuracy'])
model.fit(X0,YY0,epochs = 10,batch_size = 200,validation_data=[X1,YY1]) 


# Accuracy 高达0.9917！

# # 改进？变形？

# 1. 通过google得知原先paper里的LeNet用的是Tanh作为activation function 而不是Relu，而后期人们发现 Relu会有更高的分类精度(对于MNIST)并且更简单。

# 2. 还可以考虑的变量：parameters <br>
#     - 卷积核大小
#     - 卷积核个数
#     - 池化规格
#     - same padding or valid padding
#     - optimal number of parameters, hidden layers etc...

# 3. Dropout？<br>
#    通过阅读资料，Dropout 适用于避免过度拟合（Overfitting），而LeNet高达99.17%的accuracy说明并不存在这一问题，Dropout不必要。
