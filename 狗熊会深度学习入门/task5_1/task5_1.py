#!/usr/bin/env python
# coding: utf-8

# # 数据生成
# 
# 为什么需要使用数据生成器？ 答：数据占内存，可能没有足够的内存一下子存储所有数据，生成器可以把小量数据多批次读入内存。虽然花费了时间，但是能够处理很大的数据量。

# ## 不使用DA

# In[1]:


from keras.preprocessing.image import ImageDataGenerator #从keras导入需要需要的包

IMSIZE=128
validation_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
    '/clubear/Lecture 5.1 - Batch Normalization/data/CatDog/validation',
    target_size=(IMSIZE, IMSIZE), #不管数据里的图像大小，在input统一shape
    batch_size=200, #每一个batch只读取200个数据 
    class_mode='categorical') #分类问题
#简单的制作validation data，rescale 是因为TensorFlow需要倒入的数据值在0-1之间，而我们拥有的图片是0-255

train_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
    '/clubear/Lecture 5.1 - Batch Normalization/data/CatDog/train',
    target_size=(IMSIZE, IMSIZE),
    batch_size=200,
    class_mode='categorical')


# In[2]:


#数据展示
from matplotlib import pyplot as plt

plt.figure()
fig,ax = plt.subplots(2,5)
fig.set_figheight(6)
fig.set_figwidth(15)
ax=ax.flatten()
X,Y=next(train_generator) #next是一个简单的方程将训练数据添加到X，Y上
for i in range(10): ax[i].imshow(X[i,:,:,:])


# ## 使用DA 

# In[3]:


train_generator1 = ImageDataGenerator(
    rescale=1./255, #tensorFlow要求
    shear_range=0.5, #扭曲变形
    rotation_range=30, #旋转
    zoom_range=0.2, #放大
    width_shift_range=0.2, #水平平移
    height_shift_range=0.2 , #垂直评议
    horizontal_flip=True ).flow_from_directory(
    '/clubear/Lecture 5.1 - Batch Normalization/data/CatDog/train',
    target_size=(IMSIZE, IMSIZE),
    batch_size=200,
    class_mode='categorical')
#这里使用了Data Augmentation技术，即将图片变形，使得1）数据量更大；2）在现实生活中的图片不可能是全部正面，通过数据增强可以
#使我们的模型识别到更多变形的图像
#train现在存为了train_generator1


# In[4]:


#数据展示
from matplotlib import pyplot as plt

plt.figure()
fig,ax = plt.subplots(2,5)
fig.set_figheight(6)
fig.set_figwidth(15)
ax=ax.flatten()
X,Y=next(train_generator1) #X和Y现在从train_generator1中取
for i in range(10): ax[i].imshow(X[i,:,:,:])


# # 模型搭建 

# In[5]:


from keras.layers import Conv2D,Dense,Flatten,Input,MaxPooling2D,Dropout 
from keras import Model

input_layer = Input([IMSIZE,IMSIZE,3])
x = input_layer
for _ in range(5):
    x = Conv2D(20,[3,3], activation = 'relu')(x) 
    x = MaxPooling2D(pool_size = [2,2], strides = [2,2])(x)    

x = Flatten()(x)   
x = Dense(84,activation = 'relu')(x)
x = Dense(2,activation = 'softmax')(x)
output_layer=x
model=Model(input_layer,output_layer)
model.summary()


# ### 参数解释
# 
#  第一层input layer，无参数。 <br>
#  for loop中含有5个卷积层和5个池化层，除去第一个卷积层，其他卷积层参数为：<br>
#      conv2D： （3 * 3 * 20+1）* 20 = 181 * 20 = 3620 , 其中3 * 3 是kernel size，20是上一层遗留通道数，20是卷积核个数。 <br>
#      池化使用[2,2]矩形，步长行列都是2，没有学习项，无参数<br>
#  for loop的第一个卷积层参数计算如下： <br>
#      (3 * 3 * 3 +1）* 20 = 28 * 20 = 560 , 其中3 * 3 是kernel size，3是上一层遗留通道数，20是卷积核个数。 <br>
# 
# 
#  压扁，无参数        <br>
#  
#  Dense1： (hidden）  80 * 84 + 84 = 6720 +84 = 6804 <br>
#  Dense2：    84 * 2 + 2 = 168 + 2 = 170 <br>
#    所以总共有22014个参数。<br>
#    没有non-trainable parameters因为BN没有被用到

# In[6]:


from keras.optimizers import Adam
model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.01),metrics=['accuracy'])
model.fit_generator(train_generator,epochs=4,validation_data=validation_generator)


# ## 使用BN，无DA

# In[7]:


from keras.layers import BatchNormalization
input_layer = Input([IMSIZE,IMSIZE,3])
x = input_layer
x=BatchNormalization()(x) #使用BatchNormalisation，对input的一个batch的数据每一层进行normalisation
for _ in range(5):
    x = Conv2D(20,[3,3], activation = 'relu')(x) 
    x = MaxPooling2D(pool_size = [2,2], strides = [2,2])(x)    

x = Flatten()(x)   
x = Dense(84,activation = 'relu')(x)
x = Dense(2,activation = 'softmax')(x)
output_layer=x
model2=Model(input_layer,output_layer)
model2.summary()


# ### 参数解释
# 
#  第一层input layer，无参数。 <br>
#  第二层BatchNormalisation中，因为每一层/通道都有4个参数，3 * 4 = 12； 但是miu 和 sigma不需要训练（因为这两个参数可以从数据总直接计算，和模型学习没有关系），所以 non-trainable params 有 3 * 2 = 6个。
#  for loop中含有5个卷积层和5个池化层，除去第一个卷积层，其他卷积层参数为：<br>
#      conv2D： （3 * 3 * 20+1）* 20 = 181 * 20 = 3620 , 其中3 * 3 是kernel size，20是上一层遗留通道数，20是卷积核个数。 <br>
#      池化使用[2,2]矩形，步长行列都是2，没有学习项，无参数<br>
#  for loop的第一个卷积层参数计算如下： <br>
#      (3 * 3 * 3 +1）* 20 = 28 * 20 = 560 , 其中3 * 3 是kernel size，3是上一层遗留通道数，20是卷积核个数。 <br>
# 
# 
#  压扁，无参数        <br>
#  
#  Dense1： (hidden）  80 * 84 + 84 = 6720 +84 = 6804 <br>
#  Dense2：    84 * 2 + 2 = 168 + 2 = 170 <br>
#    所以总共有22026个参数，比没有BN的模型多了12个参数，其中包含6个non-trainable参数。<br>
#    没有non-trainable parameters因为BN没有被用到
# 

# In[8]:


model2.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.01),metrics=['accuracy'])
model2.fit_generator(train_generator,epochs=4,validation_data=validation_generator)


# ## 使用DA和BN

# In[9]:


model2.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.01),metrics=['accuracy'])
model2.fit_generator(train_generator1,epochs=4,validation_data=validation_generator)
#模型用的是加上了BN的LeNet，训练数据用的是Data Augmentation 以后的train_generator1


# # 总结

# 普通仿LeNet：accuracy 为 0.50 左右，和一个random guess差不多。
# <br>模型with BN：最后一个epoch的accuracy 为 0.67，呈增加趋势，如果epoch再高也许还有更高的精度。
# <br>模型with DA & BN：平均accruacy（在4个epoch里）达到约0.67， 比只有BN的时候要更精确。
