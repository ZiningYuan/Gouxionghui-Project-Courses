#!/usr/bin/env python
# coding: utf-8

# In[26]:


from PIL import Image
from glob import glob
import numpy as np
from keras.utils import to_categorical 
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Flatten,Input,Activation,Conv2D,MaxPooling2D,BatchNormalization
from keras import Model
import re 


# In[28]:


images = glob("/clubear/Lecture 3.3 - Case Study (CIFA10)/cifar10/*.*")
N=len(images)
imsize=32


# In[29]:


print(images[0])  #可以看到这个path里有很多个“/”所以不能只用split将图片的分类取出来


# In[44]:


Y=[]
X=np.zeros([N,imsize,imsize,3])
for i in range(N):
    Im=Image.open(images[i])
    Im=np.array(Im)/255
    X[i]=Im
    filename = re.split('/｜/｜.|-|/|/',images[i])[5] #分离多个符号的function
    cat=filename.split("_")[0] #然后再提取名字
    Y.append(cat)


# In[45]:


unique=list(set(Y))
DICT={}
for i in range(len(unique)): DICT[unique[i]]=i
YY=np.zeros(N)
for i in range(N):
    YY[i]=DICT[Y[i]]


# In[46]:


DICT #10个分类


# In[47]:


fig,ax=plt.subplots(2,5)
fig.set_figwidth(20)
fig.set_figheight(10)
ax=ax.flatten()
for i in range(len(ax)):
    Im=X[YY==i][0]
    ax[i].imshow(Im)
    ax[i].set_title(unique[i])


# In[51]:


Y=to_categorical(YY) #one-hot变量
X0,X1,Y0,Y1=train_test_split(X,Y,test_size=0.3,random_state=1)


# In[65]:


input_size=[imsize,imsize,3]
input_layer=Input(input_size)
x=input_layer
x=Conv2D(32,[3,3],padding = "same", activation = 'relu')(x) 
x=Conv2D(32,[3,3],padding = "same", activation = 'relu')(x)  
x = MaxPooling2D(pool_size = [2,2])(x) 
x=Conv2D(64,[2,2],padding = "same", activation = 'relu')(x) 
x=Conv2D(64,[2,2],padding = "same", activation = 'relu')(x) 
x = MaxPooling2D(pool_size = [2,2])(x) 
x=Conv2D(128,[2,2],padding = "same", activation = 'relu')(x) 
x=Conv2D(128,[2,2],padding = "same", activation = 'relu')(x) 
x = MaxPooling2D(pool_size = [2,2])(x) 
x=Flatten()(x)
x=Dense(10,activation='softmax')(x) #dense只能是10因为有十个分类
output_layer=x
model=Model(input_layer,output_layer)
model.summary()
#感觉一般比较常见的filter个数是32，64，128，有参考VGG模型搭建过程


# #### 参数解释

# 第一层input layer，无参数。 <br>
# 第二层conv2D： （3*3*3+1）* 32 = 28 * 32 = 896, 其中3*3是kernel size，3是上一层遗留通道数，32是卷积核个数。 <br>
# 第三层conv2D： （3*3*32+1）* 32 = 289 * 32 = 9248, 其中3*3是kernel size，32是上一层遗留通道数，32是卷积核个数。<br> 
# 第四层池化没有学习项，无参数<br>
# 第五层conv2D: （2*2*32+1）* 64 = 129 * 64 = 8256, 其中2*2是kernel size，32是上一层遗留通道数，64是卷积核个数。 <br>
# 第六层conv2D: （2*2*64+1）* 64 = 257 * 64 = 16448, 其中2*2是kernel size，64是上一层遗留通道数，64是卷积核个数。 <br>
# 第七层池化没有学习项，无参数    <br>
# 第八层conv2D: （2*2*64+1）* 128 = 257 * 128 =32896, 其中2*2是kernel size，64是上一层遗留通道数，128是卷积核个数。   <br>
# 第九层conv2D: （2*2*128+1）* 128 = 513 * 128 = 65664, 其中2*2是kernel size，128是上一层遗留通道数，128是卷积核个数。  <br>
# 第十层池化没有学习项，无参数          <br>
# 第十一层压扁，无参数        <br>
# 第十二层Dense：    2048 * 10 + 10 = 20480 +10 = 20490 <br>
# 所以总共有153898个参数。
#            

# In[66]:


from keras.optimizers import Adam
model.compile(optimizer = Adam(0.0001),loss = 'categorical_crossentropy',metrics = ['accuracy'])
model.fit(X0,Y0,validation_data=(X1,Y1),batch_size=100,epochs=30)


# #### accuracy是0.73， 参数比Lecture 3.3中多但是accuracy低了0.03

# # 与其他模型对比

# ### 这是Lecture 3.3中给出的模型

# In[75]:


input_size=[imsize,imsize,3]
input_layer=Input(input_size)
x=input_layer
x=Conv2D(100,[2,2],padding = "same", activation = 'relu')(x)  
x = MaxPooling2D(pool_size = [2,2])(x) 
x=Conv2D(100,[2,2],padding = "same", activation = 'relu')(x) 
x = MaxPooling2D(pool_size = [2,2])(x) 
x=Conv2D(100,[2,2],padding = "same", activation = 'relu')(x) 
x = MaxPooling2D(pool_size = [2,2])(x) 
x=Flatten()(x)

x=Dense(10,activation='softmax')(x)
output_layer=x
model1=Model(input_layer,output_layer)
model1.summary()


# In[68]:


from keras.optimizers import Adam
model1.compile(optimizer = Adam(0.00001),loss = 'categorical_crossentropy',metrics = ['accuracy'])
model1.fit(X0,Y0,validation_data=(X1,Y1),batch_size=100,epochs=30)


# ####  Learning rate是0.00001，得到的accuracy只有0.46。个人怀疑是learning rate太小，于是尝试了0.001，如下。

# In[72]:


from keras.optimizers import Adam
model1.compile(optimizer = Adam(0.0001),loss = 'categorical_crossentropy',metrics = ['accuracy'])
model1.fit(X0,Y0,validation_data=(X1,Y1),batch_size=100,epochs=30)


# #### 同样的model，现在的accuracy是0.7313，比lecture3.3中的0.7662低。 但应该不影响模型搭建。

# ### 和自己写的第一个模型strcture差不多，差别在于将卷积核改成了valid

# In[73]:


input_size=[imsize,imsize,3]
input_layer=Input(input_size)
x=input_layer
x=Conv2D(32,[3,3],padding = "valid", activation = 'relu')(x) 
x=Conv2D(32,[3,3],padding = "valid", activation = 'relu')(x)  
x = MaxPooling2D(pool_size = [2,2])(x) 
x=Conv2D(64,[2,2],padding = "valid", activation = 'relu')(x) 
x=Conv2D(64,[2,2],padding = "valid", activation = 'relu')(x) 
x = MaxPooling2D(pool_size = [2,2])(x) 
x=Conv2D(128,[2,2],padding = "valid", activation = 'relu')(x) 
x=Conv2D(128,[2,2],padding = "valid", activation = 'relu')(x) 
x = MaxPooling2D(pool_size = [2,2])(x) 
x=Flatten()(x)
x=Dense(10,activation='softmax')(x) #dense只能是10因为有十个分类
output_layer=x
model2=Model(input_layer,output_layer)
model2.summary()


# ### 参数解释

# 前面都和第一个模型相同，唯一的差别在于压扁以后的vector只有512 （因为valid比same处理得出的矩阵要小）<br>
# 第十二层Dense：    512 * 10 + 10 = 5120 +10 = 5130 <br>
# 所以总共有153538个参数。

# In[77]:


model2.compile(optimizer = Adam(0.0001),loss = 'categorical_crossentropy',metrics = ['accuracy'])
model2.fit(X0,Y0,validation_data=(X1,Y1),batch_size=100,epochs=30)


# #### 可以看到accuracy是0.79，比相同模型same padding高，但再run一次得到的accruacy不必same padding高，所以应该不会造成太大差别。

# In[ ]:




