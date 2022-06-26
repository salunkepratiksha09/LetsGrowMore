#!/usr/bin/env python
# coding: utf-8

# # LetsGrowMore Task2
# 
# Name:Pratiksha Hemraj Salunke
# 
# Task 2: Image to Pencil Sketch With Python
# 
# 
# 

# In[61]:


from google.colab import drive
drive.mount('/content/drive')


# In[62]:


from google.colab import files
from IPython.display import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import cv2


# In[63]:


image = cv2.imread('/content/drive/MyDrive/rose-true-pink.jpg')


# In[64]:


from google.colab.patches import cv2_imshow


# In[66]:


photo = cv2_imshow(image)


# In[67]:


gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# In[68]:


inverted = 255-gray_image


# In[69]:


blur = cv2.GaussianBlur(inverted, (21, 21), 0)


# In[70]:


invertedblur = 255-blur


# In[71]:


sketch = cv2.divide(gray_image, invertedblur, scale=256.0)


# In[72]:


cv2.imwrite("outputImage.png", sketch)


# In[73]:


cv2_imshow(sketch)


# In[74]:


cv2.waitKey(0)


# In[75]:


image = cv2.imread('/content/drive/MyDrive/rose-true-pink.jpg',0)


# In[76]:


photo = cv2_imshow(image)


# THANK YOU!
