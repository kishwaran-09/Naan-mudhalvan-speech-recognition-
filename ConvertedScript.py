#!/usr/bin/env python
# coding: utf-8

# In[3]:


from google.colab import drive
drive.mount('/content/drive')
import os

# List the contents of the root folder to see if MyDrive exists
os.listdir('/content/drive/mydrive/file/Dataset')

# Now list the contents of MyDrive to find your folder
os.listdir('/content/drive/MyDrive/file/Dataset')

