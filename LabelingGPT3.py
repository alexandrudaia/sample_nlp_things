#!/usr/bin/env python
# coding: utf-8

# In[139]:


import pandas as pd

import numpy as np
train=pd.read_excel('Full_Sentences_Dataset_GP2.xlsx')


# In[140]:


train.head(3)


# In[135]:


import os
import openai

openai.api_key ="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"

start_sequence = "<<"
restart_sequence = "\n\nQ: "

response = openai.Completion.create(
  model="text-davinci-003",
  prompt="Give me the SNP-disease or trait associations also find rsID, disease, allele, genotype, odds ratio, p-value, ethnicity, sex of the next paragraph :Individually, each of these polymorphisms only moderately predisposes to type 2 diabetes with odds ratios (ORs) ranging from ~1.15 for the Lys23 variant of the KCNJ11 variant to ~1.50 for the rs7903146 variant of TCF7L2.",
  temperature=0.0,
  max_tokens=100,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0,
    stop=">>"
    
 
)


# In[141]:


labels=[]
for row in  range(100):
    current_row=train['Sentence'].iloc[row]
    prpt="Give me the SNP-disease or trait associations also find rsID, disease, allele, genotype, odds ratio, p-value, ethnicity, sex of the next paragraph : " +str(current_row)
    #print("question " , prpt)
    start_sequence = "\nA:"
    restart_sequence = "\n\nQ: "
    openai.api_key ="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"
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prpt,
        temperature=0.0,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=">>"
    )
    print(response["choices"][0]["text"])
    labels.append(response["choices"][0]["text"])
    


# In[149]:


labels=[l[2:] for l in labels]


# In[150]:


temp=train[:100] 


# In[151]:


temp['davinci_label_one_shot']=labels


# In[152]:


temp


# In[154]:


temp.to_csv('davinci_label.csv',index=False)


# In[155]:


labels=[]
for row in  range(100):
    current_row=train['Sentence'].iloc[row]
    prpt="Please give me the sex of subjects from next paragraph: " +str(current_row)
    #print("question " , prpt)
    start_sequence = "\nA:"
    restart_sequence = "\n\nQ: "
    openai.api_key ="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"
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prpt,
        temperature=0.0,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=">>"
    )
    print(response["choices"][0]["text"])
    labels.append(response["choices"][0]["text"])
    


# In[ ]:




