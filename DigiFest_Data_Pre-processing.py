
# coding: utf-8

# In[ ]:

#parsing
import json
import pandas as pd
from pandas import DataFrame

tweets_data_path='again.txt'
tweets_data = []
tweets_file = open(tweets_data_path, "r")
for line in tweets_file:
    try:
        tweet = json.loads(line)
        tweets_data.append({"msg":tweet['text'],"user":tweet['user']['screen_name']})
    except:
        print("here")
        continue


df=DataFrame(tweets_data)
print(df)

df.to_csv('out1.csv', sep='\t')


# In[ ]:

#pre-processing
import pandas as pd
df=pd.read_csv('tweets.csv')
df1=df[['msg']]


from nltk import word_tokenize

for i in range(0,len(df1)):
        df1.iloc[i][0]=word_tokenize(df1.iloc[i][0])
        
for i in range(0,len(df1)):
        filtered_sentence=[]
        for w in df1.iloc[i][0]:
            filtered_sentence.append(w.lower())
        df1.iloc[i][0]=filtered_sentence 

from nltk.corpus import stopwords
stop_words=set(stopwords.words('english'))

for i in range(0,len(df1)):
        filtered_sentence=[]
        for w in df1.iloc[i][0]:
            if w not in stop_words:
                filtered_sentence.append(w)
        df1.iloc[i][0]=filtered_sentence


from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer=WordNetLemmatizer()

for i in range(0,len(df1)):
        filtered_sentence=[]
        for w in df1.iloc[i][0]:
            filtered_sentence.append(wordnet_lemmatizer.lemmatize(w))
        df1.iloc[i][0]=filtered_sentence   


        
import re
df2=df[['pnr']]

for i in range(0,len(df1)):
        for key in df1.iloc[i][0]:
            str1=""
            txt=re.compile(r'^[0-9]{10}$').findall(key)
            if(txt):
                for j in txt:
                    str1+=j
                df2.iloc[i][0]=str1



for i in range(0,len(df1)):
        temp=[]
        for key in df1.iloc[i][0]:
            str1=""
            txt=re.compile("^[a-z]*$").findall(key)
            if(txt):
                for j in txt:
                    str1+=j
                temp.append(str1)
        df1.iloc[i][0]=temp 
        
df3=pd.DataFrame(columns=['msg','usr','pnr','output'],index=range(0,12))

for i in range(0,len(df)):
    df3.iloc[i][0]=df1.iloc[i][0]
    df3.iloc[i][1]=df.iloc[i][1]
    df3.iloc[i][2]=df2.iloc[i][0]
    df3.iloc[i][3]=df.iloc[i][3]

