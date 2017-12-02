
# coding: utf-8

# In[14]:

import os
import sys
os.chdir(r"C:\spark\spark-files")
os.curdir

if 'SPARK_HOME' not in os.environ:
    os.environ['SPARK_HOME'] = 'C:\spark'

SPARK_HOME = os.environ['SPARK_HOME']

sys.path.insert(0,os.path.join(SPARK_HOME,"python"))
sys.path.insert(0,os.path.join(SPARK_HOME,"python","lib"))
sys.path.insert(0,os.path.join(SPARK_HOME,"python","lib","pyspark.zip"))
sys.path.insert(0,os.path.join(SPARK_HOME,"python","lib","py4j-0.10.4-src.zip"))

from pyspark import SparkContext
from pyspark import SparkConf


# In[15]:

conf=SparkConf()
conf.set("spark.executor.memory", "1g")
conf.set("spark.cores.max", "2")

conf.setAppName("DigiFest")

sc = SparkContext('local[2]', conf=conf)


# In[28]:

tweetData = sc.textFile("tweets.csv")

tweetText=tweetData.map(lambda line: line.split(",")[1])
tweetText.take(2)

from pyspark.mllib.feature import HashingTF
from pyspark.mllib.feature import IDF

hashingTF = HashingTF()
tf = hashingTF.transform(tweetText)
tf.cache()
idf = IDF(minDocFreq=2).fit(tf)
tfidf = idf.transform(tf)
tfidf.cache()
tfidf.count()

xformedData=tweetData.zip(tfidf)
xformedData.cache()
xformedData.collect()[0]

from pyspark.mllib.regression import LabeledPoint
def convertToLabeledPoint(inVal) :
    origAttr=inVal[0].split(",")
    sentiment = 0.0 if origAttr[0] == "positive" else 1.0
    return LabeledPoint(sentiment, inVal[1])

tweetLp=xformedData.map(convertToLabeledPoint)
tweetLp.cache()
tweetLp.collect()

from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
model = NaiveBayes.train(tweetLp, 1.0)
predictionAndLabel = tweetLp.map(lambda p:     (float(model.predict(p.features)), float(p.label)))
predictionAndLabel.collect()

#Form confusion matrix
from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)
predDF = sqlContext.createDataFrame(predictionAndLabel.collect(),                 ["prediction","label"])
predDF.groupBy("label","prediction").count().show()

#save the model
#model.save(sc,"TweetsSentimentModel")
import pickle
with open('IRModel', 'wb') as f:
    pickle.dump(model, f)


# In[ ]:

import pickle
from pyspark.mllib.classification import  NaiveBayesModel

with open('IRModel', 'rb') as f:
    loadedModel = pickle.load(f)
    
from pyspark.streaming import StreamingContext
streamContext = StreamingContext(sc,1)

tweets = streamContext.socketTextStream("localhost", 9000)

bc_model = sc.broadcast(loadedModel)

from pyspark.mllib.feature import HashingTF
from pyspark.mllib.feature import IDF


def predictSentiment(tweetText):
    nbModel=bc_model.value
    
    hashingTF = HashingTF()
    tf = hashingTF.transform(tweetText)
    tf.cache()
    idf = IDF(minDocFreq=2).fit(tf)
    tfidf = idf.transform(tf)
    tfidf.cache()
    prediction=nbModel.predict(tfidf)
    print "Predictions for this window :"
    for i in range(0,prediction.count()):
        print prediction.collect()[i], tweetText.collect()[i]

tweets.foreachRDD(predictSentiment)

streamContext.start()
streamContext.stop()


# In[ ]:



