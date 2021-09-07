# IPA Topic and Sentiment

Author: Reean Liao 
Date: August 2021

This is a recruitment assessment for an Australian company - hopefully you will find some inspiration.

```python
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import re
import read_xml as read_xml

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.tag import pos_tag
from nltk.sentiment import SentimentIntensityAnalyzer
#initialized stopwords and stemmer
stop_words = set(stopwords.words('english'))
stemmer=SnowballStemmer('english')
lemmatizer=WordNetLemmatizer()

from wordcloud import WordCloud, ImageColorGenerator

from gensim.utils import simple_preprocess
from gensim import corpora, models
import pyLDAvis.gensim

import pickle
import joblib

#download these if missing or if it's not up to date
#nltk.download('wordnet') 
#nltk.download('stopwords')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('vader_lexicon')
```


```python
pd.options.mode.chained_assignment = None
```

## Table of Content
* [Problem Statement](#Problem-Statement)

* [Data Exploration](#Data-Exploration)
    * [Structure Base Table](#Structure-Base-Table)
    * [Additional Features](#Additional-Features)

* [Insights](#Insights)
    * [Overall Trend](#Overall-Trend)
    * [Topic Analysis](#Topic-Analysis)
    * [Sentiment Analysis](#Sentiment-Analysis)

## Problem Statement

You are approached by an artisan brewer client who is considering adding a new hoppy beer to their range - India pale ale (IPA). The client's marketing team would like to know **how consumers feel about this product and how the attitude toward IPA beer has changed over time**. They identified a question-answering website (stack exchange) as a source of genuine opinions. Use the XML datasets found in ./dataset/beer.stackexchange.com/ for this exercise. 

## Data Exploration
### Structure Base Table
[Back to top](#Table-of-Content)


```python
df_comment=read_xml.read_xml('dataset/beer.stackexchange.com/Comments.xml')
df_comment.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>PostId</th>
      <th>Score</th>
      <th>Text</th>
      <th>CreationDate</th>
      <th>UserId</th>
      <th>ContentLicense</th>
      <th>UserDisplayName</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>This is a matter of taste I guess.</td>
      <td>2014-01-21T20:34:28.740</td>
      <td>10</td>
      <td>CC BY-SA 3.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>16</td>
      <td>4</td>
      <td>This question appears to be off-topic because ...</td>
      <td>2014-01-21T20:52:14.490</td>
      <td>39</td>
      <td>CC BY-SA 3.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>16</td>
      <td>8</td>
      <td>Obviously, you should pee into the empty beer ...</td>
      <td>2014-01-21T20:55:44.723</td>
      <td>41</td>
      <td>CC BY-SA 3.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>5</td>
      <td>3</td>
      <td>Doesn't this depend on the type of beer as well?</td>
      <td>2014-01-21T20:57:44.653</td>
      <td>43</td>
      <td>CC BY-SA 3.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7</td>
      <td>19</td>
      <td>4</td>
      <td>I retracted my close vote after reading this a...</td>
      <td>2014-01-21T20:57:47.373</td>
      <td>20</td>
      <td>CC BY-SA 3.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



Column | Description
:---|:---
`Id` | Id of the comment
`PostId` | Id of the post the comment is posted to (could be answer or question)
`Score` | Number of upvotes (if any) that the comment received
`Text` | Comment body
`CreationDate` | Comment creation date
`UserId` | UserId (null if user is deleted)
`ContentLicense` | Content distribution license
`UserDisplayName` | User's display name


```python
df_post=read_xml.read_xml('dataset/beer.stackexchange.com/Posts.xml')
df_post.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>PostTypeId</th>
      <th>AcceptedAnswerId</th>
      <th>CreationDate</th>
      <th>Score</th>
      <th>ViewCount</th>
      <th>Body</th>
      <th>OwnerUserId</th>
      <th>LastEditorUserId</th>
      <th>LastEditDate</th>
      <th>...</th>
      <th>Tags</th>
      <th>AnswerCount</th>
      <th>CommentCount</th>
      <th>ContentLicense</th>
      <th>ParentId</th>
      <th>FavoriteCount</th>
      <th>ClosedDate</th>
      <th>LastEditorDisplayName</th>
      <th>OwnerDisplayName</th>
      <th>CommunityOwnedDate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>2014-01-21T20:26:05.383</td>
      <td>20</td>
      <td>2346</td>
      <td>&lt;p&gt;I was offered a beer the other day that was...</td>
      <td>7</td>
      <td>8</td>
      <td>2014-01-21T22:04:34.977</td>
      <td>...</td>
      <td>&lt;hops&gt;</td>
      <td>1</td>
      <td>0</td>
      <td>CC BY-SA 3.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>175</td>
      <td>2014-01-21T20:27:29.797</td>
      <td>24</td>
      <td>3072</td>
      <td>&lt;p&gt;As far as we know, when did humans first br...</td>
      <td>7</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>&lt;history&gt;</td>
      <td>4</td>
      <td>3</td>
      <td>CC BY-SA 3.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>13</td>
      <td>2014-01-21T20:30:17.437</td>
      <td>21</td>
      <td>517</td>
      <td>&lt;p&gt;How is low/no alcohol beer made? I'm assumi...</td>
      <td>7</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>&lt;brewing&gt;</td>
      <td>3</td>
      <td>0</td>
      <td>CC BY-SA 3.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 22 columns</p>
</div>



Column | Description
:---|:---
`Id` | Id of the post
`PostTypeId` | Id of the Post type (1 = Question; 2 = Answer; 3 = Orphaned tag wiki; 4 = Tag wiki excerpt; 5 = Tag wiki; 6 = Moderator nomination;7 = "Wiki placeholder" (seems to election related);8 = Privilege wiki
`AcceptedAnswerId` | Id of the accepted answer (only present if PostTypeId=1, i.e. a question)
`CreationDate` | Post creation date
`Score` | Number of upvotes the post received
`ViewCount` | Number of views from registered user
`Body` | Body of the post (rendered as HTML, not Markdown)
`OwnerUserId` | Owner of the thread, only present if user is not deleted (always -1 for tag wiki entries, i.e. the community user owns them)
`LastEditorUserId` | Id of the last editor
`LastEditDate` | Last edited date of the post
`LastActivityDate` | Last post activity date
`Title` | Title of the post (only available for questions)
`Tags` | Tags inserted in by the owner of the post (only available for questions)
`AnswerCount` | Number of answers to a question
`CommentCount` | Number of comments to a post
`ContentLicense` | Content distribution license
`ParentId` | Id of the question (only present if PostTypeId=2, i.e. an answer)
`FavoriteCount` | Number of times a question is bookmarked
`ClosedDate` | Date the post is closed
`LastEditorDisplayName` | Last editor's display name
`OwnerDisplayName` | Thread owner's display name
`CommunityOwnedDate` | Community set up date (only present if the post is a community wiki)

These two tables (Posts & Comments) offer a lot of information to uncover opinion and attitudes to IPA (or other types of beer), so we will use them to form a basis for our topic analysis and sentiment analysis. 

In general, people can ask beer type (e.g. IPA, stout, lager) specific questions, or just generic beer questions (e.g. temperature, history), and people might answer back with beer type specific answers or generic answers. So here, we will not differentiate the post types (be it questions, answers or even comments). We will gather the key columns from the Posts and Comments tables, and build a combined table.


```python
#Make a smaller combined table
cols_to_keep_post=['Id','PostTypeId','AcceptedAnswerId','CreationDate','Score','ViewCount', 'Body','Title','Tags']
df_post_lite=df_post[cols_to_keep_post].copy()

cols_to_keep_comment=['Id','PostId','Score','Text','CreationDate']
df_comment_lite=df_comment[cols_to_keep_comment].copy()
df_comment_lite.columns=['Id','PostId','Score','Body','CreationDate']
df_comment_lite['Id']='C_'+ df_comment_lite['Id'] #Add a prefix to differentiate the ID
df_comment_lite['PostTypeId']='9' # 9 for comment

df=pd.concat([df_post_lite,df_comment_lite], axis=0, sort=False)
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 7200 entries, 0 to 3620
    Data columns (total 10 columns):
    Id                  7200 non-null object
    PostTypeId          7200 non-null object
    AcceptedAnswerId    636 non-null object
    CreationDate        7200 non-null object
    Score               7200 non-null object
    ViewCount           1065 non-null object
    Body                7200 non-null object
    Title               1065 non-null object
    Tags                1065 non-null object
    PostId              3621 non-null object
    dtypes: object(10)
    memory usage: 618.8+ KB


The combined table will have 7200 rows (union from the two tables) and 10 columns.

### Additional Features
[Back to top](#Table-of-Content)


```python
#Create a full verbatim of the text
df['Body'].fillna('', inplace=True)
df['Title'].fillna('', inplace=True)
df['Tags'].fillna('', inplace=True)
df['Verbatim']=df['Body']+' '+df['Title']+' '+df['Tags']
df['Verbatim']=df['Verbatim'].str.lower()

#Create keyword base indicator
df['IPA']=np.where(df['Verbatim'].str.contains(' ipa'), 1,
                   np.where(df['Verbatim'].str.contains('indian pale ale'), 1, 0))
df['Ale']=np.where((df['Verbatim'].str.contains('ale'))& (df['IPA']==0), 1, 0)
df['Lager']=np.where(df['Verbatim'].str.contains('lager'), 1, 0)
df['Pilsner']=np.where(df['Verbatim'].str.contains('pilsner'), 1, 0)
df['Stout']=np.where(df['Verbatim'].str.contains('stout'), 1, 0)
df['Wheat']=np.where(df['Verbatim'].str.contains('wheat'), 1, 0)
df['Belgium']=np.where(df['Verbatim'].str.contains('belgium'), 1, 0)

#Date and Month variable
df['CreationDate'] = pd.to_datetime(df['CreationDate'])
df['CreationMonth'] = df['CreationDate'].dt.to_period('M')
```


```python
#Remove records with less than 30 character - as a rule of thumb
df=df[df['Verbatim'].map(len) >= 30]
```

## Insights
### Overall Trend
[Back to top](#Table-of-Content)


```python
df_count=df[['CreationMonth','Id']].groupby('CreationMonth').count().reset_index()
df_count.plot(x='CreationMonth', figsize=(14,6));
```


![png](IPA_forum_volume_trend.png)


Aside from the initial rush (probably due to the platform first launched) and a few spikes around 2016-2017, beer topic seems to have a relatively consistent trend. From 2014 to recent years, there have been fairly stable volume of posts (be it questions or answers). However, there does seem to be a slightly decreasing trend in this topic. It's possible that younger generations are not drinking as much or are not as passionate about beers as the older generation. It's also possible that internet users are just not using stack exchange as much as they used to.


```python
df_count_type=df[['CreationMonth', 'IPA', 'Ale', 'Lager', 'Pilsner', 'Stout', 'Wheat',
                  'Belgium']][df['CreationDate'] >= '2014-03-01'].groupby('CreationMonth').sum().reset_index()
df_count_type.plot(x='CreationMonth', figsize=(14,6));
```


![png](IPA_forum_volume_by_type.png)


At a rough glance, Ale seems to be the most regularly mentioned beer type, followed by IPA; after that, Lager is also a popular topic.

### Topic Analysis
[Back to top](#Table-of-Content)

Since our goal is to understand people's opinion about IPA, we will do a further deep dive into the relevant topics when it comes to IPA.


```python
df_ipa=df[df['IPA']==1]
```


```python
#show top frequency words
def freq_words(x, terms = 30): 
    
    all_words = ' '.join([text for text in x]) 
    all_words = all_words.split() 
  
    fre_dist = nltk.FreqDist(all_words) 
    df_words = pd.DataFrame({'word':list(fre_dist.keys()), 'count':list(fre_dist.values())}) 
  
    # selecting top n most frequent words 
    d = df_words.nlargest(columns="count", n = terms) 
  
    # visualize words and frequencies
    plt.figure(figsize=(12,8)) 
    ax = sns.barplot(data=d, x= "count", y = "word") 
    ax.set(ylabel = 'Word') 
    plt.show()
```


```python
freq_words(df_ipa['Verbatim'], terms=40)
```


![png](IPA_top_words.png)


The most frequent words are mostly comprise stop words; but it's good to update the stop words list to cater to the specific context we are working on - IPA beer.


```python
#add extra stop words seen
stop_words.update(['beer','beers','<a','ipa','rel="nofollow"','one','ipas','would','also','even','may'])
#also added ones seen after the cleansing
```


```python
#function to find the part-of-speech tag
def find_pos(token_list):
    for word, tag in pos_tag(token_list):
        if tag.startswith('N'):
            pos = 'n'
        elif tag.startswith('V'):
            pos = 'v'
        elif tag.startswith('R'):
            pos = 'r'
        else:
            pos = 'a'
    return pos
```


```python
#one stop shop function to clean up the text
def clean_text(text):
    
    cleanText=text
    
    # clean html links on the body
    html = re.compile('<[^>]*>') # remove everything inside the html tags
   
    cleanText = re.sub(html, ' ', str(cleanText))
    
    # remove stopwords and lemmatize nouns
    tokens = simple_preprocess(cleanText) 
    # remove new line, lowercase, keep alphabet, remove punctuation and special characters 
    #- all taken care by gensim's simple preprocess
    
            
    cleanTokens = [lemmatizer.lemmatize(token, 
                                        pos=find_pos([token])) for token in tokens if not token in stop_words and len(token) > 2]
    cleanText = ' '.join(cleanTokens)

    return cleanText
```


```python
df_ipa['CleanVerbatim'] = df_ipa['Verbatim'].apply(lambda x: clean_text(x))
```


```python
#freq_words(df_ipa['CleanVerbatim'], terms=40) #- use this to add to the stop word list and clean again
```

With the cleaned up verbatim, we can start generating a word cloud.


```python
cloud_text = " ".join(doc for doc in df_ipa['CleanVerbatim'])
# Create and generate a word cloud image:
wordcloud = WordCloud(background_color="white", max_words=100).generate(cloud_text)

# Display the generated image:
plt.figure(figsize=(12,8)) 
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show();
```


![png](IPA_word_cloud.png)


Based on this word cloud, we can infer, when IPA is mentioned, it's likely people are talking about:
   * IPA being one of the many **different** beer **styles**
   * IPA's characteristics of lower **alcohol** content and strong **hoppy flavour**, and how it affects **taste**
   * IPA's unique **making** or **brewing** process
   * IPA in comparison to **wine**, **ale**, **pale ale** and other types of **craft** beers
   
Aside from word cloud, we can also try using Latent Dirichlet Allocation (LDA) for topic modelling.


```python
def tokenize(list):
    lda_tokens=[]
    for i in list:
        tokens=i.split()
        #for token in tokens:
        lda_tokens.append(tokens)
            
    return lda_tokens
```


```python
#build dictionary
token_ipa=tokenize(df_ipa['CleanVerbatim'])
dictionary_ipa = corpora.Dictionary(token_ipa)
corpus_ipa = [dictionary_ipa.doc2bow(text) for text in token_ipa]
pickle.dump(corpus_ipa, open('corpus_ipa.pkl', 'wb')) #write binary
dictionary_ipa.save('dictionary_ipa.gensim')
```


```python
#apply gemsim LDA models
ldamodel_ipa = models.ldamodel.LdaModel(corpus_ipa, num_topics = 6, id2word=dictionary_ipa, passes=15)
ldamodel_ipa.save('model_ipa.gensim')
topics = ldamodel_ipa.print_topics(num_words=6)
for topic in topics:
    print(topic)
```

    (0, '0.012*"olive" + 0.011*"hop" + 0.010*"lager" + 0.010*"style" + 0.010*"taste" + 0.009*"ale"')
    (1, '0.011*"ale" + 0.010*"hop" + 0.009*"like" + 0.008*"abv" + 0.008*"wine" + 0.008*"make"')
    (2, '0.013*"ale" + 0.012*"style" + 0.011*"like" + 0.009*"double" + 0.008*"taste" + 0.008*"get"')
    (3, '0.032*"day" + 0.028*"national" + 0.008*"alcohol" + 0.007*"hop" + 0.007*"craft" + 0.006*"brewery"')
    (4, '0.019*"drink" + 0.017*"alcohol" + 0.017*"calorie" + 0.010*"say" + 0.008*"wine" + 0.007*"well"')
    (5, '0.015*"style" + 0.012*"hop" + 0.011*"like" + 0.010*"flavor" + 0.009*"bottle" + 0.008*"brewery"')



```python
# dictionary_ipa = corpora.Dictionary.load('dictionary_ipa.gensim')
# corpus_ipa = pickle.load(open('corpus_ipa.pkl', 'rb'))
# ldamodel_ipa = models.ldamodel.LdaModel.load('model_ipa.gensim')

lda_display = pyLDAvis.gensim.prepare(ldamodel_ipa, corpus_ipa, dictionary_ipa, sort_topics=False)
pyLDAvis.display(lda_display)
```

    /Users/rliao/opt/anaconda3/lib/python3.7/site-packages/pyLDAvis/_prepare.py:257: FutureWarning: Sorting because non-concatenation axis is not aligned. A future version
    of pandas will change to not sort by default.
    
    To accept the future behavior, pass 'sort=False'.
    
    To retain the current behavior and silence the warning, pass 'sort=True'.
    
      return pd.concat([default_term_info] + list(topic_dfs))






<link rel="stylesheet" type="text/css" href="https://cdn.rawgit.com/bmabey/pyLDAvis/files/ldavis.v1.0.0.css">


<div id="ldavis_el113841406024480712488855019749"></div>
<script type="text/javascript">

var ldavis_el113841406024480712488855019749_data = {"mdsDat": {"x": [0.04014362064618739, 0.046811734518713406, 0.06644763556760278, -0.16729445828239675, -0.028419855573426054, 0.04231132312331941], "y": [-0.050624187474789716, -0.009812048097858876, -0.005659573106721284, -0.054663530836729146, 0.1609865833420228, -0.04022724382592385], "topics": [1, 2, 3, 4, 5, 6], "cluster": [1, 1, 1, 1, 1, 1], "Freq": [13.2523193359375, 25.70890235900879, 21.026718139648438, 12.282323837280273, 14.37370491027832, 13.356025695800781]}, "tinfo": {"Category": ["Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6"], "Freq": [67.0, 103.0, 47.0, 32.0, 104.0, 124.0, 139.0, 49.0, 53.0, 27.0, 76.0, 21.0, 53.0, 14.0, 86.0, 15.0, 30.0, 23.0, 65.0, 26.0, 14.0, 29.0, 10.0, 18.0, 11.0, 14.0, 80.0, 24.0, 14.0, 76.0, 31.866437911987305, 8.066734313964844, 4.888217926025391, 3.3075268268585205, 3.305975914001465, 3.3041160106658936, 2.51452374458313, 2.5134265422821045, 2.512336492538452, 2.5123074054718018, 2.504140615463257, 11.244987487792969, 5.4744367599487305, 4.100160121917725, 4.099002838134766, 1.7214607000350952, 1.7209086418151855, 1.7208093404769897, 1.720651626586914, 1.7205530405044556, 1.7205195426940918, 1.7205040454864502, 1.7204718589782715, 1.7204554080963135, 1.7204090356826782, 1.720337986946106, 1.720310091972351, 1.720226764678955, 1.7200686931610107, 1.720041275024414, 3.3092992305755615, 3.3077452182769775, 3.3086977005004883, 7.159613609313965, 6.480764865875244, 5.795683860778809, 13.561172485351562, 7.599917411804199, 3.8153748512268066, 7.279172420501709, 4.1018595695495605, 7.387483596801758, 4.035781383514404, 25.96409797668457, 2.513411283493042, 3.3085176944732666, 5.6899566650390625, 24.98251724243164, 25.7062931060791, 28.217660903930664, 9.97440242767334, 23.905590057373047, 21.034080505371094, 9.64954662322998, 7.816154479980469, 12.21242904663086, 15.354532241821289, 10.20495319366455, 8.599868774414062, 6.660122394561768, 6.273150444030762, 10.121716499328613, 9.701925277709961, 9.08358383178711, 9.523931503295898, 8.276687622070312, 7.7538604736328125, 8.31562328338623, 7.518013954162598, 7.864833354949951, 7.82863712310791, 7.285473823547363, 17.799280166625977, 8.08155632019043, 7.2085280418396, 5.431138515472412, 11.5111665725708, 4.563867092132568, 4.563868999481201, 4.5630645751953125, 4.562963008880615, 4.559915542602539, 5.262057781219482, 5.260501861572266, 9.7529935836792, 16.449392318725586, 3.6782684326171875, 3.6765732765197754, 3.6758697032928467, 3.673248767852783, 3.6699790954589844, 3.648019790649414, 3.6476902961730957, 15.116618156433105, 7.197491645812988, 6.3174238204956055, 13.501609802246094, 2.7958486080169678, 2.7955403327941895, 2.7950439453125, 2.794545888900757, 2.7943811416625977, 20.982465744018555, 42.00255584716797, 8.973737716674805, 9.480672836303711, 6.74063777923584, 13.986289024353027, 14.055088996887207, 12.801005363464355, 5.442030906677246, 41.05636978149414, 18.113327026367188, 8.582751274108887, 31.51409149169922, 27.798465728759766, 33.01177215576172, 21.823623657226562, 27.532052993774414, 12.514763832092285, 24.133337020874023, 22.42445182800293, 19.879497528076172, 21.327531814575195, 55.62920379638672, 29.971914291381836, 50.00640869140625, 39.99823760986328, 44.15509033203125, 14.838075637817383, 22.47571563720703, 31.515625, 30.659692764282227, 24.719188690185547, 23.938648223876953, 23.216503143310547, 22.03019905090332, 22.340030670166016, 18.357839584350586, 21.966854095458984, 19.694787979125977, 19.72770881652832, 13.961359024047852, 3.6009063720703125, 3.5998759269714355, 3.595215082168579, 7.0567450523376465, 11.288060188293457, 3.5287373065948486, 6.16246223449707, 2.7368459701538086, 2.7365312576293945, 2.7343826293945312, 2.7343368530273438, 2.7342684268951416, 2.7341206073760986, 2.734113931655884, 2.7339892387390137, 2.733354091644287, 2.7326951026916504, 2.7310233116149902, 2.730773448944092, 2.7221689224243164, 2.7192747592926025, 2.710455894470215, 2.703629493713379, 2.6820507049560547, 2.6671018600463867, 2.662998914718628, 2.669290542602539, 11.950495719909668, 9.629040718078613, 34.65956115722656, 11.422672271728516, 7.171779632568359, 29.1517333984375, 4.46220588684082, 4.399599552154541, 11.604381561279297, 4.391282558441162, 12.988560676574707, 4.387265682220459, 13.105548858642578, 21.03447914123535, 14.203302383422852, 29.30917739868164, 8.750754356384277, 47.21758270263672, 31.237079620361328, 29.817331314086914, 52.005760192871094, 6.691774368286133, 17.614656448364258, 15.604604721069336, 13.965056419372559, 44.198787689208984, 20.706571578979492, 12.882513999938965, 34.154666900634766, 15.709603309631348, 13.882491111755371, 19.281322479248047, 29.825117111206055, 25.962820053100586, 23.607784271240234, 21.617473602294922, 22.457408905029297, 20.531206130981445, 19.092905044555664, 25.154848098754883, 17.793682098388672, 21.375625610351562, 15.36112117767334, 14.70506477355957, 15.192971229553223, 14.858338356018066, 14.635966300964355, 14.82586669921875, 14.763663291931152, 66.3838119506836, 10.142794609069824, 8.601898193359375, 8.601612091064453, 8.601079940795898, 7.832376480102539, 7.062421798706055, 7.061607837677002, 7.0428876876831055, 6.289590835571289, 5.518829345703125, 4.750514984130859, 4.7503509521484375, 10.140356063842773, 3.9791135787963867, 3.978882312774658, 3.978447198867798, 7.831093788146973, 3.20914626121521, 3.2085683345794678, 3.207979440689087, 2.43845796585083, 2.4384660720825195, 2.4384193420410156, 2.4379923343658447, 2.4374890327453613, 2.4355220794677734, 2.435452461242676, 2.43468976020813, 2.4341166019439697, 77.1899642944336, 6.264977931976318, 7.828463077545166, 3.9692468643188477, 8.560900688171387, 10.916840553283691, 3.9816415309906006, 3.207716226577759, 3.1946632862091064, 17.034761428833008, 10.655007362365723, 11.671466827392578, 19.005455017089844, 7.041587829589844, 13.917829513549805, 17.263288497924805, 6.8614888191223145, 13.856474876403809, 6.4477314949035645, 7.349179744720459, 11.005805015563965, 6.880119800567627, 8.9232759475708, 7.724606513977051, 8.17290210723877, 7.326672077178955, 7.253173828125, 46.30470275878906, 13.77804946899414, 8.963855743408203, 8.159226417541504, 6.555057525634766, 6.550371170043945, 5.753086090087891, 4.950296878814697, 4.9503278732299805, 4.949325084686279, 4.949249744415283, 4.948894023895264, 4.148839473724365, 4.147928237915039, 4.147049427032471, 4.146120071411133, 8.967421531677246, 5.472082138061523, 12.976507186889648, 3.3442471027374268, 3.3441226482391357, 3.3439180850982666, 3.341111660003662, 3.333514451980591, 3.329035997390747, 11.798188209533691, 16.99360466003418, 5.7518181800842285, 2.5429768562316895, 2.541883945465088, 5.593239784240723, 6.472848892211914, 17.08438491821289, 16.784780502319336, 12.34667682647705, 51.77769470214844, 47.33938217163086, 26.455280303955078, 21.29960823059082, 7.892366409301758, 9.049927711486816, 11.371253967285156, 18.678335189819336, 12.422629356384277, 10.3917818069458, 6.850802421569824, 9.677786827087402, 11.26262092590332, 13.598977088928223, 15.621957778930664, 14.318207740783691, 11.4168062210083, 9.102368354797363, 10.406587600708008, 12.097206115722656, 8.85324478149414, 10.325968742370605, 9.050667762756348, 9.122282028198242, 9.264317512512207, 4.125555992126465, 7.993449687957764, 2.5306127071380615, 2.5284759998321533, 2.5268707275390625, 5.7237443923950195, 8.035490989685059, 4.127211570739746, 4.688531875610352, 3.468065023422241, 1.7318446636199951, 1.7318459749221802, 1.731841802597046, 1.7315301895141602, 1.7315177917480469, 1.7314049005508423, 1.731302261352539, 1.7313284873962402, 1.731257677078247, 1.7311726808547974, 1.7310913801193237, 1.7310417890548706, 1.7311149835586548, 1.7310080528259277, 1.7310031652450562, 1.7309778928756714, 1.730928659439087, 1.7309120893478394, 1.7309746742248535, 1.7308827638626099, 4.040411949157715, 7.32563591003418, 3.3279457092285156, 14.859620094299316, 3.3267133235931396, 3.3303565979003906, 24.414037704467773, 5.955667972564697, 8.936678886413574, 10.061261177062988, 16.06609535217285, 4.121096611022949, 3.322575807571411, 39.50468826293945, 8.009387969970703, 8.29515552520752, 3.3306431770324707, 2.529494524002075, 2.529146432876587, 2.531834840774536, 26.417638778686523, 9.460197448730469, 13.342424392700195, 8.788055419921875, 21.16246223449707, 32.26816940307617, 28.21866798400879, 12.170636177062988, 6.617937088012695, 7.149292469024658, 11.891280174255371, 14.991230964660645, 14.48935317993164, 15.381810188293457, 17.984010696411133, 13.260380744934082, 9.934351921081543, 10.505647659301758, 14.546194076538086, 12.37903881072998, 12.331073760986328, 13.148336410522461, 14.708202362060547, 11.571818351745605, 11.103507041931152, 9.327991485595703, 9.027103424072266], "Term": ["national", "day", "calorie", "olive", "drink", "alcohol", "style", "bottle", "abv", "pair", "wine", "contains", "double", "weight", "lager", "percent", "gluten", "acid", "say", "food", "amber", "red", "june", "jalape\u00f1o", "woman", "nbsp", "brewery", "vodka", "per", "pale", "olive", "beertini", "individual", "prague", "barroom", "garde", "net", "equis", "castelvetrano", "dakota", "lb", "amber", "vienna", "midwest", "perfect", "sugary", "explosm", "distribution", "tripel", "han", "crew", "beauty", "m\u00fcller", "colorful", "monastery", "sommelierbier", "strategy", "mania", "father", "sherry", "historical", "oxidation", "buckwheat", "ratebeer", "green", "major", "red", "prefer", "recommendation", "single", "break", "kind", "sort", "lager", "south", "soon", "trappist", "taste", "style", "hop", "belgian", "ale", "like", "find", "pair", "pale", "make", "know", "come", "quite", "little", "use", "stout", "malt", "brewery", "american", "brewer", "much", "go", "really", "well", "many", "jalape\u00f1o", "harvest", "xpa", "chipotle", "hangover", "yin", "yang", "evil", "twin", "oak", "japanese", "starch", "sake", "jalapeno", "suppose", "retire", "producer", "depends", "awful", "technology", "anywhere", "blend", "distil", "wood", "barrel", "decision", "pennsylvania", "row", "differentiates", "spoil", "water", "abv", "grape", "wort", "extract", "english", "boil", "ferment", "expensive", "wine", "temperature", "session", "brewing", "brew", "malt", "process", "age", "barley", "brewer", "yeast", "ingredient", "sugar", "ale", "use", "hop", "make", "like", "old", "really", "alcohol", "taste", "get", "brewery", "good", "stout", "different", "hoppy", "well", "pale", "flavor", "nbsp", "italy", "guiness", "track", "indian", "triple", "tripels", "notice", "moment", "birra", "terminology", "sorghum", "flanders", "imply", "clarex", "nobody", "data", "considerable", "approach", "dubbels", "overcarbonated", "vacuum", "fridge", "serious", "surprisingly", "thumb", "dull", "creamy", "degree", "flight", "double", "cold", "variant", "strong", "foam", "adjunct", "store", "heady", "wheat", "topper", "free", "imperial", "dark", "try", "buy", "style", "get", "pale", "ale", "warm", "go", "porter", "gluten", "like", "age", "black", "taste", "want", "something", "american", "make", "flavor", "well", "good", "lager", "different", "stout", "hop", "much", "alcohol", "yeast", "could", "best", "high", "see", "say", "really", "national", "june", "october", "august", "september", "november", "july", "january", "reykjav\u00edk", "sexual", "testosterone", "february", "april", "woman", "march", "orgasm", "arousal", "december", "weekend", "libido", "vodkabeer", "male", "orgasmic", "intoxication", "latency", "homebrew", "bruggsmi\u00f0jan", "operating", "downtown", "bruggh\u00fas", "day", "icelandic", "men", "micro", "increase", "vodka", "tea", "impact", "einst\u00f6k", "craft", "bar", "great", "alcohol", "world", "brewery", "hop", "product", "ale", "local", "new", "drink", "real", "style", "different", "make", "much", "flavor", "calorie", "weight", "terminal", "fear", "booze", "pairing", "int", "moderate", "gate", "cederquist", "proof", "health", "dessert", "gain", "fry", "carbohydrate", "airport", "gram", "percent", "journal", "excuse", "vermouth", "boost", "pork", "wise", "per", "contains", "loss", "complement", "grill", "rich", "dish", "pair", "food", "spicy", "drink", "alcohol", "say", "wine", "pack", "typically", "gluten", "well", "light", "less", "shot", "drinking", "high", "flavor", "like", "make", "try", "people", "much", "hop", "something", "day", "thing", "bottle", "taste", "sunlight", "alpha", "mbt", "corn", "sensitive", "tell", "rye", "brett", "primary", "dimension", "outmeal", "revival", "funny", "google", "phrase", "cup", "blah", "baderbrau", "affordable", "opposite", "progress", "reykjavik", "urbana", "snob", "austurstr\u00e6ti", "bell", "g\u00e6dingur", "isohumulones", "hybrid", "min", "slight", "beck", "miller", "acid", "skunky", "kolsch", "bottle", "perceive", "cider", "one", "bitterness", "heineken", "significantly", "style", "german", "specific", "doppelbock", "beta", "direct", "disappointed", "flavor", "actually", "bitter", "pretty", "brewery", "hop", "like", "find", "heavy", "include", "many", "different", "good", "lager", "make", "stout", "light", "think", "taste", "pale", "get", "drink", "ale", "well", "use", "really", "craft"], "Total": [67.0, 103.0, 47.0, 32.0, 104.0, 124.0, 139.0, 49.0, 53.0, 27.0, 76.0, 21.0, 53.0, 14.0, 86.0, 15.0, 30.0, 23.0, 65.0, 26.0, 14.0, 29.0, 10.0, 18.0, 11.0, 14.0, 80.0, 24.0, 14.0, 76.0, 32.61084747314453, 8.767006874084473, 5.587818145751953, 3.9983134269714355, 3.998347282409668, 3.998389482498169, 3.203308582305908, 3.2034709453582764, 3.2035229206085205, 3.2035090923309326, 3.2035391330718994, 14.538498878479004, 7.183002471923828, 5.596090793609619, 5.596658229827881, 2.408560037612915, 2.4085659980773926, 2.408573865890503, 2.4085731506347656, 2.4085986614227295, 2.4085988998413086, 2.4086058139801025, 2.4085917472839355, 2.408604383468628, 2.408627986907959, 2.408602476119995, 2.4085638523101807, 2.4086060523986816, 2.4086222648620605, 2.408595561981201, 4.797472953796387, 4.797492027282715, 4.861772060394287, 11.412386894226074, 10.347284317016602, 9.903142929077148, 29.089509963989258, 14.68909740447998, 6.458961486816406, 14.787074089050293, 7.259334564208984, 15.833216667175293, 7.336648941040039, 86.59327697753906, 3.974055290222168, 5.764042377471924, 12.388372421264648, 117.95079803466797, 139.78456115722656, 165.00758361816406, 34.22906494140625, 163.660888671875, 159.93885803222656, 41.09349822998047, 27.316360473632812, 76.77557373046875, 125.65300750732422, 55.45299530029297, 40.52516555786133, 23.02356719970703, 21.23943519592285, 75.67992401123047, 68.89054870605469, 64.6895980834961, 80.5876693725586, 58.14170837402344, 47.07723617553711, 66.71526336669922, 40.79036331176758, 60.74676513671875, 87.71371459960938, 47.35257339477539, 18.486732482910156, 8.768325805664062, 7.885991096496582, 6.118574142456055, 13.069412231445312, 5.236151695251465, 5.2361531257629395, 5.236094951629639, 5.236094951629639, 5.2358574867248535, 6.0964202880859375, 6.096078395843506, 11.327308654785156, 19.229211807250977, 4.352480411529541, 4.352434158325195, 4.352284908294678, 4.352175712585449, 4.351843357086182, 4.348481178283691, 4.348751544952393, 18.14584732055664, 8.688826560974121, 7.7964982986450195, 16.6630802154541, 3.469120740890503, 3.4691219329833984, 3.4690778255462646, 3.469029188156128, 3.469017267227173, 26.0511474609375, 53.26105499267578, 11.260109901428223, 12.09091567993164, 8.626272201538086, 19.203628540039062, 19.808204650878906, 18.00202751159668, 6.918553352355957, 76.8620834350586, 29.658823013305664, 12.073454856872559, 59.17063903808594, 51.17851257324219, 64.6895980834961, 38.881370544433594, 54.52250289916992, 19.70531463623047, 47.07723617553711, 44.03317642211914, 37.372371673583984, 41.476524353027344, 163.660888671875, 75.67992401123047, 165.00758361816406, 125.65300750732422, 159.93885803222656, 26.950048446655273, 60.74676513671875, 124.06795501708984, 117.95079803466797, 81.39277648925781, 80.5876693725586, 74.84892272949219, 68.89054870605469, 74.50567626953125, 43.54525375366211, 87.71371459960938, 76.77557373046875, 98.74366760253906, 14.649266242980957, 4.276769638061523, 4.276680946350098, 4.276429653167725, 8.533742904663086, 13.651660919189453, 4.268911361694336, 7.638054847717285, 3.412304162979126, 3.412313461303711, 3.412135124206543, 3.412104845046997, 3.4121687412261963, 3.412039279937744, 3.412067174911499, 3.412126064300537, 3.412076950073242, 3.4121198654174805, 3.4118218421936035, 3.4117605686187744, 3.410926103591919, 3.412224531173706, 3.4126040935516357, 3.408738613128662, 3.412607431411743, 3.4072012901306152, 3.4041645526885986, 3.412813186645508, 15.464615821838379, 12.71426773071289, 53.04048538208008, 16.248249053955078, 10.115189552307129, 48.800113677978516, 6.023542404174805, 5.936169147491455, 18.436357498168945, 6.020719051361084, 22.1455135345459, 6.020451545715332, 23.394073486328125, 42.639854431152344, 26.665307998657227, 69.7951431274414, 15.168551445007324, 139.78456115722656, 81.39277648925781, 76.77557373046875, 163.660888671875, 11.094476699829102, 40.79036331176758, 35.58160400390625, 30.67022705078125, 159.93885803222656, 54.52250289916992, 27.962739944458008, 117.95079803466797, 39.72087478637695, 33.393272399902344, 58.14170837402344, 125.65300750732422, 98.74366760253906, 87.71371459960938, 74.84892272949219, 86.59327697753906, 74.50567626953125, 68.89054870605469, 165.00758361816406, 66.71526336669922, 124.06795501708984, 44.03317642211914, 37.36531066894531, 44.25632858276367, 51.609947204589844, 41.95659637451172, 65.13506317138672, 60.74676513671875, 67.11788177490234, 10.841179847717285, 9.299459457397461, 9.299405097961426, 9.299400329589844, 8.528397560119629, 7.7574591636657715, 7.7575273513793945, 7.758195877075195, 6.986662864685059, 6.215790271759033, 5.444732189178467, 5.444750785827637, 11.644746780395508, 4.673941135406494, 4.6738600730896, 4.67388916015625, 9.4111328125, 3.902930498123169, 3.902949333190918, 3.902930736541748, 3.131998062133789, 3.1320090293884277, 3.13200044631958, 3.1320340633392334, 3.1317176818847656, 3.132089376449585, 3.1320881843566895, 3.132136583328247, 3.1321449279785156, 103.82608795166016, 8.586952209472656, 10.939886093139648, 5.473719596862793, 15.131036758422852, 24.18950080871582, 6.281063079833984, 4.701579570770264, 4.702920436859131, 62.28260040283203, 31.494953155517578, 41.905128479003906, 124.06795501708984, 18.592859268188477, 80.5876693725586, 165.00758361816406, 20.5119571685791, 163.660888671875, 18.518842697143555, 28.238208770751953, 104.40326690673828, 27.508251190185547, 139.78456115722656, 74.50567626953125, 125.65300750732422, 66.71526336669922, 98.74366760253906, 47.47698974609375, 14.484542846679688, 9.66173267364502, 8.857707977294922, 7.2499613761901855, 7.250246047973633, 6.446233749389648, 5.642271995544434, 5.642392158508301, 5.642315864562988, 5.642322540283203, 5.642238140106201, 4.838340759277344, 4.8383660316467285, 4.838311672210693, 4.838379859924316, 10.525421142578125, 6.47260856628418, 15.408965110778809, 4.03455924987793, 4.034584045410156, 4.034584999084473, 4.034280776977539, 4.035109996795654, 4.034075736999512, 14.479815483093262, 21.12019157409668, 7.240410804748535, 3.230569362640381, 3.230630397796631, 7.230210304260254, 8.931659698486328, 27.316360473632812, 26.883867263793945, 18.804630279541016, 104.40326690673828, 124.06795501708984, 65.13506317138672, 76.8620834350586, 14.732545852661133, 19.967144012451172, 30.67022705078125, 87.71371459960938, 38.48555374145508, 30.604848861694336, 12.429061889648438, 30.79939079284668, 51.609947204589844, 98.74366760253906, 159.93885803222656, 125.65300750732422, 69.7951431274414, 35.659263610839844, 66.71526336669922, 165.00758361816406, 33.393272399902344, 103.82608795166016, 41.69353103637695, 49.24330139160156, 117.95079803466797, 4.81851863861084, 9.608189582824707, 3.2187538146972656, 3.218604564666748, 3.2188103199005127, 7.301032543182373, 10.482288360595703, 5.682004928588867, 6.47312068939209, 4.813802242279053, 2.4188590049743652, 2.4188671112060547, 2.4188899993896484, 2.4188194274902344, 2.4188437461853027, 2.4188895225524902, 2.4188430309295654, 2.418937921524048, 2.418954849243164, 2.418872117996216, 2.4188694953918457, 2.418867588043213, 2.4189705848693848, 2.418846607208252, 2.418853282928467, 2.4188740253448486, 2.418851137161255, 2.4188759326934814, 2.418976306915283, 2.418872117996216, 5.688631534576416, 10.667742729187012, 4.812770366668701, 23.476612091064453, 4.813044548034668, 4.900481224060059, 49.24330139160156, 9.90823745727539, 18.346088409423828, 22.0229549407959, 42.92390060424805, 7.379061698913574, 5.583392143249512, 139.78456115722656, 17.687137603759766, 18.844154357910156, 5.758769989013672, 4.0130133628845215, 4.013131141662598, 4.022109508514404, 98.74366760253906, 24.429590225219727, 40.14984130859375, 22.560916900634766, 80.5876693725586, 165.00758361816406, 159.93885803222656, 41.09349822998047, 15.760701179504395, 18.06208610534668, 47.35257339477539, 74.50567626953125, 74.84892272949219, 86.59327697753906, 125.65300750732422, 68.89054870605469, 38.48555374145508, 44.62931823730469, 117.95079803466797, 76.77557373046875, 81.39277648925781, 104.40326690673828, 163.660888671875, 87.71371459960938, 75.67992401123047, 60.74676513671875, 62.28260040283203], "loglift": [30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 1.9979000091552734, 1.9378000497817993, 1.8871999979019165, 1.8313000202178955, 1.8308000564575195, 1.830299973487854, 1.7789000272750854, 1.77839994430542, 1.777999997138977, 1.777999997138977, 1.7747000455856323, 1.7640999555587769, 1.749400019645691, 1.7100000381469727, 1.7095999717712402, 1.6850999593734741, 1.6848000288009644, 1.6848000288009644, 1.6847000122070312, 1.6845999956130981, 1.6845999956130981, 1.6845999956130981, 1.6845999956130981, 1.684499979019165, 1.684499979019165, 1.684499979019165, 1.684499979019165, 1.684399962425232, 1.6842999458312988, 1.6842999458312988, 1.6496000289916992, 1.6491999626159668, 1.6361000537872314, 1.554800033569336, 1.5530999898910522, 1.4852999448776245, 1.2577999830245972, 1.3619999885559082, 1.4946000576019287, 1.3122999668121338, 1.4500999450683594, 1.2587000131607056, 1.42330002784729, 0.8165000081062317, 1.5628999471664429, 1.46589994430542, 1.242900013923645, 0.46889999508857727, 0.32760000228881836, 0.2549999952316284, 0.7878999710083008, 0.09730000048875809, -0.007699999958276749, 0.5720999836921692, 0.7696999907493591, 0.1826000064611435, -0.08110000193119049, 0.32829999923706055, 0.4708000123500824, 0.7806000113487244, 0.8014000058174133, 0.009200000204145908, 0.06080000102519989, 0.05790000036358833, -0.1145000010728836, 0.07159999758005142, 0.21739999949932098, -0.06129999831318855, 0.32989999651908875, -0.02329999953508377, -0.3953000009059906, 0.1492999941110611, 1.3203999996185303, 1.2768000364303589, 1.2684999704360962, 1.2391999959945679, 1.2314000129699707, 1.220900058746338, 1.220900058746338, 1.2208000421524048, 1.2207000255584717, 1.2201000452041626, 1.2111999988555908, 1.2108999490737915, 1.2086999416351318, 1.2022000551223755, 1.190000057220459, 1.1895999908447266, 1.1893999576568604, 1.1886999607086182, 1.1878999471664429, 1.1827000379562378, 1.1825000047683716, 1.1756999492645264, 1.1699999570846558, 1.1480000019073486, 1.1478999853134155, 1.1426000595092773, 1.1425000429153442, 1.142300009727478, 1.1420999765396118, 1.1420999765396118, 1.1419999599456787, 1.12090003490448, 1.1313999891281128, 1.1151000261306763, 1.1117000579833984, 1.0413000583648682, 1.0152000188827515, 1.0174000263214111, 1.118299961090088, 0.7312999963760376, 0.8651999831199646, 1.0170999765396118, 0.7282999753952026, 0.7480000257492065, 0.6855999827384949, 0.7807999849319458, 0.6751000285148621, 0.9043999910354614, 0.6901000142097473, 0.6834999918937683, 0.7271000146865845, 0.6931999921798706, 0.2791999876499176, 0.43209999799728394, 0.16449999809265137, 0.21359999477863312, 0.07119999825954437, 0.7615000009536743, 0.36410000920295715, -0.012000000104308128, 0.010999999940395355, 0.16660000383853912, 0.1445000022649765, 0.18770000338554382, 0.21819999814033508, 0.15379999577999115, 0.49459999799728394, -0.026200000196695328, -0.002199999988079071, -0.25220000743865967, 1.511299967765808, 1.3874000310897827, 1.3870999813079834, 1.3859000205993652, 1.3693000078201294, 1.3693000078201294, 1.36899995803833, 1.3446999788284302, 1.3387999534606934, 1.3387000560760498, 1.3379000425338745, 1.3379000425338745, 1.3379000425338745, 1.3379000425338745, 1.3379000425338745, 1.3378000259399414, 1.3375999927520752, 1.3372999429702759, 1.3367999792099, 1.3366999626159668, 1.333799958229065, 1.3323999643325806, 1.3289999961853027, 1.3276000022888184, 1.31850004196167, 1.3144999742507935, 1.3137999773025513, 1.3136999607086182, 1.3015999794006348, 1.2813999652862549, 1.1339000463485718, 1.2070000171661377, 1.215499997138977, 1.0441999435424805, 1.2592999935150146, 1.2597999572753906, 1.0964000225067139, 1.2438000440597534, 1.0257999897003174, 1.242900013923645, 0.9799000024795532, 0.8526999950408936, 0.9294999837875366, 0.6916999816894531, 1.0092999935150146, 0.4740000069141388, 0.6017000079154968, 0.6136000156402588, 0.41290000081062317, 1.0537999868392944, 0.7196999788284302, 0.7350999712944031, 0.772599995136261, 0.2732999920845032, 0.5911999940872192, 0.7843999862670898, 0.3199999928474426, 0.6317999958992004, 0.6815999746322632, 0.45559999346733093, 0.12120000272989273, 0.22349999845027924, 0.24690000712871552, 0.3174000084400177, 0.20980000495910645, 0.2703999876976013, 0.27619999647140503, -0.3215999901294708, 0.2378000020980835, -0.19920000433921814, 0.5062999725341797, 0.626800000667572, 0.490200012922287, 0.3142000138759613, 0.5062000155448914, 0.07930000126361847, 0.14480000734329224, 2.0859999656677246, 2.030400037765503, 2.0190000534057617, 2.0190000534057617, 2.018899917602539, 2.011899948120117, 2.0030999183654785, 2.003000020980835, 2.0002999305725098, 1.9918999671936035, 1.9780999422073364, 1.9606000185012817, 1.9606000185012817, 1.9586999416351318, 1.9361000061035156, 1.9359999895095825, 1.9358999729156494, 1.9132000207901, 1.9012999534606934, 1.9011000394821167, 1.9009000062942505, 1.8466999530792236, 1.8466999530792236, 1.8466999530792236, 1.846500039100647, 1.8464000225067139, 1.8454999923706055, 1.8453999757766724, 1.8451000452041626, 1.8449000120162964, 1.8006000518798828, 1.7817000150680542, 1.7624000310897827, 1.7755999565124512, 1.527500033378601, 1.3013999462127686, 1.6411999464035034, 1.7146999835968018, 1.7102999687194824, 0.800599992275238, 1.013200044631958, 0.8187000155448914, 0.22089999914169312, 1.126099944114685, 0.3407999873161316, -0.16040000319480896, 1.0018999576568604, -0.3720000088214874, 1.0419000387191772, 0.7508999705314636, -0.15279999375343323, 0.7111999988555908, -0.6543999910354614, -0.16949999332427979, -0.635699987411499, -0.11190000176429749, -0.5141000151634216, 1.9148000478744507, 1.889799952507019, 1.864799976348877, 1.8575999736785889, 1.8389999866485596, 1.8382999897003174, 1.8259999752044678, 1.808899998664856, 1.808899998664856, 1.8086999654769897, 1.8086999654769897, 1.8086999654769897, 1.7860000133514404, 1.7857999801635742, 1.785599946975708, 1.7854000329971313, 1.7796000242233276, 1.7718000411987305, 1.7680000066757202, 1.7520999908447266, 1.7520999908447266, 1.7519999742507935, 1.7511999607086182, 1.7488000392913818, 1.7476999759674072, 1.7350000143051147, 1.7223999500274658, 1.7095999717712402, 1.7003999948501587, 1.7000000476837158, 1.6830999851226807, 1.617799997329712, 1.4703999757766724, 1.4687000513076782, 1.5190999507904053, 1.2384999990463257, 0.9763000011444092, 1.0388000011444092, 0.6564000248908997, 1.315600037574768, 1.1483999490737915, 0.9476000070571899, 0.39309999346733093, 0.8090000152587891, 0.8596000075340271, 1.344099998474121, 0.7821000218391418, 0.41749998927116394, -0.04280000180006027, -0.3862999975681305, -0.2321999967098236, 0.12929999828338623, 0.5742999911308289, 0.08179999887943268, -0.6732000112533569, 0.6122000217437744, -0.3682999908924103, 0.4122999906539917, 0.25369998812675476, -0.6043000221252441, 1.8579000234603882, 1.829200029373169, 1.7726999521255493, 1.7719000577926636, 1.7711999416351318, 1.7697999477386475, 1.7474000453948975, 1.69350004196167, 1.6907000541687012, 1.6852999925613403, 1.6791000366210938, 1.6791000366210938, 1.6791000366210938, 1.6789000034332275, 1.6789000034332275, 1.6787999868392944, 1.6787999868392944, 1.6787999868392944, 1.6786999702453613, 1.6786999702453613, 1.6786999702453613, 1.6785999536514282, 1.6785999536514282, 1.6785999536514282, 1.6785999536514282, 1.6785999536514282, 1.6785999536514282, 1.6785000562667847, 1.6785000562667847, 1.6785000562667847, 1.6711000204086304, 1.6374000310897827, 1.6442999839782715, 1.555799961090088, 1.6439000368118286, 1.6268999576568604, 1.3115999698638916, 1.5041999816894531, 1.2940000295639038, 1.2297999858856201, 1.030500054359436, 1.4306999444961548, 1.4940999746322632, 0.7494999766349792, 1.2209999561309814, 1.1927000284194946, 1.4656000137329102, 1.5516999959945679, 1.5514999628067017, 1.5503000020980835, 0.6947000026702881, 1.0644999742507935, 0.9114999771118164, 1.0703999996185303, 0.6761000156402588, 0.3813000023365021, 0.2784000039100647, 0.7964000105857849, 1.1454999446868896, 1.086400032043457, 0.6313999891281128, 0.4097999930381775, 0.3711000084877014, 0.28519999980926514, 0.06920000165700912, 0.36550000309944153, 0.6589000225067139, 0.5666999816894531, -0.07970000058412552, 0.1882999986410141, 0.12600000202655792, -0.058800000697374344, -0.3962000012397766, -0.012299999594688416, 0.09399999678134918, 0.13950000703334808, 0.08179999887943268], "logprob": [30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, -4.387700080871582, -5.761499881744385, -6.262499809265137, -6.65310001373291, -6.653600215911865, -6.654099941253662, -6.927199840545654, -6.927599906921387, -6.928100109100342, -6.928100109100342, -6.931399822235107, -5.4293999671936035, -6.149199962615967, -6.438300132751465, -6.438600063323975, -7.306099891662598, -7.306399822235107, -7.30649995803833, -7.306600093841553, -7.306600093841553, -7.306700229644775, -7.306700229644775, -7.306700229644775, -7.306700229644775, -7.306700229644775, -7.30679988861084, -7.30679988861084, -7.30679988861084, -7.3069000244140625, -7.3069000244140625, -6.652599811553955, -6.6529998779296875, -6.652699947357178, -5.880799770355225, -5.980500221252441, -6.092199802398682, -5.242099761962891, -5.821199893951416, -6.510300159454346, -5.864299774169922, -6.437900066375732, -5.8495001792907715, -6.454100131988525, -4.592599868774414, -6.927700042724609, -6.6528000831604, -6.110599994659424, -4.631100177764893, -4.60260009765625, -4.509300231933594, -5.549300193786621, -4.67519998550415, -4.803199768066406, -5.582399845123291, -5.793099880218506, -5.346799850463867, -5.1178998947143555, -5.526400089263916, -5.697500228881836, -5.953199863433838, -6.013000011444092, -5.534599781036377, -5.577000141143799, -5.6427998542785645, -5.5954999923706055, -5.735899925231934, -5.80109977722168, -5.731200218200684, -5.831999778747559, -5.786900043487549, -5.791500091552734, -5.863399982452393, -5.632800102233887, -6.422399997711182, -6.5366997718811035, -6.819799900054932, -6.068600177764893, -6.993800163269043, -6.993800163269043, -6.99399995803833, -6.99399995803833, -6.994699954986572, -6.851399898529053, -6.8516998291015625, -6.234399795532227, -5.711699962615967, -7.209499835968018, -7.210000038146973, -7.21019983291626, -7.210899829864502, -7.2118000984191895, -7.217800140380859, -7.217899799346924, -5.796199798583984, -6.5381999015808105, -6.668600082397461, -5.909200191497803, -7.483799934387207, -7.48390007019043, -7.484099864959717, -7.484300136566162, -7.484300136566162, -5.468299865722656, -4.774199962615967, -6.317699909210205, -6.262700080871582, -6.603799819946289, -5.873899936676025, -5.86899995803833, -5.962399959564209, -6.817800045013428, -4.796999931335449, -5.615300178527832, -6.362199783325195, -5.061500072479248, -5.186999797821045, -5.015100002288818, -5.428999900817871, -5.196599960327148, -5.985099792480469, -5.328400135040283, -5.401800155639648, -5.522299766540527, -5.452000141143799, -4.493299961090088, -5.111700057983398, -4.599800109863281, -4.8231000900268555, -4.724299907684326, -5.814799785614014, -5.399499893188477, -5.061500072479248, -5.089000225067139, -5.3043999671936035, -5.33650016784668, -5.367099761962891, -5.41949987411499, -5.405600070953369, -5.601900100708008, -5.422399997711182, -5.531599998474121, -5.529900074005127, -5.674600124359131, -7.029699802398682, -7.03000020980835, -7.031300067901611, -6.356900215148926, -5.887199878692627, -7.050000190734863, -6.492400169372559, -7.304100036621094, -7.304200172424316, -7.304999828338623, -7.304999828338623, -7.305099964141846, -7.305099964141846, -7.305099964141846, -7.305200099945068, -7.3053998947143555, -7.305600166320801, -7.30620002746582, -7.306300163269043, -7.309500217437744, -7.3105998039245605, -7.313799858093262, -7.316299915313721, -7.3242998123168945, -7.329899787902832, -7.331500053405762, -7.329100131988525, -5.830100059509277, -6.04610013961792, -4.7652997970581055, -5.87529993057251, -6.340799808502197, -4.938399791717529, -6.815299987792969, -6.829400062561035, -5.859499931335449, -6.831299781799316, -5.746799945831299, -6.832200050354004, -5.7378997802734375, -5.264800071716309, -5.657400131225586, -4.933000087738037, -6.1417999267578125, -4.456200122833252, -4.86929988861084, -4.915800094604492, -4.359600067138672, -6.409999847412109, -5.442200183868408, -5.563399791717529, -5.6743998527526855, -4.522200107574463, -5.2804999351501465, -5.755000114440918, -4.78000020980835, -5.556600093841553, -5.680300235748291, -5.351799964904785, -4.915599822998047, -5.054299831390381, -5.1493000984191895, -5.237400054931641, -5.1992998123168945, -5.289000034332275, -5.361599922180176, -5.085899829864502, -5.43209981918335, -5.248700141906738, -5.579100131988525, -5.622700214385986, -5.590099811553955, -5.612400054931641, -5.627399921417236, -5.614500045776367, -5.61870002746582, -3.5778000354766846, -5.456500053405762, -5.621300220489502, -5.621300220489502, -5.621399879455566, -5.715000152587891, -5.81850004196167, -5.818600177764893, -5.821300029754639, -5.9344000816345215, -6.065100193023682, -6.215000152587891, -6.215099811553955, -5.4567999839782715, -6.392199993133545, -6.392300128936768, -6.392399787902832, -5.715199947357178, -6.6072998046875, -6.607500076293945, -6.607600212097168, -6.881899833679199, -6.881899833679199, -6.881899833679199, -6.8821001052856445, -6.882299900054932, -6.8831000328063965, -6.883200168609619, -6.883500099182129, -6.883699893951416, -3.427000045776367, -5.938300132751465, -5.7154998779296875, -6.394700050354004, -5.626100063323975, -5.382999897003174, -6.391600131988525, -6.607699871063232, -6.611800193786621, -4.938000202178955, -5.407299995422363, -5.316100120544434, -4.82859992980957, -5.821499824523926, -5.140100002288818, -4.924699783325195, -5.847400188446045, -5.144499778747559, -5.909599781036377, -5.77869987487793, -5.374899864196777, -5.844600200653076, -5.58459997177124, -5.728899955749512, -5.672500133514404, -5.781799793243408, -5.791800022125244, -4.095300197601318, -5.307400226593018, -5.737299919128418, -5.831399917602539, -6.050300121307373, -6.051000118255615, -6.180799961090088, -6.331099987030029, -6.331099987030029, -6.331299781799316, -6.331299781799316, -6.331399917602539, -6.507699966430664, -6.507900238037109, -6.5081000328063965, -6.508399963378906, -5.7368998527526855, -6.230899810791016, -5.367400169372559, -6.723299980163574, -6.723299980163574, -6.723400115966797, -6.7241997718811035, -6.726500034332275, -6.727799892425537, -5.462600231170654, -5.097700119018555, -6.181000232696533, -6.997200012207031, -6.997600078582764, -6.209000110626221, -6.062900066375732, -5.092400074005127, -5.110099792480469, -5.417099952697754, -3.983599901199341, -4.073200225830078, -4.655099868774414, -4.871799945831299, -5.86460018157959, -5.727799892425537, -5.4994001388549805, -5.003200054168701, -5.410999774932861, -5.5894999504089355, -6.006199836730957, -5.660699844360352, -5.508999824523926, -5.320499897003174, -5.18179988861084, -5.269000053405762, -5.4953999519348145, -5.7220001220703125, -5.588099956512451, -5.4375, -5.74970006942749, -5.595900058746338, -5.727700233459473, -5.719799995422363, -5.704400062561035, -6.439899921417236, -5.778500080108643, -6.928599834442139, -6.929500102996826, -6.930099964141846, -6.112500190734863, -5.773200035095215, -6.439499855041504, -6.311999797821045, -6.613500118255615, -7.3078999519348145, -7.3078999519348145, -7.3078999519348145, -7.30810022354126, -7.30810022354126, -7.308199882507324, -7.308199882507324, -7.308199882507324, -7.308199882507324, -7.308300018310547, -7.308300018310547, -7.3084001541137695, -7.308300018310547, -7.3084001541137695, -7.3084001541137695, -7.3084001541137695, -7.3084001541137695, -7.3084001541137695, -7.3084001541137695, -7.308499813079834, -6.460700035095215, -5.865699768066406, -6.654699802398682, -5.158400058746338, -6.655099868774414, -6.6539998054504395, -4.661900043487549, -6.072700023651123, -5.666900157928467, -5.548399925231934, -5.080399990081787, -6.440999984741211, -6.656400203704834, -4.180699825286865, -5.776500225067139, -5.741399765014648, -6.653900146484375, -6.929100036621094, -6.929200172424316, -6.928100109100342, -4.583099842071533, -5.610000133514404, -5.26609992980957, -5.683700084686279, -4.804900169372559, -4.382999897003174, -4.517099857330322, -5.358099937438965, -5.967299938201904, -5.890100002288818, -5.38129997253418, -5.149600028991699, -5.183700084686279, -5.123899936676025, -4.967599868774414, -5.272299766540527, -5.561100006103516, -5.505199909210205, -5.179800033569336, -5.341100215911865, -5.34499979019165, -5.280799865722656, -5.168700218200684, -5.4085001945495605, -5.44980001449585, -5.624100208282471, -5.656899929046631]}, "token.table": {"Topic": [2, 3, 5, 1, 2, 3, 5, 6, 1, 2, 3, 4, 5, 6, 3, 6, 6, 1, 2, 3, 5, 6, 3, 5, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 6, 1, 3, 1, 2, 3, 4, 5, 6, 2, 3, 4, 4, 4, 6, 2, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 5, 2, 3, 1, 1, 2, 6, 1, 1, 2, 3, 4, 6, 6, 1, 2, 3, 4, 5, 6, 1, 6, 3, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 6, 2, 4, 2, 3, 4, 6, 5, 5, 2, 3, 4, 5, 6, 1, 3, 5, 6, 3, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 4, 4, 1, 3, 2, 3, 4, 6, 2, 5, 5, 1, 5, 2, 1, 3, 4, 5, 6, 3, 2, 3, 4, 6, 1, 1, 2, 3, 4, 5, 6, 5, 3, 2, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 3, 1, 6, 1, 1, 2, 3, 4, 6, 3, 1, 2, 4, 5, 6, 2, 4, 2, 2, 3, 4, 2, 5, 1, 2, 3, 4, 5, 6, 2, 1, 6, 1, 6, 5, 6, 1, 2, 5, 2, 5, 1, 2, 3, 6, 1, 2, 3, 5, 6, 4, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 3, 3, 4, 6, 2, 3, 6, 1, 2, 5, 2, 6, 1, 2, 4, 6, 1, 5, 4, 2, 6, 1, 2, 3, 4, 5, 6, 3, 1, 2, 3, 4, 5, 6, 1, 3, 6, 2, 3, 1, 2, 3, 4, 5, 6, 1, 3, 5, 3, 5, 6, 5, 1, 5, 1, 3, 4, 6, 1, 2, 3, 4, 5, 6, 1, 3, 5, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 6, 5, 2, 5, 1, 2, 3, 4, 5, 6, 1, 4, 6, 5, 3, 6, 1, 2, 4, 2, 2, 3, 5, 2, 3, 4, 6, 1, 2, 6, 1, 2, 3, 4, 5, 6, 1, 6, 4, 1, 2, 3, 4, 5, 6, 1, 2, 3, 5, 6, 6, 4, 6, 1, 4, 1, 2, 3, 4, 5, 6, 3, 1, 2, 3, 5, 6, 2, 3, 4, 5, 3, 6, 1, 1, 2, 3, 4, 5, 4, 6, 3, 2, 3, 5, 2, 4, 2, 5, 4, 4, 1, 2, 3, 1, 2, 3, 4, 5, 6, 2, 6, 1, 2, 3, 4, 5, 6, 4, 1, 1, 2, 4, 5, 6, 4, 1, 2, 3, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 6, 1, 5, 1, 2, 1, 2, 3, 4, 5, 6, 4, 1, 2, 3, 4, 5, 6, 1, 1, 2, 3, 4, 5, 6, 4, 6, 4, 5, 4, 6, 1, 5, 1, 6, 6, 5, 3, 1, 1, 2, 3, 4, 5, 6, 1, 4, 3, 1, 1, 2, 3, 4, 5, 3, 3, 4, 4, 2, 4, 1, 2, 3, 4, 6, 1, 1, 2, 3, 5, 6, 4, 6, 4, 4, 6, 3, 1, 6, 3, 5, 6, 1, 2, 4, 5, 5, 1, 2, 3, 4, 5, 6, 2, 1, 2, 3, 5, 6, 1, 5, 6, 2, 6, 3, 5, 1, 5, 6, 5, 1, 2, 3, 4, 5, 6, 1, 1, 3, 4, 6, 2, 5, 6, 3, 6, 1, 2, 3, 4, 5, 6, 2, 2, 4, 5, 6, 6, 5, 1, 2, 3, 4, 5, 6, 1, 2, 3, 5, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 3, 6, 1, 2, 3, 4, 5, 2, 6, 6, 4, 4, 5, 2, 2, 4, 6, 2, 5, 1, 2, 3, 4, 5, 6, 1, 2, 3, 5, 6, 6, 4, 3, 1, 2, 3, 6, 4, 1, 2, 3, 5, 1, 4, 6, 1, 2, 6, 1, 6, 3, 6, 6, 1, 2, 3, 4, 5, 6, 1, 1, 2, 3, 1, 2, 4, 1, 4, 2, 3, 4, 5, 6, 1, 2, 3, 5, 2, 2, 1, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 2, 3, 5, 6, 1, 6, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 2, 2, 6, 1, 2, 3, 5, 3, 4, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 3, 2, 3, 3, 1, 3, 1, 3, 1, 3, 5, 1, 2, 3, 4, 5, 6, 2, 2, 3, 5, 6, 6, 1, 2, 3, 4, 5, 6, 3, 3, 4, 6, 5, 1, 6, 2, 3, 4, 5, 4, 1, 2, 3, 4, 5, 6, 1, 2, 3, 6, 2, 3, 4, 6, 4, 5, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 5, 4, 5, 1, 2, 1, 2, 3, 4, 5, 6, 1, 2, 6, 2, 2, 1, 2, 3, 4, 5, 6, 2], "Freq": [0.7885686755180359, 0.11265266686677933, 0.07510177791118622, 0.08519116789102554, 0.08519116789102554, 0.04259558394551277, 0.1277867555618286, 0.6389337778091431, 0.08186793327331543, 0.08186793327331543, 0.16373586654663086, 0.08186793327331543, 0.20466983318328857, 0.36840569972991943, 0.6738352179527283, 0.16845880448818207, 0.8268033862113953, 0.05502315238118172, 0.5135494470596313, 0.3851620554924011, 0.018341051414608955, 0.05502315238118172, 0.09500807523727417, 0.8550726771354675, 0.008060098625719547, 0.2579231560230255, 0.1692620813846588, 0.15314188599586487, 0.3788246512413025, 0.03224039450287819, 0.1466446816921234, 0.34217095375061035, 0.3177301585674286, 0.08554273843765259, 0.024440782144665718, 0.09165292978286743, 0.10407788306474686, 0.8326230645179749, 0.7566118240356445, 0.20634867250919342, 0.13759484887123108, 0.3095884323120117, 0.3267877697944641, 0.05159807205200195, 0.05159807205200195, 0.12039549648761749, 0.9198042154312134, 0.8792956471443176, 0.9183156490325928, 0.8558183312416077, 0.9678038358688354, 0.8268380761146545, 0.919150710105896, 0.8268091678619385, 0.1587556004524231, 0.063502237200737, 0.2222578376531601, 0.3492622971534729, 0.063502237200737, 0.127004474401474, 0.15224319696426392, 0.6597204804420471, 0.10149545967578888, 0.05074772983789444, 0.8401808142662048, 0.18003873527050018, 0.7503100037574768, 0.8303558826446533, 0.2812216281890869, 0.6561837792396545, 0.9125121235847473, 0.2921493649482727, 0.23371949791908264, 0.2921493649482727, 0.02921493723988533, 0.14607468247413635, 0.8268309831619263, 0.09038255363702774, 0.18076510727405548, 0.3389345705509186, 0.04519127681851387, 0.18076510727405548, 0.1581694632768631, 0.24918930232524872, 0.747567892074585, 0.8791689276695251, 0.049813397228717804, 0.298880398273468, 0.024906698614358902, 0.149440199136734, 0.12453349679708481, 0.3237870931625366, 0.06989113241434097, 0.16307930648326874, 0.20967338979244232, 0.02329704351723194, 0.16307930648326874, 0.37275269627571106, 0.03576187416911125, 0.28609499335289, 0.4649043679237366, 0.143047496676445, 0.03576187416911125, 0.03576187416911125, 0.8268415927886963, 0.8266354203224182, 0.11021805554628372, 0.706777811050415, 0.10096826404333115, 0.05048413202166557, 0.15145239233970642, 0.7436269521713257, 0.9655224084854126, 0.20307330787181854, 0.10153665393590927, 0.020307330414652824, 0.18276597559452057, 0.487375944852829, 0.5510147213935852, 0.1377536803483963, 0.1377536803483963, 0.1377536803483963, 0.17599421739578247, 0.7039768695831299, 0.05861835181713104, 0.5471045970916748, 0.17585505545139313, 0.03907889872789383, 0.05861835181713104, 0.1367761492729187, 0.1699335128068924, 0.5098005533218384, 0.2549002766609192, 0.02124168910086155, 0.0424833782017231, 0.12408846616744995, 0.2978123128414154, 0.08686192333698273, 0.17372384667396545, 0.062044233083724976, 0.2605857849121094, 0.05070082098245621, 0.5408087372779846, 0.23660382628440857, 0.01690027303993702, 0.11830191314220428, 0.05070082098245621, 0.6385400295257568, 0.6385513544082642, 0.6170589327812195, 0.2056863158941269, 0.13185174763202667, 0.5933328866958618, 0.13185174763202667, 0.1977776139974594, 0.021062834188342094, 0.9688904285430908, 0.826723039150238, 0.9364690184593201, 0.8861609697341919, 0.8171838521957397, 0.054507531225681305, 0.054507531225681305, 0.21803012490272522, 0.1635226011276245, 0.49056777358055115, 0.8792324066162109, 0.12309018522500992, 0.6769960522651672, 0.06154509261250496, 0.12309018522500992, 0.8303563594818115, 0.22208422422409058, 0.1727321743965149, 0.24676024913787842, 0.049352049827575684, 0.22208422422409058, 0.12338012456893921, 0.9286288619041443, 0.8792188167572021, 0.14204417169094086, 0.047348055988550186, 0.8049169182777405, 0.9320809245109558, 0.026762790977954865, 0.3211534917354584, 0.40144187211990356, 0.05352558195590973, 0.0802883729338646, 0.13381396234035492, 0.11239094287157059, 0.2729494273662567, 0.1284467875957489, 0.2729494273662567, 0.08027924597263336, 0.1445026397705078, 0.8790401816368103, 0.8303582668304443, 0.8268256783485413, 0.9364730715751648, 0.18750955164432526, 0.03750191256403923, 0.5250267386436462, 0.03750191256403923, 0.18750955164432526, 0.8792299032211304, 0.019262980669736862, 0.10594639927148819, 0.7416247725486755, 0.09631490707397461, 0.028894472867250443, 0.1062571331858635, 0.850057065486908, 0.8647724390029907, 0.12932749092578888, 0.7759649753570557, 0.06466374546289444, 0.9190805554389954, 0.8267297148704529, 0.09395257383584976, 0.29527950286865234, 0.2818577289581299, 0.10737437009811401, 0.026843592524528503, 0.20132693648338318, 0.8647952675819397, 0.20773600041866302, 0.6232079863548279, 0.24918198585510254, 0.7475459575653076, 0.2486257553100586, 0.7458772659301758, 0.11196127533912659, 0.11196127533912659, 0.6717676520347595, 0.8056323528289795, 0.11509034037590027, 0.8303669095039368, 0.1736481934785843, 0.1736481934785843, 0.5209445953369141, 0.1131211370229721, 0.16968169808387756, 0.6598733067512512, 0.037707045674324036, 0.018853522837162018, 0.6385417580604553, 0.03831297904253006, 0.17240840196609497, 0.06704770773649216, 0.10536068677902222, 0.4980687201023102, 0.12451718002557755, 0.09740452468395233, 0.19480904936790466, 0.2922135591506958, 0.09740452468395233, 0.32468175888061523, 0.8793113827705383, 0.8812735080718994, 0.6379014849662781, 0.21263383328914642, 0.7290288805961609, 0.20829397439956665, 0.05207349359989166, 0.9364842176437378, 0.9549100995063782, 0.7435711026191711, 0.722694456577301, 0.14453887939453125, 0.8303695917129517, 0.8114745020866394, 0.11592493206262589, 0.11592493206262589, 0.8303502202033997, 0.9031682014465332, 0.9183188080787659, 0.722140908241272, 0.22219718992710114, 0.24334749579429626, 0.14600850641727448, 0.19467799365520477, 0.024334749206900597, 0.09733899682760239, 0.29201701283454895, 0.8792062401771545, 0.060763388872146606, 0.20254462957382202, 0.26330801844596863, 0.07089062035083771, 0.14178124070167542, 0.26330801844596863, 0.157303586602211, 0.7865179777145386, 0.0786517933011055, 0.16601526737213135, 0.6640610694885254, 0.11159108579158783, 0.07439406216144562, 0.14878812432289124, 0.03719703108072281, 0.6323494911193848, 0.03719703108072281, 0.12823760509490967, 0.5556963086128235, 0.29922106862068176, 0.8790940642356873, 0.8267346620559692, 0.8268254995346069, 0.8267253637313843, 0.7503020763397217, 0.886148989200592, 0.339229553937912, 0.11307652294635773, 0.11307652294635773, 0.4523060917854309, 0.024572204798460007, 0.30715256929397583, 0.38086917996406555, 0.024572204798460007, 0.09828881919384003, 0.14743323624134064, 0.16302454471588135, 0.45646873116493225, 0.3586540222167969, 0.19612474739551544, 0.09806237369775772, 0.44128069281578064, 0.02451559342443943, 0.09806237369775772, 0.19612474739551544, 0.05344098433852196, 0.3072856664657593, 0.293925404548645, 0.05344098433852196, 0.09352172166109085, 0.1870434433221817, 0.826849639415741, 0.7724860906600952, 0.7992817163467407, 0.17761816084384918, 0.167043998837471, 0.2624976933002472, 0.167043998837471, 0.28636112809181213, 0.09545370936393738, 0.023863427340984344, 0.5798622965812683, 0.09664371609687805, 0.28993114829063416, 0.9286113381385803, 0.9353047609329224, 0.8268387913703918, 0.8303583264350891, 0.9181744456291199, 0.07651453465223312, 0.9123748540878296, 0.166093111038208, 0.664372444152832, 0.8861731886863708, 0.3172447681427002, 0.1268979012966156, 0.1268979012966156, 0.4441426694393158, 0.13551858067512512, 0.27103716135025024, 0.5420743227005005, 0.019376110285520554, 0.3487699627876282, 0.2906416356563568, 0.019376110285520554, 0.2131372094154358, 0.09688054770231247, 0.6253291964530945, 0.2084430754184723, 0.6386271715164185, 0.16968916356563568, 0.3030163645744324, 0.1515081822872162, 0.10302557051181793, 0.07272393256425858, 0.1939304769039154, 0.11482307314872742, 0.4133630692958832, 0.3215046226978302, 0.11482307314872742, 0.022964615374803543, 0.8267960548400879, 0.6987345218658447, 0.23291151225566864, 0.21269448101520538, 0.6380833983421326, 0.04690447449684143, 0.2814268469810486, 0.4924969971179962, 0.04690447449684143, 0.07035671174526215, 0.07035671174526215, 0.879239559173584, 0.22145836055278778, 0.11072918027639389, 0.276822954416275, 0.055364590138196945, 0.3875521421432495, 0.06608932465314865, 0.1321786493062973, 0.5948039293289185, 0.19826796650886536, 0.8202731013298035, 0.11718187481164932, 0.8948036432266235, 0.10703093558549881, 0.5351547002792358, 0.18730413913726807, 0.1605464071035385, 0.930776059627533, 0.6385695338249207, 0.8268303275108337, 0.9352853298187256, 0.8320673704147339, 0.05200421065092087, 0.05200421065092087, 0.9736712574958801, 0.9023493528366089, 0.8201534152030945, 0.7435756325721741, 0.9023573398590088, 0.9224088191986084, 0.44210851192474365, 0.2526334524154663, 0.2526334524154663, 0.18033291399478912, 0.23443278670310974, 0.21639949083328247, 0.09016645699739456, 0.10819974541664124, 0.16229961812496185, 0.2040615975856781, 0.6121847629547119, 0.3002542555332184, 0.161675363779068, 0.2540612816810608, 0.0692894458770752, 0.0346447229385376, 0.173223614692688, 0.6385626792907715, 0.9364643096923828, 0.0980236828327179, 0.3920947313308716, 0.13069824874401093, 0.3267455995082855, 0.06534912437200546, 0.7686494588851929, 0.1299188733100891, 0.0779513269662857, 0.1818864345550537, 0.3118053078651428, 0.2598377466201782, 0.1313001811504364, 0.2751051187515259, 0.2751051187515259, 0.043766725808382034, 0.10003823041915894, 0.17506690323352814, 0.28249338269233704, 0.14124669134616852, 0.14124669134616852, 0.04708223044872284, 0.14124669134616852, 0.2354111522436142, 0.32399433851242065, 0.16199716925621033, 0.05399905517697334, 0.32399433851242065, 0.10799811035394669, 0.13811370730400085, 0.8286822438240051, 0.6058682799339294, 0.40391218662261963, 0.11937636882066727, 0.31833699345588684, 0.23875273764133453, 0.06366739422082901, 0.11141794919967651, 0.14325164258480072, 0.6385700106620789, 0.13912592828273773, 0.510128378868103, 0.18550123274326324, 0.04637530818581581, 0.07729218155145645, 0.03091687150299549, 0.8303558230400085, 0.14782723784446716, 0.16894541680812836, 0.12670905888080597, 0.12670905888080597, 0.16894541680812836, 0.25341811776161194, 0.8558087944984436, 0.9320377111434937, 0.7312690615653992, 0.2742258906364441, 0.7307645082473755, 0.18269112706184387, 0.7147846817970276, 0.1786961704492569, 0.20778053998947144, 0.6233416199684143, 0.8268316388130188, 0.8861678242683411, 0.8791713118553162, 0.8303482532501221, 0.11991258710622787, 0.26980331540107727, 0.26980331540107727, 0.10492351651191711, 0.14989073574543, 0.074945367872715, 0.8303607106208801, 0.9833444952964783, 0.9556792378425598, 0.936531662940979, 0.10623903572559357, 0.17706505954265594, 0.3895431160926819, 0.24789108335971832, 0.07082602381706238, 0.8792172074317932, 0.7855403423309326, 0.13092339038848877, 0.9380425810813904, 0.9549534320831299, 0.9677981734275818, 0.11131705343723297, 0.5565852522850037, 0.07421136647462845, 0.18552842736244202, 0.11131705343723297, 0.981268584728241, 0.1362214982509613, 0.045407168567180634, 0.18162867426872253, 0.18162867426872253, 0.45407167077064514, 0.638551652431488, 0.8268316388130188, 0.8558236360549927, 0.6385677456855774, 0.826836109161377, 0.8795265555381775, 0.6253267526626587, 0.20844224095344543, 0.27150771021842957, 0.5430154204368591, 0.20363079011440277, 0.29286476969718933, 0.036608096212148666, 0.036608096212148666, 0.6223376393318176, 0.9654844999313354, 0.15629971027374268, 0.260499507188797, 0.3907492756843567, 0.013024975545704365, 0.02604995109140873, 0.15629971027374268, 0.8647721409797668, 0.08412960916757584, 0.14021602272987366, 0.2804320454597473, 0.2523888349533081, 0.2243456244468689, 0.06906165182590485, 0.8287398219108582, 0.06906165182590485, 0.30277836322784424, 0.6055567264556885, 0.12979456782341003, 0.8436647057533264, 0.7147122025489807, 0.17867805063724518, 0.8268413543701172, 0.7434741854667664, 0.02810440957546234, 0.25293970108032227, 0.44967055320739746, 0.02810440957546234, 0.08431322872638702, 0.1967308670282364, 0.7503163814544678, 0.5446216464042664, 0.3403885066509247, 0.0680777058005333, 0.0680777058005333, 0.44324439764022827, 0.132973313331604, 0.3989199697971344, 0.15448498725891113, 0.7724249362945557, 0.05143851414322853, 0.5658236742019653, 0.3086310923099518, 0.025719257071614265, 0.025719257071614265, 0.05143851414322853, 0.9190574884414673, 0.3900164067745209, 0.34126436710357666, 0.04875205084681511, 0.19500820338726044, 0.8268325328826904, 0.8861598968505859, 0.30403628945350647, 0.08686751127243042, 0.30403628945350647, 0.08686751127243042, 0.04343375563621521, 0.21716877818107605, 0.6133686304092407, 0.08762408792972565, 0.1752481758594513, 0.08762408792972565, 0.0727054551243782, 0.4362327456474304, 0.1090581864118576, 0.254469096660614, 0.0727054551243782, 0.0727054551243782, 0.1316942572593689, 0.36215919256210327, 0.24692672491073608, 0.016461782157421112, 0.08230891078710556, 0.1481560319662094, 0.6192945837974548, 0.1548236459493637, 0.1548236459493637, 0.4812731444835663, 0.06875330954790115, 0.10312996059656143, 0.06875330954790115, 0.2750132381916046, 0.9190259575843811, 0.8268333673477173, 0.826833188533783, 0.9022716283798218, 0.1383085697889328, 0.8298513889312744, 0.8647831082344055, 0.09539901465177536, 0.09539901465177536, 0.7631921172142029, 0.8828222155570984, 0.08828222751617432, 0.10746899992227554, 0.13817442953586578, 0.2302907109260559, 0.04605814069509506, 0.39917057752609253, 0.0767635703086853, 0.11917077004909515, 0.21450738608837128, 0.35751232504844666, 0.11917077004909515, 0.19067323207855225, 0.9320213794708252, 0.9678043127059937, 0.8800909519195557, 0.08282633125782013, 0.7454370260238647, 0.08282633125782013, 0.08282633125782013, 0.8587790727615356, 0.8303593993186951, 0.08045659214258194, 0.32182636857032776, 0.5631961822509766, 0.1791025847196579, 0.1791025847196579, 0.5373077988624573, 0.47338640689849854, 0.33813315629959106, 0.2028798907995224, 0.20776869356632233, 0.6233060956001282, 0.17578920722007751, 0.7031568288803101, 0.826840341091156, 0.05989230424165726, 0.11978460848331451, 0.4192461371421814, 0.02994615212082863, 0.26951536536216736, 0.11978460848331451, 0.830357015132904, 0.5204680562019348, 0.34697872400283813, 0.8792226910591125, 0.5452080368995667, 0.2726040184497833, 0.13630200922489166, 0.7548964023590088, 0.25163212418556213, 0.2122674137353897, 0.10613370686769485, 0.15920056402683258, 0.05306685343384743, 0.4245348274707794, 0.10635678470134735, 0.10635678470134735, 0.10635678470134735, 0.6381406784057617, 0.8647982478141785, 0.8201994299888611, 0.16272194683551788, 0.6508877873420715, 0.054240647703409195, 0.10848129540681839, 0.054240647703409195, 0.1451577991247177, 0.3193471431732178, 0.27579981088638306, 0.058063115924596786, 0.014515778981149197, 0.18870513141155243, 0.8303703665733337, 0.08196701854467392, 0.06147526577115059, 0.5942609310150146, 0.04098350927233696, 0.1434422880411148, 0.06147526577115059, 0.1860005110502243, 0.12161572277545929, 0.33623170852661133, 0.06438479572534561, 0.01430773176252842, 0.2861546277999878, 0.5063105225563049, 0.07233007252216339, 0.19288019835948944, 0.19288019835948944, 0.8303716778755188, 0.8301306366920471, 0.9190161824226379, 0.8790932297706604, 0.2119527906179428, 0.2628214657306671, 0.2882557809352875, 0.03391244634985924, 0.07630300521850586, 0.12717166543006897, 0.6368348598480225, 0.31841742992401123, 0.919861376285553, 0.13696692883968353, 0.8218015432357788, 0.033716779202222824, 0.6069020628929138, 0.33716779947280884, 0.9315099120140076, 0.879214882850647, 0.9652835130691528, 0.1678917557001114, 0.19187629222869873, 0.31179895997047424, 0.04796907305717468, 0.21586082875728607, 0.04796907305717468, 0.022406795993447304, 0.2912883460521698, 0.31369513273239136, 0.022406795993447304, 0.11203397810459137, 0.2464747428894043, 0.8804880380630493, 0.16610050201416016, 0.6644020080566406, 0.9353597164154053, 0.484325110912323, 0.484325110912323, 0.8303671479225159, 0.9370070695877075, 0.0732511579990387, 0.805762767791748, 0.0732511579990387, 0.10029350966215134, 0.18625937402248383, 0.4155016839504242, 0.014327644370496273, 0.15760408341884613, 0.11462115496397018, 0.9549100995063782, 0.10016454756259918, 0.35057592391967773, 0.4507404863834381, 0.05008227378129959, 0.8267979621887207, 0.13213543593883514, 0.3964063227176666, 0.18498961627483368, 0.05285417661070824, 0.09249480813741684, 0.14534898102283478, 0.8791918754577637, 0.6920285820960999, 0.0988612249493599, 0.1977224498987198, 0.7435709238052368, 0.6960877180099487, 0.1392175555229187, 0.08268050104379654, 0.20670124888420105, 0.45474275946617126, 0.24804149568080902, 0.768653154373169, 0.025175679475069046, 0.25175678730010986, 0.40281087160110474, 0.05035135895013809, 0.15105406939983368, 0.10070271790027618, 0.09013494104146957, 0.18026988208293915, 0.6309446096420288, 0.09013494104146957, 0.8061065077781677, 0.0767720490694046, 0.0383860245347023, 0.0767720490694046, 0.7686532139778137, 0.9665476083755493, 0.09120580554008484, 0.2508159577846527, 0.2736174166202545, 0.04560290277004242, 0.2166137844324112, 0.13680870831012726, 0.09031174331903458, 0.18062348663806915, 0.5870263576507568, 0.04515587165951729, 0.04515587165951729, 0.04515587165951729, 0.026020633056759834, 0.5334229469299316, 0.06505157798528671, 0.0910722091794014, 0.2732166349887848, 0.013010316528379917, 0.7436647415161133, 0.8587563037872314, 0.08587563037872314, 0.1282627135515213, 0.7695762515068054, 0.1075681746006012, 0.1075681746006012, 0.2151363492012024, 0.3764886260032654, 0.1613522619009018, 0.0537840873003006, 0.08270671963691711, 0.7443605065345764, 0.08270671963691711, 0.8876500129699707, 0.9548994898796082, 0.022710148245096207, 0.49962326884269714, 0.3406522274017334, 0.022710148245096207, 0.022710148245096207, 0.09084059298038483, 0.954899787902832], "Term": ["abv", "abv", "abv", "acid", "acid", "acid", "acid", "acid", "actually", "actually", "actually", "actually", "actually", "actually", "adjunct", "adjunct", "affordable", "age", "age", "age", "age", "age", "airport", "airport", "alcohol", "alcohol", "alcohol", "alcohol", "alcohol", "alcohol", "ale", "ale", "ale", "ale", "ale", "ale", "alpha", "alpha", "amber", "amber", "american", "american", "american", "american", "american", "american", "anywhere", "approach", "april", "arousal", "august", "austurstr\u00e6ti", "awful", "baderbrau", "bar", "bar", "bar", "bar", "bar", "bar", "barley", "barley", "barley", "barley", "barrel", "barrel", "barroom", "beauty", "beck", "beck", "beertini", "belgian", "belgian", "belgian", "belgian", "belgian", "bell", "best", "best", "best", "best", "best", "best", "beta", "beta", "birra", "bitter", "bitter", "bitter", "bitter", "bitter", "bitter", "bitterness", "bitterness", "bitterness", "bitterness", "bitterness", "bitterness", "black", "black", "black", "black", "black", "black", "blah", "blend", "blend", "boil", "boil", "boil", "boil", "boost", "booze", "bottle", "bottle", "bottle", "bottle", "bottle", "break", "break", "break", "break", "brett", "brett", "brew", "brew", "brew", "brew", "brew", "brew", "brewer", "brewer", "brewer", "brewer", "brewer", "brewery", "brewery", "brewery", "brewery", "brewery", "brewery", "brewing", "brewing", "brewing", "brewing", "brewing", "brewing", "bruggh\u00fas", "bruggsmi\u00f0jan", "buckwheat", "buckwheat", "buy", "buy", "buy", "buy", "calorie", "calorie", "carbohydrate", "castelvetrano", "cederquist", "chipotle", "cider", "cider", "cider", "cider", "cider", "clarex", "cold", "cold", "cold", "cold", "colorful", "come", "come", "come", "come", "come", "come", "complement", "considerable", "contains", "contains", "contains", "corn", "could", "could", "could", "could", "could", "could", "craft", "craft", "craft", "craft", "craft", "craft", "creamy", "crew", "cup", "dakota", "dark", "dark", "dark", "dark", "dark", "data", "day", "day", "day", "day", "day", "december", "december", "decision", "degree", "degree", "degree", "depends", "dessert", "different", "different", "different", "different", "different", "different", "differentiates", "dimension", "dimension", "direct", "direct", "disappointed", "disappointed", "dish", "dish", "dish", "distil", "distil", "distribution", "doppelbock", "doppelbock", "doppelbock", "double", "double", "double", "double", "double", "downtown", "drink", "drink", "drink", "drink", "drink", "drink", "drinking", "drinking", "drinking", "drinking", "drinking", "dubbels", "dull", "einst\u00f6k", "einst\u00f6k", "english", "english", "english", "equis", "evil", "excuse", "expensive", "expensive", "explosm", "extract", "extract", "extract", "father", "fear", "february", "ferment", "ferment", "find", "find", "find", "find", "find", "find", "flanders", "flavor", "flavor", "flavor", "flavor", "flavor", "flavor", "flight", "flight", "flight", "foam", "foam", "food", "food", "food", "food", "food", "food", "free", "free", "free", "fridge", "fry", "funny", "gain", "garde", "gate", "german", "german", "german", "german", "get", "get", "get", "get", "get", "get", "gluten", "gluten", "gluten", "go", "go", "go", "go", "go", "go", "good", "good", "good", "good", "good", "good", "google", "gram", "grape", "grape", "great", "great", "great", "great", "great", "great", "green", "green", "green", "grill", "guiness", "g\u00e6dingur", "han", "hangover", "hangover", "harvest", "heady", "heady", "health", "heavy", "heavy", "heavy", "heavy", "heineken", "heineken", "heineken", "high", "high", "high", "high", "high", "high", "historical", "historical", "homebrew", "hop", "hop", "hop", "hop", "hop", "hop", "hoppy", "hoppy", "hoppy", "hoppy", "hoppy", "hybrid", "icelandic", "icelandic", "impact", "impact", "imperial", "imperial", "imperial", "imperial", "imperial", "imperial", "imply", "include", "include", "include", "include", "include", "increase", "increase", "increase", "increase", "indian", "indian", "individual", "ingredient", "ingredient", "ingredient", "ingredient", "int", "intoxication", "isohumulones", "italy", "jalapeno", "jalapeno", "jalapeno", "jalape\u00f1o", "january", "japanese", "journal", "july", "june", "kind", "kind", "kind", "know", "know", "know", "know", "know", "know", "kolsch", "kolsch", "lager", "lager", "lager", "lager", "lager", "lager", "latency", "lb", "less", "less", "less", "less", "less", "libido", "light", "light", "light", "light", "light", "like", "like", "like", "like", "like", "like", "little", "little", "little", "little", "little", "little", "local", "local", "local", "local", "local", "loss", "loss", "major", "major", "make", "make", "make", "make", "make", "make", "male", "malt", "malt", "malt", "malt", "malt", "malt", "mania", "many", "many", "many", "many", "many", "many", "march", "mbt", "men", "men", "micro", "micro", "midwest", "midwest", "miller", "miller", "min", "moderate", "moment", "monastery", "much", "much", "much", "much", "much", "much", "m\u00fcller", "national", "nbsp", "net", "new", "new", "new", "new", "new", "nobody", "notice", "notice", "november", "oak", "october", "old", "old", "old", "old", "old", "olive", "one", "one", "one", "one", "one", "operating", "opposite", "orgasm", "orgasmic", "outmeal", "overcarbonated", "oxidation", "oxidation", "pack", "pack", "pack", "pair", "pair", "pair", "pair", "pairing", "pale", "pale", "pale", "pale", "pale", "pale", "pennsylvania", "people", "people", "people", "people", "people", "per", "per", "per", "perceive", "perceive", "percent", "percent", "perfect", "perfect", "phrase", "pork", "porter", "porter", "porter", "porter", "porter", "porter", "prague", "prefer", "prefer", "prefer", "prefer", "pretty", "pretty", "pretty", "primary", "primary", "process", "process", "process", "process", "process", "process", "producer", "product", "product", "product", "product", "progress", "proof", "quite", "quite", "quite", "quite", "quite", "quite", "ratebeer", "ratebeer", "ratebeer", "ratebeer", "real", "real", "real", "real", "real", "real", "really", "really", "really", "really", "really", "really", "recommendation", "recommendation", "recommendation", "red", "red", "red", "red", "red", "retire", "revival", "reykjavik", "reykjav\u00edk", "rich", "rich", "row", "rye", "rye", "rye", "sake", "sake", "say", "say", "say", "say", "say", "say", "see", "see", "see", "see", "see", "sensitive", "september", "serious", "session", "session", "session", "session", "sexual", "sherry", "shot", "shot", "shot", "significantly", "significantly", "significantly", "single", "single", "single", "skunky", "skunky", "slight", "slight", "snob", "something", "something", "something", "something", "something", "something", "sommelierbier", "soon", "soon", "sorghum", "sort", "sort", "sort", "south", "south", "specific", "specific", "specific", "specific", "specific", "spicy", "spicy", "spicy", "spicy", "spoil", "starch", "store", "store", "store", "store", "store", "stout", "stout", "stout", "stout", "stout", "stout", "strategy", "strong", "strong", "strong", "strong", "strong", "strong", "style", "style", "style", "style", "style", "style", "sugar", "sugar", "sugar", "sugar", "sugary", "sunlight", "suppose", "surprisingly", "taste", "taste", "taste", "taste", "taste", "taste", "tea", "tea", "technology", "tell", "tell", "temperature", "temperature", "temperature", "terminal", "terminology", "testosterone", "thing", "thing", "thing", "thing", "thing", "thing", "think", "think", "think", "think", "think", "think", "thumb", "topper", "topper", "track", "trappist", "trappist", "tripel", "tripels", "triple", "triple", "triple", "try", "try", "try", "try", "try", "try", "twin", "typically", "typically", "typically", "typically", "urbana", "use", "use", "use", "use", "use", "use", "vacuum", "variant", "variant", "variant", "vermouth", "vienna", "vienna", "vodka", "vodka", "vodka", "vodka", "vodkabeer", "want", "want", "want", "want", "want", "want", "warm", "warm", "warm", "warm", "water", "water", "water", "water", "weekend", "weight", "well", "well", "well", "well", "well", "well", "wheat", "wheat", "wheat", "wheat", "wheat", "wheat", "wine", "wine", "wine", "wine", "wine", "wine", "wise", "woman", "woman", "wood", "wood", "world", "world", "world", "world", "world", "world", "wort", "wort", "wort", "xpa", "yang", "yeast", "yeast", "yeast", "yeast", "yeast", "yeast", "yin"]}, "R": 30, "lambda.step": 0.01, "plot.opts": {"xlab": "PC1", "ylab": "PC2"}, "topic.order": [1, 2, 3, 4, 5, 6]};

function LDAvis_load_lib(url, callback){
  var s = document.createElement('script');
  s.src = url;
  s.async = true;
  s.onreadystatechange = s.onload = callback;
  s.onerror = function(){console.warn("failed to load library " + url);};
  document.getElementsByTagName("head")[0].appendChild(s);
}

if(typeof(LDAvis) !== "undefined"){
   // already loaded: just create the visualization
   !function(LDAvis){
       new LDAvis("#" + "ldavis_el113841406024480712488855019749", ldavis_el113841406024480712488855019749_data);
   }(LDAvis);
}else if(typeof define === "function" && define.amd){
   // require.js is available: use it to load d3/LDAvis
   require.config({paths: {d3: "https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.5/d3.min"}});
   require(["d3"], function(d3){
      window.d3 = d3;
      LDAvis_load_lib("https://cdn.rawgit.com/bmabey/pyLDAvis/files/ldavis.v1.0.0.js", function(){
        new LDAvis("#" + "ldavis_el113841406024480712488855019749", ldavis_el113841406024480712488855019749_data);
      });
    });
}else{
    // require.js not available: dynamically load d3 & LDAvis
    LDAvis_load_lib("https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.5/d3.min.js", function(){
         LDAvis_load_lib("https://cdn.rawgit.com/bmabey/pyLDAvis/files/ldavis.v1.0.0.js", function(){
                 new LDAvis("#" + "ldavis_el113841406024480712488855019749", ldavis_el113841406024480712488855019749_data);
            })
         });
}
</script>



Inferring from the keywords and the relative positions of the topics, we can gather:

   * **Topic 2** seems to be the most prominent, which seems to be about the brewing method  using hop and malt
   * **Topic 1** is the next biggest topic, which seems to be around different styles of beer and how IPA's hoppiness gives a very different taste
   * **Topic 4** is a topic that is very different from the rest; it seems about IPA's alcohol content (ABV) - it's of course tied to calorie
   * **Topic 3** is another different topic far from the rest, which could be about the best IPA age (how long to ferment, and how soon you should drink it after it's bottled 

### Sentiment Analysis
[Back to top](#Table-of-Content)

To understand the sentiment of the IPA related posts and comments, we will use the pre-trained `Sentiment Intensity Analyzer` from NLTK.


```python
#initiate pre-trained
sia = SentimentIntensityAnalyzer()
```


```python
#function to extract compound score - sum of positive, negative, neutral score and normalize to be between -1 and 1
def get_compound_score(text):
    
    result=sia.polarity_scores(text)
    compound=result.get('compound')
    return compound
```


```python
df_ipa['SentimentScore'] = df_ipa['CleanVerbatim'].apply(lambda x: get_compound_score(x))
```


```python
df_ipa['Score_1']=df_ipa['Score'].astype(int)+1 #add one to avoid multiplying zero
df_ipa['SentimentScore_weighted']=df_ipa['SentimentScore']*(df_ipa['Score_1'])
```

We will create a weighted sentiment score taking into account the number **up** votes and **down** votes the posts received. Naturally, if the post has a positive sentiment, but the general public is down voting it, it could suggest that the overall sentiment is negative, and vice versa.


```python
df_ipa_sentiment=df_ipa[['CreationMonth','SentimentScore_weighted', 'Score_1']].groupby('CreationMonth').sum().reset_index()
df_ipa_sentiment['SentimentScore_adjusted']=df_ipa_sentiment['SentimentScore_weighted']/df_ipa_sentiment['Score_1']
df_ipa_sentiment.plot(x='CreationMonth',y='SentimentScore_adjusted', figsize=(14,6));
```


![png](IPA_sentiment_trend.png)


It seems the sentiment towards IPA has been overall positive, with only May 2016 and July 2017 dipping into the negative. It's possible that these two months only have a few posts and thus one negative one might drag the whole month down. Further investigation into the data reveals that:

**May-2016:**
   * Two posts were made this month, one is about mixing beer with other alcohol or non alcoholic drink to create a beer cocktail - overall neutral. The other post was a complaint on the IBU (International Bitterness Units) measuring system, which received quite a few up-votes. The general consensus is that there are a lot IPAs that are rated in the 40s or 50s for IBU, but in practice they are much more bitter than stouts in the 80+ range. The rating seems to not be aligned with the hoppiness and the actual taste of the beer.

**July-2017:**
   * Only 1 comment is added this month mentioning IPA; it's about mixing vodka with beer, the comment was "Put some low quality vodka in my triple IPA and I was not disappointed", which suggest it's misclassified negative sentiment 


```python

```
