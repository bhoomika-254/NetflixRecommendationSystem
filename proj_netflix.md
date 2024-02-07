```python
import numpy as np
import pandas as pd
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import cosine_similarity
```


```python
data=pd.read_csv("netflixData.csv")
print(data.head())
```

                                    Show Id                          Title  \
    0  cc1b6ed9-cf9e-4057-8303-34577fb54477                       (Un)Well   
    1  e2ef4e91-fb25-42ab-b485-be8e3b23dedb                         #Alive   
    2  b01b73b7-81f6-47a7-86d8-acb63080d525  #AnneFrank - Parallel Stories   
    3  b6611af0-f53c-4a08-9ffa-9716dc57eb9c                       #blackAF   
    4  7f2d4170-bab8-4d75-adc2-197f7124c070               #cats_the_mewvie   
    
                                             Description  \
    0  This docuseries takes a deep dive into the luc...   
    1  As a grisly virus rampages a city, a lone man ...   
    2  Through her diary, Anne Frank's story is retol...   
    3  Kenya Barris and his family navigate relations...   
    4  This pawesome documentary explores how our fel...   
    
                          Director  \
    0                          NaN   
    1                       Cho Il   
    2  Sabina Fedeli, Anna Migotto   
    3                          NaN   
    4             Michael Margolis   
    
                                               Genres  \
    0                                      Reality TV   
    1  Horror Movies, International Movies, Thrillers   
    2             Documentaries, International Movies   
    3                                     TV Comedies   
    4             Documentaries, International Movies   
    
                                                    Cast Production Country  \
    0                                                NaN      United States   
    1                           Yoo Ah-in, Park Shin-hye        South Korea   
    2                        Helen Mirren, Gengher Gatti              Italy   
    3  Kenya Barris, Rashida Jones, Iman Benson, Genn...      United States   
    4                                                NaN             Canada   
    
       Release Date Rating  Duration Imdb Score Content Type         Date Added  
    0        2020.0  TV-MA  1 Season     6.6/10      TV Show                NaN  
    1        2020.0  TV-MA    99 min     6.2/10        Movie  September 8, 2020  
    2        2019.0  TV-14    95 min     6.4/10        Movie       July 1, 2020  
    3        2020.0  TV-MA  1 Season     6.6/10      TV Show                NaN  
    4        2020.0  TV-14    90 min     5.1/10        Movie   February 5, 2020  
    


```python
print(data.isnull().sum())
```

    Show Id                  0
    Title                    0
    Description              0
    Director              2064
    Genres                   0
    Cast                   530
    Production Country     559
    Release Date             3
    Rating                   4
    Duration                 3
    Imdb Score             608
    Content Type             0
    Date Added            1335
    dtype: int64
    


```python
data=data[["Title","Description","Genres","Content Type"]]
print(data.head())
```

                               Title  \
    0                       (Un)Well   
    1                         #Alive   
    2  #AnneFrank - Parallel Stories   
    3                       #blackAF   
    4               #cats_the_mewvie   
    
                                             Description  \
    0  This docuseries takes a deep dive into the luc...   
    1  As a grisly virus rampages a city, a lone man ...   
    2  Through her diary, Anne Frank's story is retol...   
    3  Kenya Barris and his family navigate relations...   
    4  This pawesome documentary explores how our fel...   
    
                                               Genres Content Type  
    0                                      Reality TV      TV Show  
    1  Horror Movies, International Movies, Thrillers        Movie  
    2             Documentaries, International Movies        Movie  
    3                                     TV Comedies      TV Show  
    4             Documentaries, International Movies        Movie  
    


```python
print(data.columns)
```

    Index(['Title', 'Description', 'Genres', 'Content Type'], dtype='object')
    


```python
print(data.isnull().sum())
```

    Title           0
    Description     0
    Genres          0
    Content Type    0
    dtype: int64
    


```python
data=data.dropna()
```


```python
import nltk
import re
nltk.download("stopwords")
stemmer=nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
import string
stopword = set(stopwords.words("english"))
```

    [nltk_data] Downloading package stopwords to
    [nltk_data]     C:\Users\bhoom\AppData\Roaming\nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    


```python
def clean(text):
    text= str(text).lower()
    text= re.sub('\[.*?\]','',text)
    text= re.sub('https?://\S+|www\.\S+','',text)
    text= re.sub('<.*?>+','',text)
    text= re.sub('[%s]'% re.escape(string.punctuation),'',text)
    text= re.sub('\n','',text)
    text= re.sub('\w*\d\w*','',text)
    text = [word for word in text.split(' ') if word not in stopword]
    text = " ".join(text)
    return text

data["Title"]=data["Title"].apply(clean)

```


```python
data.Title.sample(10)
```




    5010     long dumb road
    5536             tunnel
    474               axone
    951         cheer squad
    3271    country new age
    4560        taxi ballad
    1155           cut bank
    3123         mismatched
    4411         state play
    5235       surgeons cut
    Name: Title, dtype: object




```python
data.Genres
```




    0                                           Reality TV
    1       Horror Movies, International Movies, Thrillers
    2                  Documentaries, International Movies
    3                                          TV Comedies
    4                  Documentaries, International Movies
                                 ...                      
    5962            Comedies, Dramas, International Movies
    5963                 International TV Shows, TV Dramas
    5964                 International TV Shows, TV Dramas
    5965           Dramas, International Movies, Thrillers
    5966                          Children & Family Movies
    Name: Genres, Length: 5967, dtype: object




```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
```


```python
feature = data['Genres'].tolist()

tfidf = TfidfVectorizer(stop_words="english")

tfidf_matrix = tfidf.fit_transform(feature)

similarity = cosine_similarity(tfidf_matrix)

indices = pd.Series(data.index,index=data['Title']).drop_duplicates()
```


```python
def netFlix_recommendation(title,similarity = similarity):
    try:
        index=indices[title]
        similarity_scores = list(enumerate(similarity[index]))
        similarity_scores = sorted(similarity_scores,key=lambda x: x[1],reverse=True)
        similarity_scores = similarity_scores[0:10]
        movieindices = [i[0] for i in similarity_scores]
        return data['Title'].iloc[movieindices]
    except:
        print(title,"not in database")
```


```python
indices
```




    Title
    unwell                            0
    alive                             1
    annefrank  parallel stories       2
    blackaf                           3
    catsthemewvie                     4
                                   ... 
    الف مبروك                      5962
    دفعة القاهرة                   5963
    海的儿子                           5964
    반드시 잡는다                        5965
    최강전사 미니특공대  영웅의 탄생             5966
    Length: 5967, dtype: int64




```python
netFlix_recommendation("okay")
```




    2173                               okay
    3321    mystery science theater  return
    942                             charmed
    1872                         good witch
    5026                          magicians
    2458                                guo
    3522        upon time lingjian mountain
    4957                     land hypocrisy
    48                                     
    679                          biohackers
    Name: Title, dtype: object




```python

```
