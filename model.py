#movie recommendation

import numpy as np
import pandas as pd

pd.read_csv('/content/tmdb_5000_credits.csv.zip').head()

credits=pd.read_csv('/content/tmdb_5000_credits.csv.zip')
movies=pd.read_csv('/content/tmdb_5000_movies.csv.zip')


movies.sample(5)



movies=movies.merge(credits,on='title')

movies.head(2)

#important columns we will keep from the movies dataset
#genres
#id
#keywords
#title
#overview
#cast
#crew
movies=movies[['movie_id','title','overview','genres','keywords','cast','crew']]

import ast

def convert1(obj):
    l=[]
    for i in ast.literal_eval(obj):
        l.append(i['name'])
    return l

movies['genres']=movies['genres'].apply(convert1)

movies['genres'][0]

movies['keywords']=movies['keywords'].apply(convert1)

def convert2(obj):
    l=[]
    counter=0
    for i in ast.literal_eval(obj):
        if counter!=3:
           l.append(i['name'])
           counter+=1
        else:
            break
    return l

movies['cast']=movies['cast'].apply(convert2)

def convert3(obj):
    l=[]

    for i in ast.literal_eval(obj):
        if i['job']=='Director':
           l.append(i['name'])
           break
    return l

movies['crew']=movies['crew'].apply(convert3)

movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])

movies['genres']=movies['genres'].apply(lambda x:" ".join(x))
movies['keywords']=movies['keywords'].apply(lambda x:" ".join(x))
movies['cast']=movies['cast'].apply(lambda x:" ".join(x))
movies['crew']=movies['crew'].apply(lambda x:" ".join(x))

movies['tag']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']

newdf=movies[['movie_id','title','tag']]

# Handle potential float values in the 'tag' column
newdf['tag'] = newdf['tag'].apply(lambda x: x.lower() if isinstance(x, str) else x)

newdf.head(2)

# Handle missing values in the 'tag' column before vectorization
newdf['tag'] = newdf['tag'].fillna('')  # Replace NaN with empty strings

# Now proceed with CountVectorizer

import nltk

from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

def stem(text):
  y=[]
  for i in text.split():
    y.append(ps.stem(i))
  return " ".join(y)

newdf['tag']=newdf['tag'].apply(stem)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000,stop_words='english')

vectors=cv.fit_transform(newdf['tag']).toarray()

from sklearn.metrics.pairwise import cosine_similarity

similarity=cosine_similarity(vectors)

def recommend(movie):
  movie_index=newdf[newdf['title']==movie].index[0]
  distances=similarity[movie_index]
  movies_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
  for i in movies_list:
    print(newdf.iloc[i[0]].title)

recommend('Avatar')

import pickle
pickle.dump(newdf.to_dict(),open('movieseee.pkl','wb'))

pickle.dump(similarity,open('similarity.pkl','wb'))
