
from tweepy import Stream
from tweepy import OAuthHandler
#from tweepy.streaming import StreamListener
import tweepy
import numpy as np
import re
import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn import preprocessing



#menghitung jumlah karakter 
def hitung_karakter(kalimat):    
    s=set(kalimat.lower())
    return len(s)

#menghitung jumlah kata
def hitung_kata(kalimat):    
    s=kalimat.lower().split()
    return len(s)

import re

def konversi3huruf(s):
    # pattern to look for three or more repetitions of any character, including
    # newlines.
    pattern = re.compile(r"(.)\1{2,}", re.DOTALL) 
    return pattern.sub(r"\1\1", s)

def hitung_kalimat(kalimat):
    s = konversi3huruf(kalimat).replace('..','.').count('.')
    if s == 0:
        s=1
    return s

def hitung_quotes(kalimat):
    s = konversi3huruf(kalimat).replace("''","'").count("'")
    t = konversi3huruf(kalimat).replace('""','"').count('"')
    return int((s+t)/2)


# In[6]:


## API TWITTER
consumer_key = "NUigVZ9PCy5PDjeLRXqWwb29Y"
consumer_secret = "x4hXJCmucIOXJLoPf4HXtqlrpfUrZNVpl7D6coaqKrVK0DwFGI"
access_token = "229820253-9iXCsJfbKt3QZsPOIPBCEqJs8uG9RIz3IkluQlRZ"
access_token_secret = "sYSHL2TPopobPTaQUTKjYLQ2rrN5raxWRGH7VDBga4zIN"
            
OAUTH_KEYS = {'consumer_key':consumer_key, 'consumer_secret':consumer_secret,
              'access_token_key':access_token, 'access_token_secret':access_token_secret}
auth = tweepy.OAuthHandler(OAUTH_KEYS['consumer_key'], OAUTH_KEYS['consumer_secret'])
api = tweepy.API(auth)



from flask import Flask,url_for,request,render_template,jsonify,send_file,url_for
from flask_bootstrap import Bootstrap
import json
import time
import os

image_folder = os.path.join('static', 'images')


# In[9]:


import pickle
import os
vectorize=pickle.load(open("model/vectorizer.pkl", "rb"))
modelIE=pickle.load(open("model/KNN9_IE.pkl", "rb"))
modelNS=pickle.load(open("model/KNN9_NS.pkl", "rb"))
modelTF=pickle.load(open("model/KNN9_TF.pkl", "rb"))
modelJP=pickle.load(open("model/KNN9_JP.pkl", "rb"))




def proses_data(username):
    username = str(username)
    tweets_data = []
    for tweet in api.user_timeline(screen_name = username, count = 500,lang='id',include_rts = False):
        tweets_data.append([tweet.text])
    sample='Data yang diambil : '+str(len(tweets_data))
    data = pd.DataFrame(tweets_data,columns=['text'])
    data['username']=username
    df=data[['username','text']]

    data['text']=data['text'].apply(lambda x: x.lower()) 
    data['text']=data['text'].apply(lambda x: konversi3huruf(x)) 
    data['text']=data['text'].apply(lambda x: re.sub('@[^\s]+','',x)) 
    data['text']=data['text'].apply(lambda x: re.split('https:\/\/.*', str(x))[0]) 
    data['text']=data['text'].apply(lambda x: re.sub(r"\d+", "", x)) 
    data['text']=data['text'].apply(lambda x: x.translate(str.maketrans("","",string.punctuation)))
    data['text']=data['text'].apply(lambda x: re.sub(r'\w*com\w*', '', x))
    data['text']=data['text'].apply(lambda x: re.sub(r'[^a-zA-Z ]', '', x))
    #data1=data.groupby(['username'],as_index=False).agg({'text':lambda x: "{%s}" % ' '.join(x.astype(str))})

    test=vectorize.transform(data['text'].values.astype('U'))

    X_vec = test.toarray()
    bobot_kata_tfidf=pd.DataFrame(X_vec, columns=vectorize.get_feature_names())



    hasil_IE=modelIE.predict(bobot_kata_tfidf).tolist()
    j_I=round(hasil_IE.count('I')/len(hasil_IE)*100,1)
    j_E=round(hasil_IE.count('E')/len(hasil_IE)*100,1)

    kata_IE='Intorvert '+str(j_I)+'% VS '+'Ekstrovert '+str(j_E)+'% '

    if j_I>=j_E:
        KepribadianIE='I'
    else:
        KepribadianIE='E'


    hasil_NS=modelNS.predict(bobot_kata_tfidf).tolist()
    j_N=round(hasil_NS.count('N')/len(hasil_NS)*100,1)
    j_S=round(hasil_NS.count('S')/len(hasil_NS)*100,1)

    kata_NS='Intuition '+str(j_N)+'% VS '+'Sensing '+str(j_S)+'% '

    if j_N>=j_S:
        KepribadianNS='N'
    else:
        KepribadianNS='S'

    hasil_TF=modelTF.predict(bobot_kata_tfidf).tolist()
    j_T=round(hasil_TF.count('T')/len(hasil_TF)*100,1)
    j_F=round(hasil_TF.count('F')/len(hasil_TF)*100,1)

    kata_TF='Thinking '+str(j_T)+'% VS '+'Felling '+str(j_F)+'% '

    if j_T>=j_F:
        KepribadianTF='T'
    else:
        KepribadianTF='F'


    hasil_JP=modelJP.predict(bobot_kata_tfidf).tolist()
    j_J=round(hasil_JP.count('J')/len(hasil_JP)*100,1)
    j_P=round(hasil_JP.count('P')/len(hasil_JP)*100,1)

    kata_JP='Judging '+str(j_J)+'% VS '+'Perceiving '+str(j_P)+'% '

    if j_J>=j_P:
        KepribadianJP='J'
    else:
        KepribadianJP='P'
        
        
    df['IE']=hasil_IE
    df['NS']=hasil_NS
    df['TF']=hasil_TF
    df['JP']=hasil_JP
    df['Tipe']=df['IE']+df['NS']+df['TF']+df['JP']

    kepribadian=KepribadianIE+KepribadianNS+KepribadianTF+KepribadianJP
    return sample,kepribadian,[kata_IE,kata_NS,kata_TF,kata_JP],df[['text','Tipe']]




# Initialize App
app = Flask(__name__)
app.config['image_folder'] = image_folder
Bootstrap(app)


@app.route('/')
def index():
    imagehome = os.path.join(app.config['image_folder'], 'mbti.png')
    kosong=os.path.join(app.config['image_folder'], 'awal.jpg')
    return render_template('index.html',imagehome=imagehome,imagehasil=kosong,pic=kosong)


# In[14]:


def hasilkepribadian(kepribadian):
    if kepribadian=='INTJ':
        imagehasil=os.path.join(app.config['image_folder'], 'intj.svg')
        katahasil='Pemikir yang imajinatif dan strategis, dengan rencana untuk segala sesuatunya.'
    elif kepribadian=='INTP':
        imagehasil=os.path.join(app.config['image_folder'], 'intp.svg')
        katahasil='Penemu yang inovatif dengan kedahagaan akan pengetahuan yang tidak ada habisnya.'
    elif kepribadian=='ENTJ':
        imagehasil=os.path.join(app.config['image_folder'], 'entj.svg')
        katahasil='Pemimpin yang pemberani, imaginatif dan berkemauan kuat, selalu menemukan cara - atau menciptakan cara.'
    elif kepribadian=='ENTP':
        imagehasil=os.path.join(app.config['image_folder'], 'entp.svg')
        katahasil='Pemikir yang cerdas dan serius yang gatal terhadap tantangan intelektual.'
    
    elif kepribadian=='INFJ':
        imagehasil=os.path.join(app.config['image_folder'], 'infj.svg')
        katahasil='Pendiam dan mistis, tetapi idealis yang sangat menginspirasi dan tak kenal lelah.'
    elif kepribadian=='INFP':
        imagehasil=os.path.join(app.config['image_folder'], 'infp.svg')
        katahasil='Orang yang puitis, baik hati dan altruisik, selalu ingin membantu aksi kebaikan.'
    elif kepribadian=='ENFJ':
        imagehasil=os.path.join(app.config['image_folder'], 'enfj.svg')
        katahasil='Pemimpin yang karismatik dan menginspirasi, mampu memukai pendengarnya.'
    elif kepribadian=='ENFP':
        imagehasil=os.path.join(app.config['image_folder'], 'enfp.svg')
        katahasil='Semangat yang antusias, kreatif dan bebas bergaul, yang selalu dapat menemukan alasan untuk tersenyum.'
    
    elif kepribadian=='ISTJ':
        imagehasil=os.path.join(app.config['image_folder'], 'istj.svg')
        katahasil='Individu yang praktis dan mengutamakan fakta, yang keandalannya tidak dapat diragukan.'
    elif kepribadian=='ISFJ':
        imagehasil=os.path.join(app.config['image_folder'], 'isfj.svg')
        katahasil='Pelindung yang sangat berdedikasi dan hangat, selalu siap membela orang yang dicintainya.'
    elif kepribadian=='ESTJ':
        imagehasil=os.path.join(app.config['image_folder'], 'entj.svg')
        katahasil='Administrator istimewa, tidak ada duanya dalam mengelola sesuatu - atau orang.'
    elif kepribadian=='ESFJ':
        imagehasil=os.path.join(app.config['image_folder'], 'esfj.svg')
        katahasil='Orang yang sangat peduli, sosial dan populer, selalu ingin membantu.'
        
    elif kepribadian=='ISTP':
        imagehasil=os.path.join(app.config['image_folder'], 'istp.svg')
        katahasil='Eksperimenter yang pemberani dan praktis, menguasai semua jenis alat.'
    elif kepribadian=='ISFP':
        imagehasil=os.path.join(app.config['image_folder'], 'isfp.svg')
        katahasil='Seniman yang fleksibel dan mengagumkan, selalu siap menjelajahi dan mengalami hal baru.'
    elif kepribadian=='ESTP':
        imagehasil=os.path.join(app.config['image_folder'], 'estp.svg')
        katahasil='Orang yang cerdas, bersemangan dan sangat tanggap, yang benar-benar menikmati hidup yang menantang.'
    elif kepribadian=='ESFP':
        imagehasil=os.path.join(app.config['image_folder'], 'esfj.svg')
        katahasil='Orang yang spontan, bersemangan dan antusias - hidup tidak akan membosankan saat berdekatan dengan mereka.'
    else:
        imagehasil=''
        katahasil=''
    return imagehasil,katahasil


# In[ ]:





# In[ ]:





# In[15]:



@app.route('/analyze',methods=['GET','POST'])
def analyze():
    if request.method == 'POST':
        username = str(request.form['username'])
        
        sample,kepribadian,hasil,df=proses_data(username)
        
        #test = api.lookup_users(screen_names=[username])
        #for user in test:
        #    nama=user.name
        #    description=user.description
        #    pic=user.profile_image
        
        nama=username
        description=hasil
        pic='https://p.kindpng.com/picc/s/52-524250_twitter-black-twitter-logo-svg-hd-png-download.png'
        
        imagehasil,katahasil=hasilkepribadian(str(kepribadian))
        
        imagehome = os.path.join(app.config['image_folder'], 'hasilmbti.png')
    return render_template('index.html',username=username,nama=nama,description='"'+str(description)+'"',hasilkepribadian='Kepribadian MBTI kamu adalah '+kepribadian,
                           sample=sample,
                           sambutannama='Hai, "'+str(nama)+'"',katahasil=katahasil,
                          pic=pic,imagehome=imagehome,imagehasil=imagehasil,
                           tables=[df.to_html(classes='data')], titles=df.columns.values)


if __name__ == '__main__':
    app.run(debug=True,use_reloader=False)
    #app.run(host='0.0.0.0',port=8080)

