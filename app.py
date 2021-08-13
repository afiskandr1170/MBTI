from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import tweepy
import numpy as np
import re
import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn import preprocessing


# In[4]:


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





import pickle
import os
vectorize=pickle.load(open("model/TFIDF.pkl", "rb"))
modelIE=pickle.load(open("model/KNN9_IE.pkl", "rb"))
modelNS=pickle.load(open("model/KNN7_NS.pkl", "rb"))
modelTF=pickle.load(open("model/KNN9_TF.pkl", "rb"))
modelJP=pickle.load(open("model/KNN9_JP.pkl", "rb"))


# In[8]:


def proses_data(username):
    username = str(username)
    tweets_data = []
    for tweet in api.user_timeline(screen_name = username, count = 500,lang='id',include_rts = True):
        tweets_data.append([tweet.text])
    sample='Data yang diambil : '+str(len(tweets_data))
    data = pd.DataFrame(tweets_data,columns=['text'])
    data['username']=username
    df=data[['username','text']]

    data['txt_cunq']= data['text'].apply(lambda x: hitung_karakter(str(x)))
    data['txt_word']= data['text'].apply(lambda x: hitung_kata(str(x)))
    data['txt_sentences']= data['text'].apply(lambda x: hitung_kalimat(str(x)))
    data['txt_url'] = data['text'].apply(lambda x: x.count('http'))
    data['txt_media'] = data['text'].apply(lambda x: x.count('pic.twitter.com'))
    data['txt_question'] = data['text'].apply(lambda x: x.count('?'))
    data['txt_imperative'] = data['text'].apply(lambda x: x.count('!'))
    data['txt_hastags'] = data['text'].apply(lambda x: x.count('#'))
    data['txt_retweet'] = data['text'].apply(lambda x: x.count(' RT '))
    data['txt_mention'] = data['text'].apply(lambda x: x.count('@'))
    data['txt_quotes'] = data['text'].apply(lambda x: hitung_quotes(x))
    data['tweet_liked']=0
    data['tweet_replied']=0
    data['tweet_retweeted']=0

    f_e=['tweet_liked', 'tweet_replied', 'tweet_retweeted', 
         'txt_cunq', 'txt_word', 'txt_sentences', 'txt_url', 'txt_media',
         'txt_question', 'txt_imperative', 'txt_hastags', 'txt_retweet', 'txt_mention',
         'txt_quotes', 
         'usr_follower', 'usr_following', 
         'usr_liked', 'usr_media', 'usr_cunq_bio', 'usr_word_bio']

    test = api.lookup_users(screen_names=[username])
    for user in test:
        data['usr_follower']=user.followers_count
        data['usr_following']=user.friends_count
        data['usr_liked']=user.favourites_count
        data['usr_media']=0
        data['bio']=user.description
        data['usr_cunq_bio']=data['bio'].apply(lambda x: hitung_karakter(str(x)))
        data['usr_word_bio']=data['bio'].apply(lambda x: hitung_kata(str(x)))

    fe=data[f_e].astype(float)
    x = fe.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    fe_norm = pd.DataFrame(x_scaled)
    fe_norm.columns=fe.columns
    fe_norm=fe_norm.mean().reset_index().T
    fe_norm.columns=fe.columns
    fe_norm=fe_norm.drop('index')

    data['text']=data['text'].apply(lambda x: x.lower()) 
    data['text']=data['text'].apply(lambda x: konversi3huruf(x)) 
    data['text']=data['text'].apply(lambda x: re.sub('@[^\s]+','',x)) 
    data['text']=data['text'].apply(lambda x: re.split('https:\/\/.*', str(x))[0]) 
    data['text']=data['text'].apply(lambda x: re.sub(r"\d+", "", x)) 
    data['text']=data['text'].apply(lambda x: x.translate(str.maketrans("","",string.punctuation)))
    data['text']=data['text'].apply(lambda x: re.sub(r'\w*com\w*', '', x))
    data['text']=data['text'].apply(lambda x: re.sub(r'[^a-zA-Z ]', '', x))
    data1=data.groupby(['username'],as_index=False).agg({'text':lambda x: "{%s}" % ' '.join(x.astype(str))})

    test=vectorize.transform(data1['text'].values.astype('U'))

    X_vec = test.toarray()
    bobot_kata_tfidf=pd.DataFrame(X_vec, columns=vectorize.get_feature_names())
    bobot_kata_tfidf_f_e=pd.merge(bobot_kata_tfidf,fe_norm, left_index=True, right_index=True)
    bobot_kata_tfidf=bobot_kata_tfidf_f_e

    kolom_IE=['adek','adies','ah','ai','aing','akun','album','ambo','ambyar','ampun','anggar','anjay','anjing','anjir','apas','army','armyluvbts','aron','arsul','at','awak','babi','bagus','bandung','banget','banten','bapak','bbmastopsocial','begitu','belum','benda','benny','berkah','best','bgst','bila','bodoh','bosan','boywithluv','bpk','bpkh','bukan','bumi','cakep','calon','capek','carat','caratdeul','carats','caratss','cinta','cont','curhat','czu','dana','day','deh','dek','demi','dewas','dewasa','dinda','dingin','dino','dirut','djsn','dkt','doain','doi','dom','drop','duh','duk','ehehehe','email','enggak','eri','esok','exo','fb','follback','gabole','gaenak','galau','gi','gila','goyang','gua','gue','hahah','hahaha','hahahah','hahahaha','haji','hamba','hara','hati','have','hee','hehe','hehehe','heran','hhe','hidup','hoteldelluna','ia','ikon','indah','indonesia','investasi','ivotebtsbbmas','jabar','jangan','jateng','jatim','jatuh','jay','je','jimin','jodoh','johnny','jr','junggaram','kae','kait','kakak','kali','kalian','kamera','kami','kapolri','karasuma','kau','kawan','kaya','kelas','kelola','kenal','ketemu','kim','kimsohyun','kkn','kok','kom','kpop','ktm','kucing','kur','lo','loli','lovealarm','lu','lupa','macam','mak','maneh','manusia','masa','masak','massi','masyarakat','mcp','melon','memang','member','menko','mereka','mgmavote','mingyu','miz','mpn','mtvlafandombtsarmy','mutualan','my','nak','nder','nggi','nowisjayparkday','ntt','nuest','nya','ode','off','ojol','orang','others','otw','oty','package','pastu','pcas','perumnas','pic','pink','pls','pn','polri','pt','puasa','ra','random','ren','rep','ria','ribu','rt','rumah','sagu','sakit','satu','sebab','section','sedang','selamat','semangat','sender','senpai','sensei','setan','si','sia','sistem','songkang','soompiawards','sub','sumut','svt','syahroni','tag','tak','takde','tanah','tanda','tante','teambts','teamexo','teman','temu','tengok','tep','themusicvideo','toto','tuhan','twitterbestfandom','ubah','ume','uni','utara','vote','wae','weh','wes','with','wkw','wkwk','wkwkwkw','xd','yaallah','yampun','yang','yoongi','yuk','yunseong','tweet_liked','txt_media','txt_hastags','txt_retweet','txt_mention','usr_follower','usr_following','usr_media','usr_cunq_bio','usr_word_bio']
    kolom_NS=['abang','adek','ai','aiman','alhamdulillah','ambo','angin','anjir','army','aron','asih','asrama','astaga','at','au','awak','bagai','bahan','balik','banget','beda','begini','besok','bikin','bila','bola','boywithluv','bpk','bpkh','bts','cakap','calon','carat','caratdeul','carats','catat','cha','dan','dapat','dari','day','deh','dek','depok','dinda','dirut','djsn','dkt','dom','drop','duk','efek','eh','ehe','email','eri','esok','galau','gi','grab','grup','gua','gue','haha','hahah','hahaha','han','have','hehehe','heran','hmm','hoteldelluna','huhuhu','ia','id','ikon','ila','investasi','istirahat','ivotebtsbbmas','jabar','jadi','jangan','jatim','jay','je','jika','jr','kadang','kae','kah','kak','kakak','kalong','kami','kapolri','karasuma','kau','kawan','ke','kelas','keluar','kom','ktm','ku','lah','loh','lucu','lupa','macam','main','malang','massi','masyarakat','melon','member','menang','mereka','mgmavote','mingyu','miz','mom','mtvlafandombtsarmy','mutualan','nad','nak','nggi','nice','nowisjayparkday','ntt','nuest','nya','nyata','of','off','package','pagi','pak','pastu','pcas','pic','plot','pn','polri','pt','pun','ramai','raya','rt','salken','sarap','sayang','section','selamat','sembuh','senpai','sensei','si','siang','sih','sikit','soompiawards','suka','sumut','svt','tadi','taehyung','tahun','tak','takde','takut','tante','teamexo','tengok','themusicvideo','tiati','tidur','turut','twitterbestfandom','ume','uni','vote','with','wkkw','wkw','wkwkw','wkwkwk','xd','yang','ye','yuk','txt_sentences','txt_media','txt_hastags','txt_retweet','usr_follower','usr_following','usr_media','usr_word_bio']
    kolom_TF=['abang','adik','ai','aku','album','allah','amin','ampun','anak','and','angin','apasih','armyluvbts','aron','asrama','at','atuh','avaku','awal','bakal','balik','baru','beli','benny','bila','blue','boleh','bong','boywithluv','bpk','bpkh','brian','by','calon','carat','caratdeul','carats','caratss','cc','cinta','cont','cuman','czu','dana','dari','datang','day','dek','depok','dewas','dinda','dirut','djsn','dk','dkt','drop','duk','eh','elsa','email','entar','esok','exo','fear','gaes','gais','garagara','gc','ghiegi','gi','green','grup','gua','guys','habis','hahah','hahaha','haji','halu','han','hara','hari','hehe','hehehe','heran','hotel','hoteldelluna','http','huhu','hukum','ia','ic','id','idol','ikon','innalillahi','investasi','ivotebtsbbmas','jabar','jakarta','jatim','jay','je','jeonghan','jihoon','jimin','jr','juga','just','kait','kakak','kalong','kami','kamu','kan','kapolri','karasuma','kau','ke','kemudian','ketua','kia','kini','kira','kom','kos','ktm','ku','kur','lalu','lo','loli','lu','macam','mah','makan','malas','mas','massi','masyarakat','mau','mcp','mecima','member','menko','mgmavote','minggu','mingyu','minum','miz','mom','mpn','mtvlafandombtsarmy','mutualan','nad','nak','nct','nder','nggi','novel','nowisjayparkday','ntt','nuest','ode','off','ot','others','oty','pacar','package','pak','papa','pasang','pastu','pcas','persona','perumnas','pic','pindah','pink','plot','pn','polri','premiosmtvmiaw','qn','ra','relate','rep','reply','room','saja','salah','saya','section','segar','senpai','sensei','seokmin','si','sih','soompiawards','stage','streaming','streamingnya','studio','sub','suka','sumut','susu','svt','taehyun','tak','takde','tanpa','tanya','teamexo','telah','tengok','tep','themusicvideo','tiba','ticketing','tiket','toto','tt','twitterbestfandom','twt','ucap','uji','ume','uni','utara','uu','venue','ver','versi','vid','with','wkw','wkwk','wkwkkw','wkwkwk','ya','yaampun','yang','yaudah','yuk','tweet_replied','txt_sentences','txt_url','txt_imperative','txt_hastags','txt_retweet','txt_mention','usr_follower','usr_following','usr_media','usr_word_bio']
    kolom_JP=['abang','ada','adek','adies','agar','ai','aku','ambo','and','anggar','apa','army','armyluvbts','aron','arsul','asa','asik','astaga','at','bagaimana','bagus','balik','banget','banten','belum','benda','benny','bisa','blue','bonus','boywithluv','bpk','bpkh','buat','bye','calon','carat','caratdeul','carats','caratss','cont','dana','day','debut','dek','depok','dewas','dinda','diri','dirut','djsn','dkt','drop','duk','ekonomi','elsa','email','eri','esok','exo','gara','ghiegi','gi','gua','guru','guys','haha','hahah','hahaha','hai','haji','han','hara','hari','have','hehehe','hm','hoteldelluna','http','hukum','ikon','investasi','ivotebtsbbmas','jabar','jago','jangan','jateng','jatim','jay','je','jimin','junggaram','kah','kakak','kalau','kalo','kami','kamu','kantor','kapolri','karasuma','kau','kelola','kenapa','kepada','kia','kimsohyun','kom','ktm','kur','lama','lho','loli','lovealarm','lumayan','macam','makan','malam','maneh','massi','masyarakat','mcp','melon','memang','menang','menko','meski','mirip','miz','mom','mpn','mtvlafandombtsarmy','mtvlakpopbts','mutual','nad','nak','nanti','nder','nggi','not','nowisjayparkday','ns','ntt','nuest','nya','oleh','on','others','oty','package','pak','pastu','pcas','perumnas','pic','pn','pokok','polri','premiosmtvmiaw','ra','rakyat','rasa','realisasi','rt','sagu','salah','saya','sdr','section','selamat','semua','senpai','sensei','seungyoun','si','siang','siapa','sih','sing','skincare','skinker','sms','soal','songkang','soompiawards','stasiun','studio','sumut','svt','syahroni','tak','takde','taman','tante','teambts','teamexo','teamnuest','tengok','tep','themusicvideo','tiati','tidak','tidur','tolong','toto','tv','twitterbestfandom','twt','ume','uni','untuk','utara','uu','vote','with','wkw','wkwk','wkwkw','xd','yampun','yang','ye','tweet_liked','tweet_replied','txt_url','txt_media','txt_retweet','usr_follower','usr_following','usr_media','usr_cunq_bio','usr_word_bio']

    bobot_kata_tfidf_IE=bobot_kata_tfidf[kolom_IE]
    bobot_kata_tfidf_NS=bobot_kata_tfidf[kolom_NS]
    bobot_kata_tfidf_TF=bobot_kata_tfidf[kolom_TF]
    bobot_kata_tfidf_JP=bobot_kata_tfidf[kolom_JP]

    if modelIE.predict(bobot_kata_tfidf_IE)[0]==0:
        KepribadianIE='I'
    else:
        KepribadianIE='E'

    if modelNS.predict(bobot_kata_tfidf_NS)[0]==0:
        KepribadianNS='N'
    else:
        KepribadianNS='S'

    if modelTF.predict(bobot_kata_tfidf_TF)[0]==0:
        KepribadianTF='T'
    else:
        KepribadianTF='F'
    if modelJP.predict(bobot_kata_tfidf_JP)[0]==0:
        KepribadianJP='J'
    else:
        KepribadianJP='P'

    kepribadian=KepribadianIE+KepribadianNS+KepribadianTF+KepribadianJP
    return sample,kepribadian,df


# In[ ]:





# In[ ]:





# In[9]:


# Initialize App
app = Flask(__name__)
app.config['image_folder'] = image_folder
Bootstrap(app)


@app.route('/')
def index():
    imagehome = os.path.join(app.config['image_folder'], 'mbti.png')
    kosong=os.path.join(app.config['image_folder'], 'awal.jpg')
    return render_template('index.html',imagehome=imagehome,imagehasil=kosong,pic=kosong)


# In[10]:


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




@app.route('/analyze',methods=['GET','POST'])
def analyze():
    if request.method == 'POST':
        username = str(request.form['username'])
        
        sample,kepribadian,df=proses_data(username)
        
        test = api.lookup_users(screen_names=[username])
        for user in test:
            nama=user.name
            description=user.description
            pic=user.profile_image_url
        
        imagehasil,katahasil=hasilkepribadian(str(kepribadian))
        
        imagehome = os.path.join(app.config['image_folder'], 'hasilmbti.png')
    return render_template('index.html',username=username,nama=nama,description='"'+str(description)+'"',hasilkepribadian='Kepribadian MBTI kamu adalah '+kepribadian,
                           sample=sample,
                           sambutannama='Hai, "'+str(nama)+'"',katahasil=katahasil,
                          pic=pic,imagehome=imagehome,imagehasil=imagehasil,
                           tables=[df.to_html(classes='data')], titles=df.columns.values)



if __name__ == '__main__':
    app.run()
    #app.run(debug=True,use_reloader=False)
    #app.run(host='0.0.0.0',port=8080)
