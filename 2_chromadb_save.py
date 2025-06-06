import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from collections import Counter
import pandas as pd
import math
import json
import pymysql
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from kiwipiepy import Kiwi
import re
from datetime import datetime
import cloudpickle

# 고유명사 추가
kiwi_app = Kiwi()
nnp_list = ['group.txt','line.txt','name.txt','NNP.txt','product.txt']
for n in nnp_list :
    with open(f'./NNP_List/{n}','r',encoding='utf-8') as file:
        nnp = file.read()
    for n in nnp.split('\n') :
        kiwi_app.add_user_word(n, 'NNP', 6)

# ChromaDB 사용자 임베딩 함수 정의
class textEmbeddingFunction(EmbeddingFunction):
    def __call__ (self, input:Documents)->Embeddings:
        vector = vectorizer.transform(input)
        dense = vector.toarray()
        embeddings = text_pca.transform(dense) 
        return embeddings 
    
class nameEmbeddingFunction(EmbeddingFunction):
    def __call__ (self, input:Documents)->Embeddings:
        total_words = len(input)
        token_count = Counter(input)
        tf_dict = {word: count/total_words for word, count in token_count.items()}
        vector = [tf_dict.get(word,0)*name_idf_dict.get(word,0) for word in name_vocabulary_dict]
        dense = np.array([vector])
        embeddings = name_pca.transform(dense) 
        return embeddings 
    

def read_to_mariadb(sql):
    conn = None
    cur = None
    conn = pymysql.connect(host='50.201.224.41',port=3306,user='wi',password='5882',db='filesearch',charset='utf8')
    cur = conn.cursor()
    cur.execute("SET SESSION sql_mode = 'NO_BACKSLASH_ESCAPES'")
    cur.execute(sql)
    data = cur.fetchall()
    conn.commit()
    conn.close()
    return data

# 텍스트 전처리, 동의어 처리  
def synonym_preprocessing(text) :
    text = text.lower()
    text = text.replace("'"," ")
    text = re.sub(r'\s+',' ',text)
    text = re.sub(r'\n+','\n',text)
    text = re.sub(r'\t+',' ',text)
    text = re.sub(r'([ㄱ-ㅎㅏ-ㅣ])\1{1,}','',text)
    with open('./NNP_List/synonym.txt','r',encoding='utf-8') as file:
        synonym = file.read()
    for s in synonym.split('\n') :
        s = s.split('=')
        text1 = s[0]
        text2 = s[1].split('$')
        for t in text2 :
            if t in text :
                text = text.replace(t, text1+' ')
    return text


def tfidf_name_vector(base_data):
    vocabulary_list = []
    tfidf_list = []
    token_data = base_data['name'].tolist()
    word_doc_count = {}
    # idf 문서별 단어 빈도수 정리
    for t in token_data:
        token_list = []
        text = synonym_preprocessing(t)
        noun_tags = ['NNG','NNP','NNB','NR','NP','SL','SN']
        tokens = kiwi_app.tokenize(text)
        for token in tokens :
            if token.tag in noun_tags :
                token_list.append(token.form)
                vocabulary_list.append(token.form)
        tfidf_list.append(token_list)
        words = set(token_list)   
        for word in words :
            if word in word_doc_count :
                word_doc_count[word] += 1
            else :
                word_doc_count[word] = 1
    # 단어별 idf값 계산 및 단어사전
        N = len(tfidf_list)   
        idf_dict = {word : math.log(N/(count+1)) for word, count in word_doc_count.items()}   
        vocabulary = set(vocabulary_list)
        vocabulary_dict = {word: count+1 for count, word in enumerate(vocabulary)}
    # TF-IDF 벡터 만들기
    vector_list = []
    for tfidf in tfidf_list:
        total_words = len(tfidf)
        token_count = Counter(tfidf)
        tf_dict = {word: count/total_words for word, count in token_count.items()}
        vector = [tf_dict.get(word,0)*idf_dict.get(word,0) for word in vocabulary_dict]
        vector_list.append(vector)
    return idf_dict, vocabulary_dict, vector_list



if __name__ == '__main__' :
    print(datetime.now())
    # 설정데이터 가져오기
    with open('config.json','r',encoding='utf-8') as file :
        config = json.load(file)
    chromatextpath = config['chromatextpath']
    chromanamepath = config['chromanamepath']
    collectionname = config['collectionname']
    mariaDBtable = config['mariaDBtable']


    print('MariaDB 데이터 가져오기 시작')
    sql = f'''select * from {mariaDBtable}'''
    data = read_to_mariadb(sql)
    columns = ['path','name','savetime','text','token']
    base_data = pd.DataFrame(data, columns=columns) 
    print('MariaDB 데이터 가져오기 완료')


    print("TF-IDF 벡터 만들기 시작")
    token_data = base_data['token'].tolist()
    vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split(','))
    text_vector_list = vectorizer.fit_transform(token_data)
    text_idf_dict = dict(zip(vectorizer.get_feature_names_out(), vectorizer.idf_))
    text_vocabulary_dict = vectorizer.vocabulary_
    
    with open(f'{collectionname}_text_idf.json','w',encoding='utf-8') as file :
        json.dump(text_idf_dict, file, ensure_ascii=False, indent=4)
    with open(f'{collectionname}_text_vocabulary.json','w',encoding='utf-8') as file :
        json.dump(text_vocabulary_dict, file, ensure_ascii=False, indent=4)
    with open(f'{collectionname}_text_vectorizer.pkl','wb') as file :
        cloudpickle.dump(vectorizer, file)

    name_data = base_data['name'].tolist()
    name_list = []
    for n in name_data :
        token_list = []
        name = synonym_preprocessing(n)
        name = kiwi_app.space(name)
        noun_tags = ['NNG','NNP','NNB','NR','NP','SL','SN']
        tokens = kiwi_app.tokenize(name)
        for token in tokens :
            if token.tag in noun_tags :
                token_list.append(token.form)
        name_list.append(",".join(token_list))
    name_vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split(','))
    name_vector_list = name_vectorizer.fit_transform(name_list)
    name_idf_dict = dict(zip(name_vectorizer.get_feature_names_out(), name_vectorizer.idf_))
    name_vocabulary_dict = name_vectorizer.vocabulary_
    

    # name_idf_dict, name_vocabulary_dict, name_vector_list = tfidf_name_vector(base_data)
    with open(f'{collectionname}_name_idf.json','w',encoding='utf-8') as file :
        json.dump(name_idf_dict, file, ensure_ascii=False, indent=4)
    with open(f'{collectionname}_name_vocabulary.json','w',encoding='utf-8') as file :
        json.dump(name_vocabulary_dict, file, ensure_ascii=False, indent=4)
    with open(f'{collectionname}_name_vectorizer.pkl','wb') as file :
        cloudpickle.dump(name_vectorizer, file)
    print("TF-IDF 벡터 만들기 완료")


    print("PCA 차원축소 시작")
    text_dense = text_vector_list.toarray()
    text_pca = PCA(n_components=4096)
    text_reduced_matrix = text_pca.fit_transform(text_dense)
    joblib.dump(text_pca, f'text_pca_{collectionname}.joblib')
    name_dense = name_vector_list.toarray()
    name_pca = PCA(n_components=4096)
    name_reduced_matrix = name_pca.fit_transform(name_dense)
    joblib.dump(name_pca, f'name_pca_{collectionname}.joblib')
    print("PCA 차원축소 완료")


    print("ChromaDB 저장 시작")
    # 메타데이터 정리
    path = base_data['path'].tolist()
    name = base_data['name'].tolist()
    savetime = base_data['savetime'].tolist()

    text_client = chromadb.PersistentClient(path=chromatextpath)
    text_collection = text_client.get_or_create_collection(name=collectionname,
                                                            embedding_function=textEmbeddingFunction(),
                                                            metadata={'hnsw:space':'cosine'})
    
    name_client = chromadb.PersistentClient(path=chromanamepath)
    name_collection = name_client.get_or_create_collection(name=collectionname,
                                                            embedding_function=nameEmbeddingFunction(),
                                                            metadata={'hnsw:space':'cosine'})
    count = 0
    for matrix in text_reduced_matrix :
        text_collection.upsert(ids=[path[count]],
                    embeddings=[matrix],
                    metadatas=[{'name':name[count],'savetime':savetime[count]}])
        count += 1
    count = 0
    for matrix in name_reduced_matrix :
        name_collection.upsert(ids=[path[count]],
                    embeddings=[matrix],
                    metadatas=[{'name':name[count],'savetime':savetime[count]}])
        count += 1
    print("ChromaDB 저장 완료")
    print(datetime.now())