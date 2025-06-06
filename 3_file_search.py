system_prompt = """
당신은 친절하고 유능한 요약 전문 AI 어시스턴트입니다.
당신의 주요 목표는 주어진 텍스트에 기반하여 **정보를 구조화된 형식으로, 부드럽고 이해하기 쉬운 문장으로 요약**하는 것입니다.

다음의 원칙을 따르세요 :
1. **질문에 대한 응답은 반드시 제공된 맥락(텍스트)에만 기반**해야 하며, 외부 지식이나 추측은 포함하지 않습니다.
2. **딱딱하거나 기계적인 어투를 피하고**, 자연스럽고 부드러운 문장으로 표현하세요.
3. 각 항목은 **짧은 문단으로 줄바꿈하여** 출력하고, 번호 뒤에 바로 문장이 붙지 않도록 하세요.
3. **적절한 제목을 내용의 맨 위에 추가하고, **[각 문단 사이에 한 줄 씩 공백]을 넣어 가독성을 높이세요.
4. 각 문단은 **3-4문장 이내**로 짧고 간결하게 작성하세요.
5. **가독성을 높이기 위해 각 항목 사이에 빈 줄**을 넣으세요.
6. 복잡한 정보는 **번호 목록, 요점 정리, 또는 항목화된 문단**을 활용해 시각적으로 정돈하세요.
7. 마지막에는 핵심 내용을 3~5문장으로 **간결하게 요약**하세요.
8. **제목과 내용에서 중요 핵심 단어를 <b>내용</b>로** 출력하여 강조하세요.
9. ["l", "PC", "PLC", "IP"]와 같은 요약어는 모두 대문자로 출력하도록 하세요.
10. [Clean Room]과 같은 영단어를 출력 시 첫 글자만 대문자로 출력하도록 하세요.
11. 날짜는 [년 월 일]의 형식에 맞춰 출력하도록 하세요. Ex 240101=24년 01년 01일 

출력 형식 예시 :
<b>제목</b> : ...
    1. <b>주요 내용 A</b> : ...\n
    2. <b>주요 내용 B</b> : ...\n
    3. <b>주요 내용 C</b> : ...\n

핵심 요약 :\n 
이 문서는 ...입니다. 핵심적으로 ...에 초점을 맞추며 ...을 설명합니다.
결과적으로 독자는 ...을 이해할 수 있습니다.

[중요] : 전체 응답은 맥락에서 제공된 정보에만 기반해야 합니다. 주어진 텍스트에 없는 외부 지식이나 가정은 포함하지 마세요. 답변은 항상 한국어로 답변해야 합니다.
"""

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from collections import Counter
from kiwipiepy import Kiwi
from fastapi import FastAPI
from pydantic import BaseModel
import json
import ollama
import re, math
import pymysql
import pandas as pd
import numpy as np
import joblib
import cloudpickle


# 고유명사 추가
kiwi_app = Kiwi()
nnp_list = ['group.txt','line.txt','name.txt','NNP.txt','product.txt']
for n in nnp_list :
    with open(f'./NNP_List/{n}','r',encoding='utf-8') as file:
        nnp = file.read()
    for n in nnp.split('\n') :
        kiwi_app.add_user_word(n, 'NNP', 6)

# 설정값 가져오기
with open('config.json','r',encoding='utf-8') as file :
    config = json.load(file)
chromatextpath = config['chromatextpath']
chromanamepath = config['chromanamepath']
collectionname = config['collectionname']
mariaDBtable = config['mariaDBtable']

with open(f'{collectionname}_text_vectorizer.pkl','rb') as file :
    text_vectorizer = cloudpickle.load(file)
with open(f'{collectionname}_name_vectorizer.pkl','rb') as file :
    name_vectorizer = cloudpickle.load(file)
with open(f'{collectionname}_name_idf.json','r',encoding='utf-8') as file :
    name_idf_dict = json.load(file)
with open(f'{collectionname}_name_vocabulary.json','r',encoding='utf-8') as file :
    name_vocabulary_dict = json.load(file)

text_pca = joblib.load(f'text_pca_{collectionname}.joblib')
name_pca = joblib.load(f'name_pca_{collectionname}.joblib')


# ChromaDB 사용자 임베딩 함수 정의
class textEmbeddingFunction(EmbeddingFunction):
    def __call__ (self, input:Documents)->Embeddings:
        vector = text_vectorizer.transform(input)
        dense = vector.toarray()
        embeddings = text_pca.transform(dense) 
        return embeddings 
    
class nameEmbeddingFunction(EmbeddingFunction):
    def __call__ (self, input:Documents)->Embeddings:
        # total_words = len(input)
        # token_count = Counter(input)
        # tf_dict = {word: count/total_words for word, count in token_count.items()}
        # vector = [tf_dict.get(word,0)*name_idf_dict.get(word,0) for word in name_vocabulary_dict]
        # dense = np.array([vector])
        # embeddings = name_pca.transform(dense) 
        vector = name_vectorizer.transform(input)
        dense = vector.toarray()
        embeddings = name_pca.transform(dense) 
        return embeddings 
    
# 동의어 처리  
def synonym_preprocessing(text) :
    text = text.lower()
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

def get_reg_text_data(question, n_results):
    client = chromadb.PersistentClient(path=chromatextpath)
    collection = client.get_or_create_collection(name=collectionname,
                                                embedding_function=textEmbeddingFunction(),
                                                metadata={'hnsw:space':'cosine'})
    token_question = synonym_preprocessing(question)
    noun_tags = ['NNG','NNP','NNB','NR','NP','SL']
    question_token_list = []
    tokens = kiwi_app.tokenize(token_question)
    for token in tokens :
        if token.tag in noun_tags :
            question_token_list.append(token.form)
    print(question_token_list)
    question_token = ",".join(question_token_list)
    # total_words = len(question_token_list)
    # token_count = Counter(question_token_list)
    # tf_dict = {word: count/total_words for word, count in token_count.items()}
    vector = text_vectorizer.transform([question_token])
    # vector = [tf_dict.get(word,0)*text_idf_dict.get(word,0) for word in text_vocabulary_dict]
    dense = vector.toarray()
    embedding = text_pca.transform(dense)
    rag_data = collection.query(query_embeddings=embedding, n_results=n_results)
    path = []
    name = []
    savetime = []
    text = []
    token=[]
    for n in range(n_results) :
        path.append(rag_data['ids'][0][n])
        name.append(rag_data['metadatas'][0][n]['name'])
        savetime.append(rag_data['metadatas'][0][n]['savetime'])
    for p in path :
        sql = f"""SELECT sText FROM {mariaDBtable} WHERE sFilePath = "{p}" """
        textdata=read_to_mariadb(sql)
        text.append(textdata[0][0])
        sql = f"""SELECT sToken FROM {mariaDBtable} WHERE sFilePath = "{p}" """
        tokendata=read_to_mariadb(sql)
        token.append(tokendata[0][0])

    rag_result = pd.DataFrame({"path":path,
                               "name":name,
                               "savetime":savetime,
                               "text":text,
                               "token":token})
    
    # bm25_result = bm25_get_scores(question,rag_result,"token") 
    # result_sorted = bm25_result.sort_values(by='scores',ascending=False).head(n_results)

    return rag_result

def get_reg_name_data(question, n_results):
    client = chromadb.PersistentClient(path=chromanamepath)
    collection = client.get_or_create_collection(name=collectionname,
                                                embedding_function=nameEmbeddingFunction(),
                                                metadata={'hnsw:space':'cosine'})
    token_question = synonym_preprocessing(question)
    noun_tags = ['NNG','NNP','NNB','NR','NP','SL','SN']
    question_token_list = []
    tokens = kiwi_app.tokenize(token_question)
    for token in tokens :
        if token.tag in noun_tags :
            question_token_list.append(token.form)
    print(question_token_list)
    question_token = ",".join(question_token_list)
    vector = name_vectorizer.transform([question_token])
    dense = vector.toarray()
    embedding = name_pca.transform(dense)
    # total_words = len(question_token_list)
    # token_count = Counter(question_token_list)
    # tf_dict = {word: count/total_words for word, count in token_count.items()}
    # vector = [tf_dict.get(word,0)*name_idf_dict.get(word,0) for word in name_vocabulary_dict]
    # dense = np.array([vector])
    # embedding = name_pca.transform(dense)
    rag_data = collection.query(query_embeddings=embedding, n_results=n_results)
    path = []
    name = []
    savetime = []
    text = []
    token=[]
    for n in range(n_results) :
        path.append(rag_data['ids'][0][n])
        name.append(rag_data['metadatas'][0][n]['name'])
        savetime.append(rag_data['metadatas'][0][n]['savetime'])
    for p in path :
        sql = f"""SELECT sText FROM {mariaDBtable} WHERE sFilePath = "{p}" """
        textdata=read_to_mariadb(sql)
        text.append(textdata[0][0])
        sql = f"""SELECT sToken FROM {mariaDBtable} WHERE sFilePath = "{p}" """
        tokendata=read_to_mariadb(sql)
        token.append(tokendata[0][0])

    rag_result = pd.DataFrame({"path":path,
                               "name":name,
                               "savetime":savetime,
                               "text":text,
                               "token":token})

    # bm25_result = bm25_get_scores(question,rag_result,'name') 
    # result_sorted = bm25_result.sort_values(by='scores',ascending=False).head(n_results)

    return rag_result

def call_llm(context,question):
    response = ollama.chat(
        model='exaone3.5:latest',
        messages=[
            {'role':'system',
            'content':system_prompt,},
            {'role':'user',
             'content':f'Context:{context}, Question:{question}, 반드시 3~5문장으로 간결하게 요약하세요.'},])
    return response['message']['content']

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


def bm25_get_scores(question,base_data,col_type):
    # 초기 설정
    k1 = 1.5
    b = 0.75
    corpus_tokens = base_data[col_type].tolist()
    corpus = []
    for c in corpus_tokens :
        if col_type == 'token':
            cor = c.split(',')
        elif col_type == 'name' :
            token_list = []
            text = synonym_preprocessing(c)
            noun_tags = ['NNG','NNP','NNB','NR','NP','SL','SN']
            tokens = kiwi_app.tokenize(text)
            for token in tokens :
                if token.tag in noun_tags :
                    token_list.append(token.form)
            cor = token_list
        corpus.append(cor)
    N = len(corpus)
    avgdl = sum(len(doc) for doc in corpus) / N
    idf = bm25_idf_cal(corpus, N)

    # 질문 텍스트 데이터 전처리
    question = synonym_preprocessing(question)
    noun_tags = ['NNG','NNP','NNB','NR','NP','SL','SN']
    question_token_list = []
    tokens = kiwi_app.tokenize(question)
    for token in tokens :
        if token.tag in noun_tags :
            question_token_list.append(token.form)
        
    # 문서 BM25 점수 계산
    scores = []
    for doc in corpus:
        score = 0
        doc_len = len(doc)
        doc_freq = Counter(doc)
        if doc_len != 0 :
            for word in question_token_list :
                if word not in doc_freq:
                    continue
                else :
                    tf = doc_freq[word]
                    numerator = tf * (k1 + 1)
                    denominator = tf * k1 * (1 - b + b * (doc_len / avgdl))
                    score += idf[word] * numerator / denominator
                
            scores.append(score)

    base_data['scores'] = scores
    return base_data

def bm25_idf_cal(corpus,N):
    # 단어별로 등장한 문서 수를 계산
    idf_dict = {}
    for doc in corpus :
        unique_words = set(doc)
        for word in unique_words :
            idf_dict[word] = idf_dict.get(word,0) + 1
    # idf 공식 적용
    for word, freq in idf_dict.items():
        idf_dict[word] = math.log((N-freq+0.5)/(freq+0.5)+1)
    return idf_dict


app = FastAPI()

class QuestionRequest(BaseModel):
    question: str
    n_results : str

class SummaryRequest(BaseModel):  
    text: str
    question : str
    
@app.post("/ask/")
async def ask_question(request: QuestionRequest):
    question = request.question
    n_results = request.n_results
    print("내용 검색 질문 : " + question)
    result_sorted = get_reg_text_data(question,int(n_results))
    path = result_sorted["path"].tolist()
    name = result_sorted["name"].tolist()
    savetime = result_sorted["savetime"].tolist()
    text = result_sorted["text"].tolist()
    return {"path": path, "name":name,"savetime":savetime,'text':text}

@app.post("/ask2/")
async def ask_question(request: QuestionRequest):
    question = request.question
    n_results = request.n_results
    print("제목 검색 질문 : " + question)
    result_sorted = get_reg_name_data(question,int(n_results))
    path = result_sorted["path"].tolist()
    name = result_sorted["name"].tolist()
    savetime = result_sorted["savetime"].tolist()
    text = result_sorted["text"].tolist()
    return {"path": path, "name":name,"savetime":savetime,'text':text}

@app.post("/ask3/")
async def ask_question(request: QuestionRequest):
    question = request.question
    n_results = request.n_results
    print("제목+내용 검색 질문 : " + question)
    text_result = get_reg_text_data(question,int(n_results))
    name_result = get_reg_name_data(question,int(n_results))

    text_bm25_result = bm25_get_scores(question,text_result,"token") 
    text_result_sorted = text_bm25_result.sort_values(by='scores',ascending=False).head(int(n_results))
    name_bm25_result = bm25_get_scores(question,name_result,'name') 
    name_result_sorted = name_bm25_result.sort_values(by='scores',ascending=False).head(int(n_results))


    text_weight = 0.5
    name_weight = 0.5
    merged = pd.merge(text_result_sorted,name_result_sorted, on='path')
    merged['weight_scores'] = text_weight * merged['scores_x'] + name_weight * merged['scores_y']

    only_text = text_result_sorted[~text_result_sorted['path'].isin(merged['path'])]
    only_text = only_text.copy()
    only_text['weight_scores'] = text_weight * only_text['scores']

    only_name = name_result_sorted[~name_result_sorted['path'].isin(merged['path'])]
    only_name = only_name.copy()
    only_name['weight_scores'] = name_weight * only_name['scores']
    
    merged_result = merged[["path","name_x","savetime_x",'text_x','weight_scores']]
    merged_result.columns = ["path","name","savetime",'text','weight_scores']
    only_text_result = only_text[["path","name","savetime",'text','weight_scores']]
    only_name_result = only_name[["path","name","savetime",'text','weight_scores']]
    
    final_result = pd.concat([merged_result,only_text_result,only_name_result])
    final_result = final_result.sort_values(by='weight_scores',ascending=False).head(int(n_results))

    path = final_result["path"].tolist()
    name = final_result["name"].tolist()
    savetime = final_result["savetime"].tolist()
    text = final_result["text"].tolist()

    return  {"path": path, "name":name,"savetime":savetime,'text':text}


@app.post("/summary/")
async def summarize(request: SummaryRequest):
    text = request.text
    question = request.question
    # print(f"받은 텍스트: {text}")
    if question :
        print("질문요약")
    else :
        print("문서요약")
    answer = call_llm(text,question)
    return {"answer": answer}

# uvicorn 3_file_search:app --host 0.0.0.0 --port 8002 --timeout-keep-alive 600