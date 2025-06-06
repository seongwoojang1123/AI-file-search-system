import pymysql
import pyodbc
import re
from kiwipiepy import Kiwi
import pandas as pd
import json

def read_to_mssql(sql):
    conn = None
    cursor = None
    server = "50.201.209.73,1433"
    database = "FileSearch"
    username = "predict-eye"
    password = "predict-eye"
    conn = pyodbc.connect(f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}')
    cursor = conn.cursor()
    cursor.execute(sql)
    data = cursor.fetchall()
    conn.commit()
    conn.close()
    return data

def update_to_mssql(sql):
    conn = None
    cursor = None
    server = "50.201.209.73,1433"
    database = "FileSearch"
    username = "predict-eye"
    password = "predict-eye"
    conn = pyodbc.connect(f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}')
    cursor = conn.cursor()
    cursor.execute(sql)
    conn.commit()
    conn.close()

def send_to_mariadb(sql):
    conn = None
    cur = None
    conn = pymysql.connect(host='50.201.224.41',port=3306,user='wi',password='5882',db='filesearch',charset='utf8')
    cur = conn.cursor()
    cur.execute("SET SESSION sql_mode = 'NO_BACKSLASH_ESCAPES'")
    cur.execute(sql)
    conn.commit()
    conn.close()

# 텍스트 전처리, 동의어 처리  
def synonym_preprocessing(text) :
    text = text.lower()
    text = text.replace("'"," ")
    # text = text.replace("\u"," ")
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

if __name__ == '__main__' :
    # 고유명사 추가
    kiwi_app = Kiwi(model_path='C:\\Users\\fpc\\Downloads',model_type='cong')
    nnp_list = ['group.txt','line.txt','name.txt','NNP.txt','product.txt']
    for n in nnp_list :
        with open(f'./NNP_List/{n}','r',encoding='utf-8') as file:
            nnp = file.read()
        for n in nnp.split('\n') :
            kiwi_app.add_user_word(n, 'NNP', 6)



    print("MS-SQL 데이터 가져오기 시작")
    sql = '''SELECT FT.* FROM FileName_deep8b FN JOIN FileTextSave_deep8b FT ON FN.sFilePath = FT.sFilePath WHERE FN.sMariaDBYN = N'N' AND FT.sText IS NOT NULL '''
    data = read_to_mssql(sql)
    columns = ['path','name','savetime','text']
    data_pd = pd.DataFrame(columns=columns)
    print("MS-SQL 데이터 가져오기 완료")
    print("MS-SQL 데이터 행 개수 : " + str(len(data)))


    print("텍스트 데이터 전처리 시작")
    for d in data :
        path = d[0]
        name = d[1]
        savetime = d[2]
        text = d[3]
        name_lower = name.lower()
        if "업무일지" in name or "readme" in name_lower or "license" in name_lower or 'venv' in path or 'key.txt' in name_lower :
            continue
        else :
            text = synonym_preprocessing(text)
            text = kiwi_app.space(text)
            df = pd.DataFrame(data=[[path,name,savetime,text]], columns=columns)
            data_pd = pd.concat([data_pd, df])
    data_pd_result = data_pd.groupby('path').agg({'name':'last','savetime':'last','text': lambda x : ''.join(x)}).reset_index()
    print("텍스트 데이터 전처리 완료")


    with open('config.json','r',encoding='utf-8') as file :
        config = json.load(file)
    chromatextpath = config['chromatextpath']
    chromanamepath = config['chromanamepath']
    collectionname = config['collectionname']
    mariaDBtable = config['mariaDBtable']

    print("MariaDB 저장 및 토큰 분리 시작")
    for row in data_pd_result.itertuples() :
        token_list = []
        noun_tags = ['NNG','NNP','NNB','NR','NP','SL']
        tokens = kiwi_app.tokenize(row.text)
        for token in tokens :
            if token.tag in noun_tags :
                token_list.append(token.form)
        if len(token_list) > 10 :
            sToken = ",".join(token_list)
            try :
                send_to_mariadb_sql = f"""INSERT INTO {mariaDBtable} values("{row.path}","{row.name}","{row.savetime}",'{row.text}',"{sToken}")
                ON DUPLICATE KEY UPDATE sFilePath="{row.path}",sFileName="{row.name}",sSaveTime="{row.savetime}",sText='{row.text}',sToken="{sToken}" """
                send_to_mariadb(send_to_mariadb_sql)
                sFilepath = row.path
                sFilepath = sFilepath.replace("'","''")
                update_to_mssql_sql = f"""UPDATE FileName_deep8b SET sMariaDBYN =N'Y' WHERE sFilePath = N'{sFilepath}'"""
                update_to_mssql(update_to_mssql_sql)
            except Exception as e :
                print(row.path)
                print(e)
    print("MariaDB 저장 및 토큰 분리 완료")
