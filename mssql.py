import pyodbc
from datetime import datetime, timedelta

def save_text_tomssql(FilePath,FileName,SaveTime,text):
    conn = None
    cursor = None

    try:
        server = "50.201.209.73,1433"
        database = "FileSearch"
        username = "predict-eye"
        password = "predict-eye"

        conn = pyodbc.connect(f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}')
        cursor = conn.cursor()

        # 텍스트를 8000바이트 단위로 자르기 (UTF-8 기준)
        byte_text = text.encode('utf-8')
        chunks = []
        start = 0
        while start < len(byte_text):
            end = start + 8000
            # 문자 깨짐 방지 위해 적절한 위치에서 자르기
            while end < len(byte_text) and (byte_text[end] & 0xC0) == 0x80:
                end -= 1
            chunk = byte_text[start:end].decode('utf-8', errors='ignore')
            chunks.append(chunk)
            start = end

        # 문자열 시간을 datetime으로 변환
        base_time = datetime.strptime(SaveTime, "%Y-%m-%d %H:%M:%S")
        
        query = """
        INSERT INTO FileTextSave_New(sFilePath,sFileName,sSaveTime,sText)
        VALUES(?,?,?,?)
        """

        for i, chunk in enumerate(chunks):
            save_time = base_time + timedelta(seconds=i)
            formatted_time = save_time.strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute(query, (FilePath, FileName, formatted_time, chunk))
        conn.commit()

        query = """
        INSERT INTO FileName_New(sFilePath,sFileName,sSaveTime,sMariaDBYN)
        VALUES(?,?,?,N'N')
        """

        cursor.execute(query,(FilePath,FileName,SaveTime))
        conn.commit()

        print("Text Successfully saved to MSSQL.")

    except pyodbc.Error as e:
        print(f"Error while connecting to MSSQL: {e}")
    except Exception as e:
        print(f"Unexpected error while saving text to MSSQL: {e}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

        print("Database connection cloded.")

def check_sFileName_exists(sFilname:str,sSaveTime:str)->bool:
    
    conn = None
    cursor = None

    try:
        server = "50.201.209.73,1433"
        database = "FileSearch"
        username = "predict-eye"
        password = "predict-eye"

        conn = pyodbc.connect(f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}')
        cursor = conn.cursor()

        query = f"Select Count(*) From FileName_New Where sFileName = N'{sFilname}' And sSaveTime = N'{sSaveTime}'"
        cursor.execute(query)

        print(query)

        count = cursor.fetchone()[0]

        return count == 0
    
    except Exception as e:
        print(f"중복 파일 검색 중 오류발생:{e}")
        return False
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

if __name__=="__main__":
   #save_text_tomssql("AA","BB","CC","DD")
   sfilename = "[설비이상]_191121 3라인 연신 하TAC 수세수절 Nip Roller 주름 발생.xlsx"
   sSavetime = "2019-12-27 17:09:06"
   result = check_sFileName_exists(sfilename,sSavetime)
   print(f"'{sfilename}' 사용 가능 여부:{result}")