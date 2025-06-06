import os
import pptReader
import datetime
import mssql
import shutil

#특정 폴더의 파일 검색
def list_files_in_folders(folder_path):

    folder_path = os.path.abspath(folder_path)

    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"폴더를 찾을수 없습니다:{folder_path}")
    
    file_list = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    return file_list

#단독 실행 시 코드 (파일 검색)
#if __name__ == "__main__":
#    folder = r"D:\#Personal Space\01.투자"
#    files = list_files_in_folders(folder)
#    print(files)l

#하위 폴더 포함 전체 파일 검색
def list_all_files(folder_path):
    folder_path=os.path.abspath(folder_path)
    file_list = []
    
    for root,__,files in os.walk(folder_path):
        for file in files:
            file_list.append(os.path.join(root,file))

    return file_list

#ppt 파일 인지 확인
def is_ppt_file(file_path):
    if not isinstance(file_path,str) or not file_path.strip():
        return False
    
    file_name = os.path.basename(file_path)
    _,ext = os.path.splitext(file_path)

    if file_name.startswith("~$"):
        return False

    return ext.lower() in [".ppt",".pptx"]

#word 파일 인지 확인
def is_doc_file(file_path):
    if not isinstance(file_path,str) or not file_path.strip():
        return False
    file_name = os.path.basename(file_path)
    _,ext = os.path.splitext(file_path)
    
    if file_name.startswith("~$"):
        return False
    
    return ext.lower() in [".doc",".docx"]

#Excel 파일 인지 확인
def is_excel_file(file_path):
    if not isinstance(file_path,str) or not file_path.strip():
        return False
    
    file_name = os.path.basename(file_path)
    _,ext = os.path.splitext(file_path)

    if file_name.startswith("~$"):
        return False
    
    return ext.lower() in [".xls",".xlsx"]

#Text 파일 인지 확인
def is_Text_file(file_path):
    if not isinstance(file_path,str) or not file_path.strip():
        return False
    
    file_name = os.path.basename(file_path)
    _,ext = os.path.splitext(file_path)

    if file_name.startswith("~$"):
        return False
    
    return ext.lower() in [".txt"]

#하위 폴더까지 포함 하여 파일 검색 후 ppt, word 파일 안의 텍스트 추출
def list_all_ppt_files(folder_path):
    
    try:
        temp_folder = r"\\50.201.101.141\설비기술g\15.AI\Temp Folder"
        
        folder_path = os.path.abspath(folder_path)
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"폴더를 찾을 수 없습니다:{folder_path}")
    
        for root,_,files in os.walk(folder_path):
            for file in files:
                if is_ppt_file(file):
                    full_path = os.path.join(root,file)
                    file_name = os.path.basename(file)
                    modified_time = os.path.getmtime(full_path)
                    formatted_time = datetime.datetime.fromtimestamp(modified_time).strftime('%Y-%m-%d %H:%M:%S')
                                    
                    if mssql.check_sFileName_exists(file_name,formatted_time):
                        ppt_output = ""
                        try:
                            if len(full_path) > 250:
                                temp_path = os.path.join(temp_folder, file_name)
                                shutil.copy2(full_path, temp_path)
                                ppt_output = pptReader.extract_text_from_ppt(temp_path,file_name,formatted_time)
                                os.remove(temp_path)
                            else:
                                ppt_output = pptReader.extract_text_from_ppt(full_path,file_name,formatted_time)
                        except Exception as e:
                            pass    
                        if(ppt_output):
                            ppt_output = ppt_output.replace("\r","\n")
                        print(ppt_output)
                        print(f"파일 제목 :{file_name}")
                        print(f"최종 수정 날짜 :{formatted_time}")
                        mssql.save_text_tomssql(full_path,file_name,formatted_time,ppt_output)
                        #ppt_files.append(os.path.join(root,file))

                if is_doc_file(file):
                    full_path = os.path.join(root,file)
                    file_name = os.path.basename(file)
                    modified_time = os.path.getmtime(full_path)
                    formatted_time = datetime.datetime.fromtimestamp(modified_time).strftime('%Y-%m-%d %H:%M:%S')

                    if mssql.check_sFileName_exists(file_name,formatted_time):    
                        doc_output = ""
                        try:
                            if len(full_path) > 250:
                                temp_path = os.path.join(temp_folder, file_name)
                                shutil.copy2(full_path, temp_path)
                                doc_output = pptReader.extract_text_from_word(temp_path,file_name,formatted_time)
                                os.remove(temp_path)
                            else:
                                doc_output = pptReader.extract_text_from_word(full_path,file_name,formatted_time)
                        except Exception as e:
                            pass
                        if(doc_output):
                            doc_output = doc_output.replace("\r","\n")
                        print(doc_output)
                        print(f"파일 제목 :{file_name}")
                        print(f"최종 수정 날짜 :{formatted_time}")
                        mssql.save_text_tomssql(full_path,file_name,formatted_time,doc_output)

                if is_excel_file(file):
                    full_path = os.path.join(root,file)
                    file_name = os.path.basename(file)
                    modified_time = os.path.getmtime(full_path)
                    formatted_time = datetime.datetime.fromtimestamp(modified_time).strftime('%Y-%m-%d %H:%M:%S')

                    if mssql.check_sFileName_exists(file_name,formatted_time):    
                        excel_output = ""
                        try:
                            if len(full_path) > 250:
                                temp_path = os.path.join(temp_folder, file_name)
                                shutil.copy2(full_path, temp_path)
                                excel_output = pptReader.extract_text_from_excel(temp_path,file_name,formatted_time)
                                os.remove(temp_path)
                            else:
                                excel_output = pptReader.extract_text_from_excel(full_path,file_name,formatted_time)
                        except Exception as e:
                            pass
                        if(excel_output):
                            excel_output = excel_output.replace("\r","\n")
                        print(excel_output)
                        print(f"파일 제목l :{file_name}")
                        print(f"최종 수정 날짜 :{formatted_time}")
                        mssql.save_text_tomssql(full_path,file_name,formatted_time,excel_output)

                if is_Text_file(file):
                    full_path = os.path.join(root,file)
                    file_name = os.path.basename(file)
                    modified_time = os.path.getmtime(full_path)
                    formatted_time = datetime.datetime.fromtimestamp(modified_time).strftime('%Y-%m-%d %H:%M:%S')

                    if mssql.check_sFileName_exists(file_name,formatted_time):
                        if len(full_path) > 250:
                            temp_path = os.path.join(temp_folder, file_name)
                            shutil.copy2(full_path, temp_path)
                            text_output = pptReader.extract_text_from_Text(temp_path,file_name,formatted_time)    
                            os.remove(temp_path)
                        else:
                            text_output = pptReader.extract_text_from_Text(full_path,file_name,formatted_time)    
                        print(text_output)
                        print(f"파일 제목 :{file_name}")
                        print(f"최종 수정 날짜 :{formatted_time}")
                        mssql.save_text_tomssql(full_path,file_name,formatted_time,text_output)

    #return ppt_files
    except Exception as e:
        print(f"List 문서 추출 중 오류 발생:{e}")
        
list_all_ppt_files(r"\\50.201.101.141\설비기술g")

#file_print = list_all_ppt_files("D:\#Personal Space\00.주간보고\01.사업장장 주간보고\24년 혁신 회의 자료")

#print(file_print)