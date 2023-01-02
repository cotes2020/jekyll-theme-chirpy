
# from urllib.request import urlopen

# from bs4 import BeautifulSoup

import requests
import re

import nltk   
from urllib.request import urlopen



# # from csv import writer
# # import time
# # import random
# # from lxml import etree as et

# # from urllib.request import Request, urlopen
# # from urllib.error import URLError, HTTPError




# url = "https://www.irtliving.com/Apartments-In/Charleston-SC/Talison-Row"
# # pages_url=[]
# # listing_url=[]


# # #Opening
# # req = Request(url, headers = header)

# # #Open url
# # response = urlopen(req)

# # #Read HTML
# # print(response.read())


# # html_bytes = urlopen(url).read()

# # html = html_bytes.decode("utf-8")

# # print(html)

# # for i in range (1,23):
# #     page_url=url + str(i)
# #     pages_url.append(page_url)

def set_cookies(url):
    r = requests.get(
        the_url, 
        timeout=30,
        headers=header,
        )
    code = r.status_code
    # print(code)
    if code == 200:
        # print(type(r.text))
        # print(r.json())  
        # print(type(r.json()))
        # print(r.cookies)
        jar = requests.cookies.RequestsCookieJar()
        for key, value in r.cookies.items():
            jar.set(key,value)
    return jar
        

def get_html_text(the_url, header, jar):
    r = requests.get(
        the_url, 
        timeout=30,
        headers=header,
        cookies=jar,
        )
    code = r.status_code
    # print(code)
    if code == 200:
        # print(type(r.text))
        # print(r.json())  
        # print(type(r.json()))
        # print(r.cookies)
        # for key, value in r.cookies.items():
        #     print(key + '=' + value)

        # return "get r.text"
        print(r.text)
        return r
    else: 
        return "error"
    
    
def get_content(r):
    pattern = re.compile('explore-feed.*?question_link.*?>(.*?)</a>', re.S)
    content = re.findall(pattern, r.text)
    print(content)
    
    # try:
    #     r = requests.get(the_url, timeoue=30)
        
    #     code = r.status_code
    #     r.raise_for_status()
        
    #     r.encoding = r.apparent_encoding
    #     return r.text 
    
    # except:
    #     return "error"

def clean_html(html):
    """
    Copied from NLTK package.
    Remove HTML markup from the given string.

    :param html: the HTML string to be cleaned
    :type html: str
    :rtype: str
    """

    # First we remove inline JavaScript/CSS:
    cleaned = re.sub(r"(?is)<(script|style).*?>.*?(</\1>)", "", html.strip())
    # Then we remove html comments. This has to be done before removing regular
    # tags since comments can contain '>' characters.
    cleaned = re.sub(r"(?s)<!--(.*?)-->[\n]?", "", cleaned)
    # Next we can remove the remaining tags:
    cleaned = re.sub(r"(?s)<.*?>", " ", cleaned)
    # Finally, we deal with whitespace
    cleaned = re.sub(r"&nbsp;", " ", cleaned)
    cleaned = re.sub(r"  ", " ", cleaned)
    cleaned = re.sub(r"  ", " ", cleaned)
    return cleaned.strip()

def test(url): 
    html = urlopen(url).read()    
    raw = nltk.clean_html(html)  
    print(raw)



# # def get_dom(the_url):
# #     response = requests.get(the_url, headers=header)
# #     soup = BeautifulSoup(response.text,'lxml')
# #     dom = et.HTML(str(soup))
# #     return dom


# # def get_listing_url(page_url):
# #     dom = get_dom(page_url)
# #     page_link_list=dom.xpath('/html/body/section[2]')
# #     # for page_link in page_link_list:
# #     #     listing_url.append("https://www.pararius.com"+page_link)
# #     print(page_link_list)


# # # dom = get_dom(page_url)
# # # print(dom)

# # # get_listing_url(page_url)


# # soup = BeautifulSoup(html, "html.parser")
# # # soup = BeautifulSoup(html, "html5lib")

# # # print(soup.get_text().replace())

# # for tag in soup.find_all('body', class_='floating-cta-activated'):
# #     print(tag)

# # # print(ul)

# # # for u in ul:
# # #     if u.find.all('li', class_="fp-group-item"):
# # #         print(u)

# import time
# from selenium import webdriver
# from selenium.webdriver.common.keys import Keys
# import pandas as pd

# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.wait import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC

# # Establish chrome driver and go to report site URL
# driver = webdriver.Chrome()

# driver.implicitly_wait(10) # seconds

# driver.get(url)

# ID = "id"
# NAME = "name"
# XPATH = "xpath"
# LINK_TEXT = "link text"
# PARTIAL_LINK_TEXT = "partial link text"
# TAG_NAME = "tag name"
# CLASS_NAME = "class name"
# CSS_SELECTOR = "css selector"

# # myDynamicElement = driver.find_element_by_id("floorplan-overview-content")

# try:
#     # element = WebDriverWait(driver, 20).until(
#     #     EC.presence_of_element_located((By.ID, "floorplan-overview-content"))
#     # )

#     # myDynamicElement = driver.find_element(By.XPATH, '//*[@id="floorplan-overview-content"]')


#     myDynamicElement = driver.find_element(By.ID, 'floorplan-overview-content')

#     print("yes")
# finally:
#     driver.quit()



# # print(players)


if __name__ == "__main__":
    
    the_url = "https://www.livetalisman.com/redmond/talisman/conventional/"
    
    header = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.5060.66 Safari/537.36"}
    
    
    # jar = set_cookies(the_url)
    
    # get_html_text(the_url, header, jar)
    
    test(the_url)
