
import requests
page = requests.get('https://codedamn.com')
text = page.text
content = page.content
status_code = page.status_code

from bs4 import BeautifulSoup
soup = BeautifulSoup(page.content, 'html.parser')
title = soup.title.text # gets you the text of the <title>(...)</title> 
page_body = soup.body
page_head = soup.head

results = soup.find(id="ResultsContainer")
job_elements = results.find_all("div", class_="card-content")
python_jobs = results.find_all("h2", string="Python")

images = soup.select('img')
products = soup.select('div.thumbnail')
first_h1 = soup.select('h1')[0].text
seventh_p_text = soup.select('p')[6].text 
title = soup.select('h4 > a.title')[0].text
review_label = soup.select('div.ratings')[0].text

links = soup.select('a')
for link in links:
    href = link.get('href')
    href = href.strip() if href is not None else ''
    text = link.text
    text = text.strip() if text is not None else ''




# Part 1: Loading Web Pages with 'request'
import requests
page = requests.get('https://codedamn.com')
print(page.text)
# print(page.content)
print(page.status_code)


# Part 2: Extracting title with BeautifulSoup
from bs4 import BeautifulSoup
page = requests.get("https://codedamn.com")
soup = BeautifulSoup(page.content, 'html.parser')
title = soup.title.text # gets you the text of the <title>(...)</title>


# Part 3: Soup-ed body and head
import requests
from bs4 import BeautifulSoup
# Make a request
page = requests.get("https://codedamn.com")
soup = BeautifulSoup(page.content, 'html.parser')
page_title = soup.title.text
page_body = soup.body
page_head = soup.head
# print(page_body, page_head)


# Part 4: select with BeautifulSoup
import requests
from bs4 import BeautifulSoup
# Make a request
page = requests.get("https://codedamn-classrooms.github.io/webscraper-python-codedamn-classroom-website/")
soup = BeautifulSoup(page.content, 'html.parser')
# Extract first <h1>(...)</h1> text
first_h1 = soup.select('h1')[0].text
seventh_p_text = soup.select('p')[6].text
print(seventh_p_text)



# Part 5: Top items being scraped right now
import requests
from bs4 import BeautifulSoup
page = requests.get('https://codedamn-classrooms.github.io/webscraper-python-codedamn-classroom-website/')
soup = BeautifulSoup(page.content, 'html.parser')
top_items = []
product = soup.select('hdiv.thumbnail1')
for elem in product:
    title = elem.select('h4 > a.title')[0].text
    review_label = elem.select('div.ratings')[0].text
    info = {
        "title": title.strip(),
        "review": review_label.strip()
    }
    top_items.append(info)



# Part 6: Extracting Links
import requests
from bs4 import BeautifulSoup 
page = requests.get("https://codedamn-classrooms.github.io/webscraper-python-codedamn-classroom-website/")
soup = BeautifulSoup(page.content, 'html.parser') 
image_data = []
images = soup.select('img')
for image in images:
    src = image.get('src')
    alt = image.get('alt')
    image_data.append({"src": src, "alt": alt})
# print(image_data)
all_links = []
links = soup.select('a')
for link in links:
    href = link.get('href')
    href = href.strip() if href is not None else ''
    text = link.text
    text = text.strip() if text is not None else ''
    info = {
        "href": href,
        "text": text
    }
    all_links.append(info)
# print(all_links)



# Part 7: Generating CSV from data
import requests
from bs4 import BeautifulSoup
import csv
page = requests.get("https://codedamn-classrooms.github.io/webscraper-python-codedamn-classroom-website/")
soup = BeautifulSoup(page.content, 'html.parser')
all_products = []
products = soup.select('div.thumbnail')
for product in products:
    print(product)
    name = product.select('h4 > a')[0].text.strip()
    description = product.select('p.description')[0].text.strip()
    price = product.select('h4.price')[0].text.strip()
    reviews = product.select('div.ratings')[0].text.strip()
    image = product.select('img')[0].get('src')
    all_products.append({
        "name": name,
        "description": description,
        "price": price,
        "reviews": reviews,
        "image": image
    })
keys = all_products[0].keys()
with open('products.csv', 'w', newline='') as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(all_products)




# 