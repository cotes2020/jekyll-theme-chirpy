import requests
import html2markdown
from bs4 import BeautifulSoup
import re
from dateutil import parser
from datetime import datetime
import os
import requests
import time
import xmltodict
import json
import os

#python3 -m pip install beautifulsoup4 html2markdown xmltodict python-dateutil --upgrade

ysfdir = "."

def get_clean_title(title):
    # Başlıkta yer alan özel işaretleri çıkartın
    cleaned_title = re.sub(r'[^\w\s]', '', title)
    # Başlığı küçük harfe çevirin ve boşluklara göre ayırın
    words = cleaned_title.lower().split()
    # Sadece ilk kelimeyi alın
    first_word = words[0]
    return first_word

# RSS feed URL
rss_url = "https://api.rss2json.com/v1/api.json?rss_url=https://medium.com/feed/@yusuf.deniz"

while True:
    # API isteğini gönder
    response = requests.get(rss_url)

    books = []

    # Yanıtı JSON olarak alın
    data = response.json()

    # Sadece "items" anahtarını alın
    items = data.get("items", [])

    # Hedef dizin
    output_directory = ysfdir+'/_posts'

    # Hedef dizindeki tüm dosyaları sil
    existing_files = os.listdir(output_directory)
    for file in existing_files:
        file_path = os.path.join(output_directory, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Dosya silindi: {file_path}")


    for item in items:
        pub_date = item.get("pubDate", "")
        title = item.get("title", "")
        description_html = item.get("description", "")
        categories = item.get("categories", [])
        tags = item.get("categories", [])  # 'tags' yerine 'categories' anahtarını kullanıyoruz.

        if("ｂｏｏｋ" in tags): books.append(item)

        # HTML'i Markdown'a çevirin
        soup = BeautifulSoup(description_html, "html.parser")
        description_markdown = html2markdown.convert(str(soup))

        # Tarihi düzenleyin
        parsed_date = parser.parse(pub_date)
        formatted_date = parsed_date.strftime("%Y-%m-%d")

        # Dosya adı oluşturun (YYYY-MM-DD-title.md)
        clean_title = get_clean_title(title)
        file_name = f"{formatted_date}-{clean_title}.md"

        # YAML ön bilgisi oluşturun
        yaml_front_matter = f"""--- \n title: {title} \n date: {parsed_date.strftime("%Y-%m-%d %H:%M:%S")} \n categories: {categories} \n tags: {tags if tags else []} \n--- \n"""

        # Markdown dosyasını hedef dizine kaydedin
        output_file_path = os.path.join(output_directory, file_name)
        with open(output_file_path, "w", encoding="utf-8") as file:
            file.write(yaml_front_matter)
            file.write(description_markdown)

        print(f"Markdown dosyası oluşturuldu: {output_file_path}")

        file_path = ysfdir+"/_tabs/bookshelf.md"
        with open(file_path, 'w') as f:
            f.write("---\n")
            f.write("# the default layout is 'page'\n")
            f.write("icon: fas fa-book\n")
            f.write(f"order: 6\n")
            f.write("---\n")
            f.write("\n")
            for book in books:
                title = book.get("title", "")
                f.write(f"- [{title}](/posts/{get_clean_title(title=title)})\n")

            print(f"Yeni md dosyası oluşturuldu: {file_path}")

    git_url = "https://api.github.com/users/ysf-dnz/repos"

    response = requests.get(git_url)
    file_path = ysfdir+"/_tabs/projects.md"
    with open(file_path, 'w') as f:
        f.write("---\n")
        f.write("# the default layout is 'page'\n")
        f.write("icon: fas fa-diagram-project\n")
        f.write(f"order: 5\n")
        f.write("---\n")
        f.write("\n")
        for repo in response.json():
            name = repo.get("name", "")
            description = repo.get("description", "")
            f.write(f"- [{name}](https://github.com/ysf-dnz/{name})\n")
            f.write(f"   {description}\n")

            print(f"Yeni md dosyası oluşturuldu: {file_path}")

    

    os.system("git add .")
    os.system("git commit -m 'Content update'")
    os.system("git push origin master")
    print("Git push yapıldı")
    time.sleep(60*60)
