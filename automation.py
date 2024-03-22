import requests
import html2markdown
from bs4 import BeautifulSoup
import re
from dateutil import parser
from datetime import datetime
import os
import time
import xmltodict
import json

# Python3 -m pip install beautifulsoup4 html2markdown xmltodict python-dateutil --upgrade

ysfdir = "."

def get_clean_title(title):
    cleaned_title = re.sub(r'[^\w\s]', '', title)
    words = cleaned_title.lower().split()
    first_word = words[0]
    return first_word

# RSS feed URL
rss_url = "https://api.rss2json.com/v1/api.json?rss_url=https://medium.com/feed/@yusuf.deniz"

while True:
    try:
        # Send API request
        response = requests.get(rss_url)

        # Check for successful response
        if response.status_code != 200:
            print(f"Error: Unexpected response from server - Status Code: {response.status_code}")
            time.sleep(60*5)  # Wait for 5 minutes before trying again
            continue

        data = response.json()

        # Check if response contains 'items'
        items = data.get("items", [])

        # Target directory
        output_directory = ysfdir+'/_posts'

        # Delete all files in the target directory
        existing_files = os.listdir(output_directory)
        for file in existing_files:
            file_path = os.path.join(output_directory, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted file: {file_path}")

        books = []
        for item in items:
            pub_date = item.get("pubDate", "")
            title = item.get("title", "")
            description_html = item.get("description", "")
            categories = item.get("categories", [])
            tags = item.get("categories", [])

            if "ｂｏｏｋ" in tags:  # Check for 'ｂｏｏｋ' in tags
                books.append(item)

            # HTML to Markdown conversion
            soup = BeautifulSoup(description_html, "html.parser")
            description_markdown = html2markdown.convert(str(soup))

            # Parse date
            parsed_date = parser.parse(pub_date)
            formatted_date = parsed_date.strftime("%Y-%m-%d")

            # Create file name (YYYY-MM-DD-title.md)
            clean_title = get_clean_title(title)
            file_name = f"{formatted_date}-{clean_title}.md"

            # YAML front matter
            yaml_front_matter = f"""---\ntitle: {title}\ndate: {parsed_date.strftime("%Y-%m-%d %H:%M:%S")}\ncategories: {categories}\ntags: {tags if tags else []}\n---\n"""

            # Write Markdown file to target directory
            output_file_path = os.path.join(output_directory, file_name)
            with open(output_file_path, "w", encoding="utf-8") as file:
                file.write(yaml_front_matter)
                file.write(description_markdown)

            print(f"Created Markdown file: {output_file_path}")

        # Update bookshelf.md
        file_path = ysfdir + "/_tabs/bookshelf.md"
        with open(file_path, 'w') as f:
            f.write("---\n")
            f.write("# the default layout is 'page'\n")
            f.write("icon: fas fa-book\n")
            f.write("order: 6\n")
            f.write("---\n\n")
            for book in books:
                title = book.get("title", "")
                f.write(f"- [{title}](/posts/{get_clean_title(title=title)})\n")

            print(f"Updated md file: {file_path}")

        # Update projects.md
        git_url = "https://api.github.com/users/ysf-dnz/repos"
        response = requests.get(git_url)
        file_path = ysfdir + "/_tabs/projects.md"
        with open(file_path, 'w') as f:
            f.write("---\n")
            f.write("# the default layout is 'page'\n")
            f.write("icon: fas fa-diagram-project\n")
            f.write("order: 5\n")
            f.write("---\n\n")
            for repo in response.json():
                name = repo.get("name", "")
                description = repo.get("description", "")
                f.write(f"- [{name}](https://github.com/ysf-dnz/{name})\n")
                f.write(f"   {description}\n")

                print(f"Updated md file: {file_path}")

        # Git commands
        os.system("git add .")
        os.system(f"git commit -m 'Content update {datetime.now()}'")
        os.system("git push origin master")
        print("Git push successful")

        time.sleep(60*60)  # Wait for 1 hour before running again

    except requests.exceptions.RequestException as e:
        print(f"Error: Request Exception - {e}")
        time.sleep(60*5)  # Wait for 5 minutes before trying again

    except json.JSONDecodeError as e:
        print(f"Error: JSON Decode Error - {e}")
        print(f"Response Text: {response.text}")
        time.sleep(60*5)  # Wait for 5 minutes before trying again

