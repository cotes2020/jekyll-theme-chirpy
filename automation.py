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
from flask import Flask, request, jsonify

app = Flask(__name__)

ysfdir = "."

def get_clean_title(title):
    cleaned_title = re.sub(r'[^\w\s]', '', title)
    words = cleaned_title.lower().split()
    first_word = words[0]
    return first_word

degis = 0

@app.route('/update', methods=['GET'])
def update():
    global degis  # Use global keyword to modify global variable

    try:
        # RSS feed URL
        rss_url = "https://api.rss2json.com/v1/api.json?rss_url=https://medium.com/feed/@yusuf.deniz"

        # Send API request
        response = requests.get(rss_url)

        # Check for successful response
        if response.status_code != 200:
            return jsonify({"message": f"Error: Unexpected response from server - Status Code: {response.status_code}"}), 500

        data = response.json()

        # Check if response contains 'items'
        items = data.get("items", [])

        degis = len(items) - degis

        # Target directory
        output_directory = ysfdir + '/_posts'

        # Delete all files in the target directory
        existing_files = os.listdir(output_directory)
        for file in existing_files:
            file_path = os.path.join(output_directory, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

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

        # Update bookshelf.md
        bookshelf_file_path = ysfdir + "/_tabs/bookshelf.md"
        with open(bookshelf_file_path, 'w') as f:
            f.write("---\n")
            f.write("# the default layout is 'page'\n")
            f.write("icon: fas fa-book\n")
            f.write("order: 6\n")
            f.write("---\n\n")
            for book in books:
                title = book.get("title", "")
                f.write(f"- [{title}](/posts/{get_clean_title(title=title)})\n")

        # Update projects.md
        git_url = "https://api.github.com/users/ysf-dnz/repos"
        git_response = requests.get(git_url)
        projects_file_path = ysfdir + "/_tabs/projects.md"
        with open(projects_file_path, 'w') as f:
            f.write("---\n")
            f.write("# the default layout is 'page'\n")
            f.write("icon: fas fa-diagram-project\n")
            f.write("order: 5\n")
            f.write("---\n\n")
            for repo in git_response.json():
                name = repo.get("name", "")
                description = repo.get("description", "")
                f.write(f"- [{name}](https://github.com/ysf-dnz/{name})\n")
                f.write(f"   {description}\n")

        # Git commands
        os.system("git add .")
        os.system(f"git commit -m 'Content update {datetime.now()}'")
        os.system("git push origin master")
        print("Git push successful")

        return jsonify({"message": "Update successful", "degisiklik": degis}), 200

    except requests.exceptions.RequestException as e:
        return jsonify({"message": f"Error: Request Exception - {e}"}), 500

    except json.JSONDecodeError as e:
        return jsonify({"message": f"Error: JSON Decode Error - {e}\nResponse Text: {response.text}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000)
