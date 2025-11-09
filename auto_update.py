import requests
import html2markdown
from bs4 import BeautifulSoup
import re
from dateutil import parser
from datetime import datetime
import os
import time
import json

ysfdir = "."
rss_url = "https://api.rss2json.com/v1/api.json?rss_url=https://medium.com/feed/@yusuf.deniz"
git_url = "https://api.github.com/users/ysf-dnz/repos"

STATE_FILE = "last_state.json"


def get_clean_title(title):
    cleaned_title = re.sub(r"[^\w\s]", "", title)
    words = cleaned_title.lower().split()
    first_word = words[0] if words else "post"
    return first_word


def load_last_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {"latest_pub_date": None}


def save_last_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


def yaml_escape(value):
    """Escape YAML-breaking characters safely."""
    if isinstance(value, str):
        value = value.replace('"', '\\"')
        if ":" in value or '"' in value or "'" in value:
            return f'"{value}"'
    return value


def process_feed():
    print(f"\n[{datetime.now()}] Checking feed...")
    try:
        response = requests.get(rss_url)
        if response.status_code != 200:
            print(f"Feed error: {response.status_code}")
            return

        data = response.json()
        items = data.get("items", [])
        if not items:
            print("No feed items found.")
            return

        last_state = load_last_state()
        last_pub_date = last_state.get("latest_pub_date")

        new_items = []
        for item in items:
            pub_date = parser.parse(item.get("pubDate", ""))
            if not last_pub_date or pub_date > parser.parse(last_pub_date):
                new_items.append(item)

        if not new_items:
            print("No new articles.")
            return

        print(f"New articles found: {len(new_items)}")

        # Sort by date ascending to process oldest first
        new_items.sort(key=lambda x: parser.parse(x["pubDate"]))

        output_directory = os.path.join(ysfdir, "_posts")
        os.makedirs(output_directory, exist_ok=True)

        books = []

        for item in new_items:
            title = item.get("title", "")
            description_html = item.get("description", "")
            categories = item.get("categories", [])
            tags = item.get("categories", [])
            pub_date = parser.parse(item.get("pubDate", ""))

            if "ｂｏｏｋ" in tags:
                books.append(item)

            # Convert to Markdown
            soup = BeautifulSoup(description_html, "html.parser")
            description_markdown = html2markdown.convert(str(soup))

            formatted_date = pub_date.strftime("%Y-%m-%d")
            clean_title = get_clean_title(title)
            file_name = f"{formatted_date}-{clean_title}.md"

            # Safe YAML block
            yaml_front_matter = (
                "---\n"
                f"title: {yaml_escape(title)}\n"
                f"date: \"{pub_date.strftime('%Y-%m-%d %H:%M:%S')}\"\n"
                f"categories:\n"
                + "".join([f"  - {yaml_escape(c)}\n" for c in categories])
                + "tags:\n"
                + "".join([f"  - {yaml_escape(t)}\n" for t in tags])
                + "---\n\n"
            )

            output_file_path = os.path.join(output_directory, file_name)
            with open(output_file_path, "w", encoding="utf-8") as file:
                file.write(yaml_front_matter)
                file.write(description_markdown)

        # Update bookshelf.md
        bookshelf_file_path = os.path.join(ysfdir, "_tabs/bookshelf.md")
        os.makedirs(os.path.dirname(bookshelf_file_path), exist_ok=True)
        with open(bookshelf_file_path, "w", encoding="utf-8") as f:
            f.write("---\n")
            f.write("# the default layout is 'page'\n")
            f.write("icon: fas fa-book\n")
            f.write("order: 6\n")
            f.write("---\n\n")
            for book in books:
                title = book.get("title", "")
                f.write(f"- [{title}](/posts/{get_clean_title(title=title)})\n")

        # Update projects.md
        git_response = requests.get(git_url)
        projects_file_path = os.path.join(ysfdir, "_tabs/projects.md")
        with open(projects_file_path, "w", encoding="utf-8") as f:
            f.write("---\n")
            f.write("# the default layout is 'page'\n")
            f.write("icon: fas fa-diagram-project\n")
            f.write("order: 5\n")
            f.write("---\n\n")
            for repo in git_response.json():
                name = repo.get("name", "")
                description = repo.get("description", "")
                f.write(f"- [{name}](https://github.com/ysf-dnz/{name})\n")
                if description:
                    f.write(f"   {description}\n")

        # Update state
        latest_date = parser.parse(items[0]["pubDate"]).isoformat()
        save_last_state({"latest_pub_date": latest_date})

        # Git operations
        os.system(
            "git add . && git commit -m 'auto: new Medium post sync' && git push origin master"
        )
        print("Git push successful")

    except Exception as e:
        print(f"Error: {e}")


def main():
    while True:
        process_feed()
        time.sleep(600)  # 10 dakika


if __name__ == "__main__":
    main()

