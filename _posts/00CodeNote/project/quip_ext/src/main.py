import base64
import logging
import os
import sys
import time
import zipfile

import requests
from bs4 import BeautifulSoup

# Logs will go to CloudWatch log group corresponding to lambda,
# If Lambda has the necessary IAM permissions.
# Set logLevel to logging.INFO or logging.DEBUG for debugging.
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
LOGGER = logging.getLogger(__name__)
# Retrieve log level from Lambda Environment Variables
LOGGER.setLevel(level=os.environ.get("LOG_LEVEL", "INFO").upper())


# Define your constants
QUIP_API_BASE_URL = "https://quip.com/"
QUIP_ACCESS_TOKEN = "YOUR_ACCESS_TOKEN"
ROOT_DIR = "abc"
MINUTELY_LIMIT = 50
HOURLY_LIMIT = 750

# Initialize variables
waiter = {
    "totalC": 0,
    "exportPaused": False,
    "exportPausedByUser": False,
    "running": False,
    "zipFile": None,
}


# Helper function to wait for unpause
def wait_for_unpause():
    while waiter["exportPaused"] or waiter["exportPausedByUser"]:
        print(
            f"Paused: {waiter['exportPaused']}, Paused by User: {waiter['exportPausedByUser']}, Running: {waiter['running']}"
        )
        time.sleep(2)


# Helper function to clean strings
def clean_string(string):
    tr = {"ä": "ae", "ü": "ue", "ö": "oe", "ß": "ss", "Ä": "Ae", "Ü": "Ue", "Ö": "Oe"}
    clean = "".join(e for e in string if e.isalnum() or e in tr)
    clean = "".join(tr.get(c, c) for c in clean)
    return clean


# Helper function to make API requests
def make_api_request(url):
    headers = {"Authorization": f"Bearer {QUIP_ACCESS_TOKEN}"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()


# Function to export a thread
def export_thread(thread_id, thread_title, html):
    # Create a folder in the zip
    folder_name = f"_images_{thread_title[:6]}_{thread_id}"
    zip_file.writestr(f"{ROOT_DIR}/{folder_name}/", "")

    # Extract and replace images in HTML
    images = []
    soup = BeautifulSoup(html, "html.parser")
    for img in soup.find_all("img"):
        src = img["src"]
        image_folder_name = f"_images_{thread_title[:6]}_{thread_id}"
        image_file_name = f"{thread_title[:6]}_{len(images)}.png"
        image_file_path = f"{ROOT_DIR}/{image_folder_name}/{image_file_name}"
        images.append(
            {
                "src": src,
                "image_folder_name": image_folder_name,
                "image_file_name": image_file_name,
                "image_file_path": image_file_path,
            }
        )
        img["src"] = image_file_path

    # Write HTML file
    html_file_name = f"{ROOT_DIR}/{thread_title}.html"
    zip_file.writestr(html_file_name, str(soup))

    # Write images to zip
    for image in images:
        src = image["src"]
        image_file_path = image["image_file_path"]
        blob_path = src.split("/", 1)[-1]
        img_data = make_api_request(QUIP_API_BASE_URL + blob_path)
        zip_file.writestr(image_file_path, base64.b64decode(img_data["blob"]["data"]))


# Function to export a folder
def export_folder(folder_id, folder_title):
    folder_name = clean_string(folder_title)
    folder_path = f"{ROOT_DIR}/{folder_name}"
    zip_file.writestr(folder_path + "/", "")
    folder_data = make_api_request(QUIP_API_BASE_URL + f"folders/{folder_id}")
    for child in folder_data["children"]:
        export_item(child, folder_path)


# Function to export an item (either thread or folder)
def export_item(item, parent_path):
    if "thread_id" in item:
        thread_data = make_api_request(
            QUIP_API_BASE_URL + f"threads/{item['thread_id']}"
        )
        thread_title = clean_string(thread_data["thread"]["title"])
        export_thread(item["thread_id"], thread_title, thread_data["html"])
    elif "folder_id" in item:
        folder_data = make_api_request(
            QUIP_API_BASE_URL + f"folders/{item['folder_id']}"
        )
        folder_title = folder_data["folder"]["title"]
        export_folder(item["folder_id"], folder_title)


# Function to start exporting
def start_exporting():
    global zip_file
    global waiter
    global ROOT_DIR
    global QUIP_ACCESS_TOKEN

    target_folder_id = "abc"

    # Initialize zip file
    zip_file = zipfile.ZipFile(f"{ROOT_DIR}.zip", "w")
    LOGGER.info(f"+++++++ create zip file: {ROOT_DIR}.zip")

    # Set up rate limiting
    waiter = {
        "totalC": 0,
        "exportPaused": False,
        "exportPausedByUser": False,
        "running": False,
        "zipFile": None,
    }

    # Export items
    export_item({"folder_id": target_folder_id}, ROOT_DIR)
    LOGGER.info(f"+++++++got zip file: {ROOT_DIR}.zip")

    # Close the zip file
    zip_file.close()

    print(f"Export finished. Total API calls: {waiter['totalC']}")


if __name__ == "__main__":
    start_exporting()
