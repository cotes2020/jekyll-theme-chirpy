import concurrent.futures

import requests


# Function to fetch repository size
def get_repo_size(url):
    repo_name = url.split("/")[-1].strip()
    response = requests.get(f"https://api.github.com/repos/{repo_name}")
    if response.status_code == 200:
        repo_data = response.json()
        return repo_data.get("size", "NA")
    else:
        return "NA"


# Task 1: Copying fileA to fileB
with open("fileA.txt") as infile, open("fileB.txt", "w") as outfile:
    outfile.write(infile.read())

# Task 2: Getting repository sizes
with concurrent.futures.ThreadPoolExecutor() as executor:
    urls = [line.strip() for line in open("fileA.txt")]
    repo_sizes = list(executor.map(get_repo_size, urls))

# Task 3: Updating fileB with repository sizes
with open("fileB.txt", "r+") as fileB:
    for url, size in zip(urls, repo_sizes):
        fileB.write(f"{url}\nSize: {size}\n")

# Task 4: Retrying repository sizes
with concurrent.futures.ThreadPoolExecutor() as executor:
    for url, size in zip(urls, repo_sizes):
        if size == "NA":
            retries = 0
            while retries < 3:
                new_size = executor.submit(get_repo_size, url).result()
                if new_size != "NA":
                    with open("fileB.txt") as fileB:
                        file_content = fileB.read()
                    updated_content = file_content.replace(
                        url, f"{url}\nSize: {new_size}\n"
                    )
                    with open("fileB.txt", "w") as fileB:
                        fileB.write(updated_content)
                    break
                retries += 1
