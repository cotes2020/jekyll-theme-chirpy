#!/usr/bin/env python
#%% Preliminaries.
import glob
import os

import git


#%% Define paths.
post_dir = '_posts/'
tag_dir = 'tag/'


#%%
def get_tags(post_dir=post_dir, verbose=True):
    # Get Markdown posts files.
    filenames = glob.glob(post_dir + '*md')

    # Loop through all files.
    total_tags = []
    for filename in filenames:
        f = open(filename, 'r', encoding='utf8')
        crawl = False
        tag_lines_coming = False
        for line in f:
            current_line = line.strip()
            if crawl:
                if current_line == 'tags:':
                    tag_lines_coming = True
                    continue

            # If --- delimiter is found, start crawling.
            if current_line == '---':
                if not crawl:
                    crawl = True
                else:
                    crawl = False
                    break

            # If we are in the actual tag lines (that is, tag_lines_coming is
            # True and we aren't in the tags: line), extract them.
            if tag_lines_coming and (current_line != 'tags:'):
                total_tags.append(current_line.strip('- '))
        f.close()

    # Make tags unique in a set.
    total_tags = set(total_tags)

    if verbose:
        print("Found " + str(total_tags.__len__()) + " tags")

    return total_tags


#%%
def create_tags_posts(tag_dir=tag_dir, total_tags=set(), verbose=True):
    if total_tags.__len__() == 0:
        print("No tags. Thus, no tag posts were created")
        return None

    else:

        old_tags = glob.glob(tag_dir + '*.md')
        for tag in old_tags:
            os.remove(tag)

        if not os.path.exists(tag_dir):
            os.makedirs(tag_dir)

        for tag in total_tags:
            tag_filename = tag_dir + tag + '.md'
            f = open(tag_filename, 'a')
            write_str = '---\nlayout: tag_page\ntitle: \"Tag: ' + tag + '\"\ntag: ' + tag + '\nrobots: noindex\n---\n'
            f.write(write_str)
            f.close()

        if verbose:
            print("Created " + str(total_tags.__len__()) + " tag posts")

        return None


#%%
if __name__ == '__main__':
    tags = get_tags(post_dir)
    create_tags_posts(tag_dir, tags)

    # For Git.
    repo = git.Repo(os.getcwd())

    # Add files for commit.
    try:
        repo.git.add(tag_dir)
    except:
        print("Error ocurred while adding files to Git.")

    # Commit changes.
    try:
        repo.git.commit('-m', 'Updated tags and created corresponding posts', author='deadpackets@protonmail.com')
    except:
        print("Error occurred while commiting.")

    # Push commit.
    try:
        origin = repo.remote(name='origin')
        origin.push()
    except:
        print("Error occurred while pushing.")