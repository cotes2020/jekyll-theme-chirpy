---
published: true
date: 2023-05-05
title: Forward request from a path to specific port by .htaccess
---
    Options +FollowSymLinks -Indexes
        IndexIgnore *
        DirectoryIndex
        <IfModule mod_rewrite.c>
        RewriteEngine on
        RewriteRule ^(.*)$ http://localhost:<internal-port>/$1 [P]
        </IfModule>