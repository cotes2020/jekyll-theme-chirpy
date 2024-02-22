---
published: true
date: 2023-05-15
title: .htaccess for “next export”
---
    <IfModule mod_rewrite.c>
        Options +FollowSymLinks -MultiViews
    
        RewriteEngine On
    
        RewriteCond %{REQUEST_FILENAME} -d
        RewriteRule ^(.*)/$ $1.html
    
        RewriteCond %{REQUEST_FILENAME}.html -f
        RewriteRule !.*\.html$ %{REQUEST_FILENAME}.html [L]
    
        RewriteCond %{REQUEST_FILENAME} !-d
        RewriteCond %{REQUEST_FILENAME}\.html -f
        RewriteRule ^(.*)$ $1.html [NC,L]
    
        # Protect some contents
        RewriteRule ^.*/?\.git+ - [F,L]
    
        RewriteCond %{ENV:HTTPS} !on
        RewriteRule ^(.*)$ https://%{HTTP_HOST}%{REQUEST_URI} [L,R=301]
    </IfModule>

Source: [https://github.com/vercel/next.js/discussions/10522#discussioncomment-3687225](https://github.com/vercel/next.js/discussions/10522#discussioncomment-3687225)