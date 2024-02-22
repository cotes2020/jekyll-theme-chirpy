---
published: true
date: 2023-10-21
title: Setup Deploy key for Github repository on hosting
---
1.  Generate OpenSSH here: [https://8gwifi.org/sshfunctions.jsp](https://8gwifi.org/sshfunctions.jsp)
    
2.  Add public key to **Deploy keys.**
    
3.  Add private key to `~/.ssh/` on hosting.
    
4.  Setup `~/.ssh/config` on hosting:
    

    Host github.com-<repo>
        Hostname github.com
        IdentityFile=~/.ssh/<private-key>

5.  And finally you can clone repo with SSH.
    

    git clone git@github.com<repo>:<username>/<repo>.git

5.1. If you got key permission errors

    chmod 700 ~/.ssh
    chmod 600 ~/.ssh/*
    

Sources:

*   [https://docs.github.com/en/authentication/connecting-to-github-with-ssh/managing-deploy-keys](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/managing-deploy-keys)