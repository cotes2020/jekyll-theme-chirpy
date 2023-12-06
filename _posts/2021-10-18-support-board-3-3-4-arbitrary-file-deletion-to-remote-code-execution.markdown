---
layout:	post
title:	"Support Board 3.3.4 Arbitrary File Deletion to Remote Code Execution"
date:	2021-10-18
medium_url: https://noob3xploiter.medium.com/support-board-3-3-4-arbitrary-file-deletion-to-remote-code-execution-da4c45b45c83
categories: [Hacking, Code Review]
tags: [Hacking, Wordpress, Code Review, Web]
---


  Hi. In this writeup, i will show you a bug that i found. Allowing an Authenticated user to delete any file in the system in the Support Board 3.3.4 and also will show you a possible exploit scenario with it. Even though this is an authenticated bug, there is no csrf protection on it so we can chain it with csrf.

While reversing the functions, i found a function called delete-file

![](/img/1*LjVNhye2PiZY5VZO6i_RPQ.png)

It accepts the post parameter path as an argument.

![](/img/1*h5tHgXyz8xok-DygeznovA.png)

You can see that if the path parameter has http on it, it will delete the file in the SB\_PATH\uploads . This implementation is pretty safe since we can only delete files in the /uploads directory which doesnt have that much impact

![](/img/1*g_MdK5uwV53Cg-ehdvFw9g.png)However, if you look at the code again.

![](/img/1*WA9l7oi-yT2L6TMkpFgbbg.png)

It will only delete files in the /uploads directory if the path contains the string http. If not, it will directly use our input in $path into the unlink. So if we supplied a file without the http, it will get deleted. For reproduction purpose, i made a file called test.txt in `C:\xampp\htdocs\wordpress\wp-content`

![](/img/1*eMS_pBtM2f6ejJHAQ82hPA.png)Now back to burpsuite, if we supply the location of this file, this file should be deleted.

![](/img/1*KCyx9XxO4XdeHjdSmVTWZg.png)

We can see that it succeed and our file is now deleted

![](/img/1*_N8j0J6MurQNvy0s2cKRQA.png)

So we got a bug now. Now like i said earlier, i will show you a possible exploit scenario with this. In a wordpress installation, there is a file called `wp-config.php` . It contains the config of a wordpress installation. When we delete this file, we can then reinstall the whole wordpress application. So lets try it out. Normally, this `wp-config.php` is commonly found in the root directory of wordpress. So i tried deleting it and it succeed

![](/img/1*ysaQqRFOHqckX1hZnOu16w.png)Now if we go to the site. We can reinstall the wordpress installation. In there, we can easily achieve rce

![](/img/1*7qnTFndidibiN2_GeWeL-g.png)This is the end of my writeup, thanks for reading.

**Update:**

It is fixed now in the version 3.3.6.

![](/img/1*CLs6APmDMLk8gd5oMY6YoQ.png)

Thanks for reading

  