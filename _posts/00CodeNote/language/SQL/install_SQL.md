# SQL 安装问题

## Mac安装mysql问题之-bash: mysql: command not found
mysql -u root -p
-bash: mysql: command not found

解决方法：

* 在你的Mac终端,输入： `cd ~` //进入~文件夹
* 然后输入：`touch .bash_profile`
* 回车执行后，
* 再输入：`open -e .bash_profile`
* 这时候会出现一个TextEdit，如果以前没有配置过环境变量，呈现在你眼前的就是一个空白文档，你需要在这个空白文档里输入：`export PATH=$PATH:/usr/local/mysql/bin`
* 然后关闭这个TextEdit
* 回到终端面板，输入：`source ~/.bash_profile`

以上，问题解决

再输入：mysql -u root -p
回车后就会显示：Enter password:
正确输入你的密码
