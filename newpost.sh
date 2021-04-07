#!/bin/bash
echo "--- Jekyll 创建博文使用指南 ---"

echo -n "请输入想要新建的博文名(注意不要有空格): -> "
read newpost 

file_name="$newpost"
timestamp_hms=$(date "+%H-%M-%S")
timestamp=$(date "+%Y-%m-%d")

if [ -f "_posts/"$timestamp-$file_name".md" ];then
  echo ">>> 该文件存在!"
  while :
	do
	read -r -p "是否覆盖文件? [Y/n] (注意：覆盖文件时，对应的图片文件夹也将覆盖): " input

	case $input in
	    [yY][eE][sS]|[yY])
			echo "确定删除文件 ..."
			rm -rf _posts/"$timestamp-$file_name".md
			echo "重新写入文件 ..."
			echo "---
title: 
author: jiap
date: $timestamp $timestamp_hms
categories: []
tags: []
---

<img src='/assets/img/$timestamp-$file_name/xx.png' style='zoom:40%; margin: 0 auto; display: block;'/>"  > "_posts/"$timestamp-$file_name".md"
			echo $timestamp-$file_name".md 文件已覆盖"
			echo "删除文件夹 ..."	
			rm -rf pics/$timestamp-$file_name/
			echo "创建 "$timestamp-$file_name "文件夹 ..."	
			mkdir assets/img/$timestamp-$file_name/
			break
			;;

	    [nN][oO]|[nN])
			echo "不覆盖文件 ..."
			echo "创建名为"$timestamp-$file_name-$timestamp_hms".md 的文件" 
			echo "---
title: 
author: jiap
date: $timestamp $timestamp_hms
categories: []
tags: []
---

<img src='/assets/img/$timestamp-$file_name-$timestamp_hms/[Replace Pic Name]' style='zoom:40%; margin: 0 auto; display: block;'/>"  > "_posts/"$timestamp-$file_name-$timestamp_hms".md"
			echo $timestamp-$file_name-$timestamp_hms".md 文件已生成"	
			echo "创建 "$timestamp-$file_name-$timestamp_hms "文件夹 ..."
			mkdir assets/img/$timestamp-$file_name-$timestamp_hms/
			break
	       	;;

	    *)
			echo "Invalid input..."
			;;
	esac
	done

  else
  # echo "不存在"
  echo ">>> 创建新的博文 Blog ..."
  echo "---
title: 
author: jiap
date: $timestamp $timestamp_hms
categories: []
tags: []
---

<img src='/assets/img/$timestamp-$file_name/[Replace Pic Name]' style='zoom:40%; margin: 0 auto; display: block;'/>"  > "_posts/"$timestamp-$file_name".md"
  
  echo "新博文"$timestamp-$file_name".md 创建完成！"
  echo "在 /assets/img 文件夹下创建 "$timestamp-$file_name" 用来存放图片"
  mkdir assets/img/$timestamp-$file_name/
fi

echo "--------"
echo "Finished"



