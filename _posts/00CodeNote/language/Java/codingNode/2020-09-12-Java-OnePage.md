---
title: Java - Specific Classes / API
date: 2020-09-12 11:11:11 -0400
description:
categories: [00CodeNote, JavaNote]
tags: [Java]
toc: true
---

[toc]

---

# Java - Specific Classes / API

---

## Specific Classes / API

---

### `FileResource`: accessing file on computer.

```java
create a FileResource:

new FileResource() : opens a dialog box prompting you to select a file on your computer
new FileResource("path/to/file.ext") : find a file on computer or within your BlueJ project
new FileResource(existingFile) : uses the given File (typically returned by using a DirectoryResource)


Method

.lines()
// provides access to the contents of this opened file one line at a time
for (String line : fr.lines()) { }

.words()
// provides access to the contents of this opened file one word at a time
for (String word : fr.words()) { }

.asString()
// returns the entire contents of this opened file as one String
String contents = fr.asString();
```

---

### `URLResource`: accessing a web page.

```java

create a URLResource by giving it a complete URL, or web address

new URLResource("http://www.something.com/file.ext"),
// uses the given address to download the referenced file

new URLResource("https://www.something.com/file.ext"),
// uses the given address to download the referenced file


Method
.lines()
// provides access to the contents of this opened web page one line at a time
for (String line : ur.lines()) { }

.words()
// provides access to the contents of this opened web page one word at a time
for (String word : ur.words()) { }

.asString()
// returns the entire contents of this opened web page as one String
String contents = ur.asString();
```

---

### `DirectoryResource`: choosing one or more files on your computer.

```java
// can only create a DirectoryResource with no parameters:
new DirectoryResource()

Method

.selectedFiles()
// provides access to each of the files selected by the user one at a time
for (File f : dr.selectedFiles()) { }
```

---


### `StorageResource`: storing and accessing a list of strings of any length.

```java
// creating an empty StorageResource

new StorageResource()
// creates an empty list
new StorageResource(otherList)
// creates a list that is an exact copy of otherList



Method
.add(item)
// adds the given item to the end of the list of strings
sr.add("first!");
sr.add("next ...");

.size()
// returns the number of strings stored in this list
sr.size() // is 2 (after the example above)
sr.size() // is 0 (immediately after clear() is called)


.data()
// provides access to each string in the list one at a time
for (String item : sr.data()) { }

.contains(item)
// returns true only if the given item is in the list	sr.contains("first!") is true
sr.contains("last") // is false

.clear()
// removes all strings from this list, making it empty
sr.clear();
```

---

## StringBuilder
- string cannot change,
- StringBuilder can change,

```java
StringBuilder sb = new StringBuilder(“Hello”);

append
// Put String, int, char, etc.. on end

insert
// Insert String, int, char, etc... into middle

charAt
// Gets character at specified index

setCharAt
// Changes the character at specified index

toString
// Get back String that you made
```

---

## Character Building

```java
isLowerCase(ch)
// returns boolean if ch is 'a', 'b' …

isDigit(ch)
// returns boolean if ch is '0','1',…'9'

toLowerCase(ch)
// returns lowercase version of ch

toUpperCase(ch)
// returns uppercase version of ch
```




.
