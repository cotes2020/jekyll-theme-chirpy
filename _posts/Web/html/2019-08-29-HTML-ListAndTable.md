---
title: HTML - List And Table
date: 2019-08-29 11:11:11 -0400
description: Learning Path
categories: [Web, HTML]
img: /assets/img/sample/rabbit.png
tags: [HTML]
---

[toc]

---

# HTML - List And Table

---

## HTML lists
- list: `<ul>`, `<ol>`, `<dl>`
- data inside: `<li>`

```html

<body>

// Unordered HTML List
<ul>
  <li>Coffee</li>
  <li>Tea</li>
  <li>Milk</li>
</ul>

// Ordered HTML List
<ol>
  <li>Coffee</li>
  <li>Tea</li>
  <li>Milk</li>
</ol>


// Description Lists
<dl>
  <dt>Coffee</dt>
  <dd>- black hot drink</dd>
  <dt>Milk</dt>
  <dd>- white cold drink</dd>
</dl>

</body>
```


---


## table

- An HTML table: `<table>` tag.
- Each table row: `<tr>` tag.
- A table header: `<th>` tag. `<thead>`
  - By default table headings are bold and centered.
- A table data/cell: `<td>` tag.

`<tbody> <tfoot>`


```html
<!DOCTYPE html>
<html>

<head>
    <style>
        table {
            font-family: arial, sans-serif;
            border-collapse: collapse;      // 只留一条边框
            border-spacing: 5px;            // 留2条边框
            width: 100%;
        }

        td, th {
            border: 1px solid #dddddd;      // Adding a Border
            text-align: left;               // 字体靠边对齐
            padding: 8px;                   // 字体和边框宽度
        }

        tr:nth-child(even) {
            background-color: #dddddd;
        }
    </style>
</head>


<body>
<h2>HTML Table</h2>

<table>
  <tr>
    <th>Company</th>
    <th>Contact</th>
    <th>Country</th>
  </tr>
  <tr>
    <td>Alfreds Futterkiste</td>
    <td>Maria Anders</td>
    <td>Germany</td>
  </tr>
  <tr>
    <td>Centro comercial Moctezuma</td>
    <td>Francisco Chang</td>
    <td>Mexico</td>
  </tr>
</table>


<table style="width:100%">
  <caption>Table name</caption>    // add a caption to a table
  <tr>
    <th>Name</th>
    <th colspan="2">Telephone</th>      // Cells that Span Many Columns
  </tr>
  <tr>
    <td>Bill Gates</td>
    <td>55577854</td>
    <td>55577855</td>
  </tr>
  <tr>
    <th rowspan="2">Telephone:</th>     // Cells that Span Many rows
  </tr>
    <td>1</td>
    <td>1.1</td>
  </tr>
  <tr>
    <td>2</td>
    <td>2.1</td>
  </tr>
</table>


<table>
    <tr> <th> AAA </th><td> EEE </td> </tr>
    <tr> <th> OOO </th><td> III </td> </tr>
</table>


</body>
</html>

```


### A Special Style for One Table
add an id attribute to the table:

```html

<head>
<style>

table {
  width:100%;
}

// 主要，可控制全部
table, th, td {
  border: 1px solid black;
  border-collapse: collapse;
  padding: 15px;
  text-align: left;
}

// 单独的其他样式
table#t01 tr:nth-child(even) {
  background-color: #eee;
}
table#t01 tr:nth-child(odd) {
 background-color: #fff;
}
table#t01 th {
  background-color: black;
  color: white;
}

</style>
</head>


<table id="t01">
  <tr>
    <th>Firstname</th>
    <th>Lastname</th>
    <th>Age</th>
  </tr>
  <tr>
    <td>Eve</td>
    <td>Jackson</td>
    <td>94</td>
  </tr>
</table>

```
