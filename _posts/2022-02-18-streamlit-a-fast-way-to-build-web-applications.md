---
title: Streamlit - A Fast Way to Build Web Applications
author: Tai Le
date: 2022-02-18
tags: [Front-end Engineering, Python]
---


After I switched to the NLP team in my company, I have had many opportunities to research new technologies, both in Web and AI. My leader introduced Streamlit as a Front-end development library, coding entirely in Python, to build a demo for AI's API. After using it for several days, I decided to write a post to summarize the library, list the advantages as well as disadvantages for easily getting back in the future.


## 1. Introduction

Streamlit is a relatively handy library for developers and data scientists to build web applications without prior knowledge about Web Development. Developers and Data Scientists might not need to know HTML, CSS, or JS to build a website. The library has supported plenty of components, and the way to use all of them is simply by calling Python functions.

What I am writing out is quite hard to believe, so I will walk you through some examples using Streamlit. Let's go.


## 2. Components and Mechanisms

#### a. Install the library

```
# if you're using Anaconda, please install Protobuf first
conda install protobuf

# install the library
pip install streamlit
```

#### b. Components

The main components of Streamlit are:
- Data display (text, table, JSON, Markdown)
- Chart (pie chart, bar-chart, ...)
- Form (text, number, datetime, and other inputs)
- Layout
- Media
- Status

Below is an information form with just a few lines of code:

```python
import streamlit as st

with st.form(key="my_form"):
    st.header("Personal Information")
    full_name = st.text_input("Fullname")
    age = st.slider("Age", min_value=10, max_value=100)
    gender = st.radio("Gender", ("Male", "Female"))
    introduction = st.text_area("Introduction")
    is_submit = st.form_submit_button("Submit")

    if is_submit:
        st.subheader("Result")
        st.write(f"**Fullname**: {full_name}")
        st.write(f"**Age**: {age}")
        st.write(f"**Gender**: {gender}")
        st.write(f"**Introduction**: {introduction}")
```
![/assets/img/2022-02-18/demo-form.png](/assets/img/2022-02-18/demo-form.png)


And here is a visualization for [data](https://www.kaggle.com/johnharshith/hollywood-theatrical-market-synopsis-1995-to-2021?select=HighestGrossers.csv) downloaded from Kaggle.

```python
import streamlit as st
import pandas as pd

df = pd.read_csv("HighestGrossers.csv")

st.header("Highest Grossers")
st.dataframe(df[["YEAR", "MOVIE", "DISTRIBUTOR", "TOTAL IN 2019 DOLLARS"]])
```
![/assets/img/2022-02-18/demo-table.png](/assets/img/2022-02-18/demo-table.png)

Quite simple, isn't it? There are more complex products made from this library as well, such as [Object Labeling Tool](https://github.com/streamlit/demo-self-driving) for Self-Driving Car applications. With these components, you can build various tools without knowing HTML, CSS, and JS.


#### c. Mechanism:

Just like static HTML pages, applications built with Streamlit are stateless, which means data is not shared between actions. Furthermore, in Streamlit, any time something must be updated on the screen, the Python script is re-run entirely. Therefore, in many cases, an event like form submission can cause the disappearance of data on the same page.

To us, stateless applications would be pointless unless we are building a landing page. Streamlit developers knew this problem, they have supported Session as well as In-Memory Cache to build stateful applications and reduce workload.

In a simple way, Session allows states to be stored in memory and they can be retrieved at any time (even after many re-runs). And cache helps store the result of a function using the arguments as the key (hash), so we need not repeatedly do the same action when the application is re-run.

For more information, you can check out the document for [Session](https://docs.streamlit.io/library/advanced-features/session-state) and [Cache](https://docs.streamlit.io/library/advanced-features/caching).


#### d. Underlying process:

To my knowledge, Streamlit server-side is served via a web framework called [Tonardo](https://www.tornadoweb.org/en/stable/), client-side is built with React. On top of that, the communication process between them is not via REST API, but via Protocol Buffers. So when you submit a form, please don't wonder why it still works even when no API has been called (except SegmentIO API for tracking actions).


## 3. Personal Thoughts

IMHO, this library is out of the box for Data Scientists to visualize their data and models' results. Furthermore, they can make some running demos with less effort and spend the rest of their cognitive energy researching.

But in some ways, Software Engineers might find it inconvenient because it is extremely hard to customize the CSS, JS, and add extra components. New components built are also separated in different iframes. Once I got frustrated when I tried to add a pop-up to Streamlit, then gave up and found another approach to solve the same problem. Adding a class to the HTML might sound easy at first but it's impossible.

So if you want to build a small application using the supported components without customization in style, Streamlit is a great library to use. Otherwise, you should find other libraries code or delve into the Front-end world if possible.
