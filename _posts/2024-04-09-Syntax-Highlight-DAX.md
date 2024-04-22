---
title: DAX Highlighting in Github Pages
description: Examples of text, typography, math equations, diagrams, flowcharts, pictures, videos, and more.
author: duddy
date: 2024-04-09 20:33:00 +0000
categories: [Syntax Highlighting, DAX]
tags: [rogue, highlight.js, github pages, dax, syntax highlighting]
pin: false
image:
  path: /assets/img/syntaxHighlightDAX/daxSyntaxHighlightingRackupDemo.png
  alt: Example of Dax Syntax Highlighting with Rogue
---

This blog is hosted on Github Pages. Github pages uses [Jekyll](https://jekyllrb.com/) to allow you to create static webpages, with the blogs post written with Markdown. In the future I will include DAX code for Power BI context, and I want to have Syntax Highlighting. 

## Syntax Highlighting
In markdown, Fenced Code Blocks allow cause text to render it inside a box, and are created by placing triple backticks ```` ``` ````  before and after a code snippet.

````
```
Some Code
```
````

If you add a language identifier after the opening backticks 
```` ```sql ```` you can apply Syntax Highlighting.

```sql
SELECT *
FROM tbl
WHERE foo = 'bar'
```
 
## Tokenization
In order to colour and style to the text, we first need to parse the  raw text into a list of token. When we think of a programming language we have various different elements (keywords, operators, data types, comments etc) which need to be identified and assigned a specific token type. A Lexer consists of list of rules, that provide the patterns describing the structure of specific token types. For example a single line comment in SQL starts with two dashes ```` -- ```` followed by any length of text.

```sql
-- a single-line SQL comment
```

The pattern of this token type (comment) can be described via a regex expression ```%r/--.*/```. A theme can then be applied to colour and style tokens according to their token type.

![Syntax Highlighting Process](/assets/img/syntaxHighlightDAX/Process.png)

## DAX Lexer
Jekyll's default Syntax Highlighter is [Rogue](https://rouge.jneen.net/), but this doesn't have a DAX lexer, so we'll have to develop one. Helpfully, Rogue provides a [Lexer Development Guide](https://rouge-ruby.github.io/docs/file.LexerDevelopment.html). 

### Rogue Development Environment
Rogue is a Ruby application, and therefore development and testing of the lexer requires a Linux [Development Environment](https://rouge-ruby.github.io/docs/file.DevEnvironment.html) running Ruby with the required gems. To keep the development environment self-contained, Rogue suggests using [Docker Development Environment](https://rouge-ruby.github.io/docs/file.Docker.html). 
- [Install Docker desktop](https://docs.docker.com/desktop/install/windows-install/)  
- *If you are on Windows, setup Windows Subsystem for Linux (WSL)*
- Fork the Rogue repo
- Mount the local windows directory that contains the repo to your Linux terminal

> You can run Ruby locally on WSL but I would advise against it. I ran into issues with the mapping of folder/file permission between Windows and Linux when installing gems.
{: .prompt-warning } 

Rogue suggests mounting a local directory to store dependences, so they don't have to be re-downloaded everytime a new container is created.

```bash
#Install gemfile dependences to /tmp/vendor with bundler
docker run -t -v $PWD:/app -v /tmp/vendor:/vendor -w /app -e BUNDLE_PATH=/vendor ruby bundle
```

Rogue offers a automated tested suite (rake) and a visual testing website (rackup).

```bash
#Run test suite
docker run -t -v $PWD:/app -v /tmp/vendor:/vendor -w /app -e BUNDLE_PATH=/vendor ruby bundle exec rake
```

```bash
#Run a web app on localhost with highlighted code snippets http://localhost:9292
docker run -t -v $PWD:/app -v /tmp/vendor:/vendor -w /app -e BUNDLE_PATH=/vendor -p 9292:9292 ruby bundle exec rackup --host 0.0.0.0
```

### DAX Lexer Development
Now we have our development environment setup we can start developing our DAX lexer.  If found a few existing DAX lexer in other framworks that I used for inspiration ([Tabuar Editor](https://github.com/TabularEditor/TabularEditor/blob/master/AntlrGrammars/DAXLexer.g4), SQLBI, and Microsoft learn). Resulting in a [Rogue DAX Lexer](https://github.com/EvaluationContext/rouge/blob/feature.dax/lib/rouge/lexers/dax.rb). Once we provide some code snippets ([Demo](https://github.com/EvaluationContext/rouge/blob/feature.dax/lib/rouge/demos/dax) & [Sample](https://github.com/EvaluationContext/rouge/blob/feature.dax/spec/visual/samples/dax)) we are ready to perform a visual check of the lexer.

> If you have the local website running on rackup. Any changes saved to your files are reflected on the web page, without having to restart the server; just refresh your browser.
{: .prompt-tip }

![Rogue: Testing](/assets/img/syntaxHighlightDAX/daxSyntaxHighlightingRackupSample.png)

Once we are happy with our visual inspection we can perform automated testing on our code with rake. Once this passes we are ready to push to our fork and submit a Pull Request (PR) to get our files added to the main Rogue repo. 

At the time of publishing, this PR is still under review. So in the meantime I turned to [highlight.js](https://highlightjs.org/) which allows you to fork and selfhost a languages. Like Rogue, highlighter.js provides some guides on [contributing](https://github.com/highlightjs/highlight.js/blob/main/CONTRIBUTING.md) and how to setup a [Docker Development Environment](https://highlightjs.readthedocs.io/en/latest/building-testing.html#building-and-testing-with-docker). To translate our lexer we can reference the [Language Defination Guide](https://highlightjs.readthedocs.io/en/latest/language-guide.html). Unlike Rogue, instead of saving files to your local machine and mounting the directory to the container, highlight.js creates a self contained build.

> I had to update the base docker image from node:12-slim to node:21-bullseye-slim for the container to build sucessfully
{: .prompt-info }


```bash
#### Create Docker build
docker build -t highlight-js .
```

```bash
# Run a web app on localhost with highlighted code snippets http://127.0.0.1/tools/developer.html
docker run -d --name highlight-js --volume $PWD/src:/var/www/html/src --rm -p 80:80 highlight-js
```

```bash
# Rebuilds based on local changes
docker exec highlight-js node tools/build.js -n dax
```

Highlighter.js gives us a great development environment. We provide sample code, and the webpage show how our text is being tokenized and styled.

![HighlightJS: Testing](/assets/img/syntaxHighlightDAX/daxSyntaxHighlightingHighlighterJSDemo.png)

There are some extra steps to for [Language Contrubition](https://github.com/highlightjs/highlight.js/blob/main/extra/3RD_PARTY_QUICK_START.md) which I decided to skip at this point, as ideally I want to use Rogue in the longer term.


##TODO how to get hosted by highlighter.js
#TODO host highlighter.js in github page


And NOW we are finally able to apply DAX highlighting with ```` ```dax ````!

```dax
DEFINE MEASURE 'foo'[measure1] = 
    VAR variable = 4.5
    RETURN
    SUMX( VALUES( 'bar'[col] ), 'bar'[col] + variable ) 

EVALUATE
    ADDCOLUMNS(
        VALUES( 'Date'[Year Month] )
        ,"@sumBar"
        ,CALCULATE(
            [measure1]
            ,REMOVEFILTERS()
            ,USERELATIONSHIP( 'Date'[Date], 'bar'[Order Date] )
        )
    )
```

As an end note, in addition to the more staticly defined token, DAX also has the concept of variables. To highlight these would require Semantic Highlighting, which is beyond the scope of this project.
