---
title: Using Bison C++ API With Hand-written Scanner
date: 2020-10-24 22:00:00 +/-0800
categories: [Compilers, Bison]
tags: [Cpp, Bison]
---
In this post, I'll talk about how can you use Bison's C++ API. Take a look at the [tutorial I wrote about using C API](https://thesharpowl.github.io/posts/HAND_WRITTEN_SCANNER_WITH_BISON_C_API/) especially the parts about installing Bison latest version and Compiler Construction. Also, to gain more knowledge about how Bison works.<br>
<br>
1. Well, as I said in the C API post, you can't use dynamic types such as ``std::string`` or ``std::vector`` as Bison types which is one of the reasons of using C++ API (The main reason is that C++ is cooler of course !). 

    Again this tutorial is not focusing on Bison itself, just the C++ API part. 

2. We have a simple input file in an imperative language :<br>
```
    var some_identifer is integer
```
<br>
3. Now let's write a simple Scanner in C++ (same idea as the C one) :<br>

```
#include<iostream>
#include<fstream>
#include<string>

// I know that global variables are often bad. Forgive me I wanna just explain the idea ((
std::ifstream fin;

std::string get_next_token()
{
    std::string s;
    char c;

    while (true)
    {
        // could be done better but organized it like this to edit only assignments and returns when using Bison API
        if (!fin.eof())
            fin.get(c);// get one character
        else
            return "";

        if (c == ' ' || c == '\n' || fin.eof())
        {
            if (s.empty()) // we only have this character
            {
                if (fin.eof())
                    return "";
                //otherwise go and see what's next
                return get_next_token();
            }
            else
            {
                if (!fin.eof())
                {
                   // now we need to put the pointer exactly after the word (undo the last step)
                   // NOTE : don't use unget if you reach the end of the file because it will clear eof bit and bad stuff will happen !!!
                   fin.unget();
                }

                if (s == "var") // the last word is var
                    return s;
                if (s == "integer")
                    return s;
                if (s == "is")
                    return s;
                if (!s.empty())// it means it is some identifier name
                    return  "Identifier";
            }
        }
        else if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_') // reading some name
        {
            s += c; // add the char to the string
        }
        else
        {
            // we don't know what's that
            return "ERROR";
        }
    }
}
int main()
{

    fin.open("input.txt");
    std::string temp = get_next_token();
    while (!temp.empty())
    {
        // printing a space since our example is only one line for now
        std::cout << temp << " ";
        temp = get_next_token();

    }
    return 0;
}
```

4. Now the Bison part. (not going to go deep into grammar rules or the way you design them. I learned grammar in Theoretical Computer Science course) but I will talk about the programming aspects.<br>
    * ####  **Save the Bison file with .ypp extenstion to make it work properly with a C++ compiler**
    * Bison file consists of 4 parts :

        I. Defines and requirements (statementsstart    with ``%`` usually).

        II. C/С++ code part(s) for includes and signatures   (will be at the beginning of the generated .h  file)

        III. Grammar part

        IV. Function defintions' part
### **Note that in Bison's C++ API, a class ``Parser`` is generated unlike in the C API were we have only functions.**<br>
5. Take a look at this Bison with C++ example (I will explain after that) :

```
%require "3.2"
%language "C++"
%define api.value.type variant
%define api.token.constructor
%define parse.assert

%code requires
{
    #pragma once
    #include <iostream>
    #include <string>
    
    // forward decleration (1)
    namespace yy
    {
        class parser;
    }
}

%code
{    
    namespace yy
    {
        parser::symbol_type yylex(); // override (2)
        // no need to override the error function because it is already waiting for our implementation (3)
    }
}

%token IDENT VAR INT IS
%type <std::string> IDENT

%%

Program:
|Program VariableDeclaration
;

VariableDeclaration: VAR IDENT IS INT { /* no actions for now (4) */ }
;

%%
namespace yy
{
    parser::symbol_type yylex()
    {
        return  yy::parser::make_YYEOF ();
    }
    void parser::error(const std::string& msg) //(3+)
    {
        std::cout<< "syntax error!\n";
    }
}

int main()
{
    yy::parser p; // (5)
    // will be deleted later just make sure it prints it
    std::cout << "hello\n" << std::endl;
    return 0;
}
```

1. First, we are using Bison macros (not sure of their official name) which have ``%`` at the beginning of them:
    
    * Adding a condition about the least version which can be used with the used programming language.
    * **Note: As the documentation says, C++ Bison API is pure always so no need to add ``%define api.pure full`` as we did in the C API.**
    * ``%define api.value.type variant`` : As we know in C++, we don't use ``unions`` (you can but the standard doesn't recommend it usually. You can check [this](https://en.cppreference.com/w/cpp/utility/variant) and [this question](https://stackoverflow.com/questions/42082328/where-to-use-stdvariant-over-union)). So now you can directly just write the type of each token without the need to define a union with field names as we did in the C API.
    * ``%define api.token.constructor``: this one will generate functions in our .hpp file for each token which has a type. For example ``%type <std::string> IDENT`` generates a constructor : ``make_IDENT(std::string)`` (also another one ``make_IDENT(std::string&)``). As a result, we can use it in our scanner later, it will put the string in the parameter in the value which corresponds to our ``IDENT`` token and all we have to do is ``return make_IDENT(string_variable)`` (in the C API we had to do ``lval->s = some_char_array`` where s is the name of the char array field in our union and then return the token).

    * ``%define parse.assert`` seems to help us with useful error messages and warnings according to [this doc page](https://www.gnu.org/software/bison/manual/html_node/_0025define-Summary.html) :

        > Directive: ``%define parse.assert``
        >   * Languages(s): C, C++ <br>
        >   * Purpose: Issue runtime assertions to   catch invalid uses. In C, some important  invariants in the implementation of the  parser are checked when this option is   enabled.<br><br>
        >   * In C++, when variants are used (see    section C++ Variants), symbols must be     constructed and destroyed properly. This    option checks these constraints using  runtime type information (RTTI). Therefore   the generated code cannot be compiled with    RTTI disabled (via compiler options such as    -fno-rtti).<br>
        >   * Accepted Values: Boolean <br>
        >   * Default Value: false


2. ``%code requires`` block which contains the required things which must be added at the beginning of our .hpp file. For example, we need ``<string>`` because in the header file, Bison wants to use it for making the ``IDENT`` constructor. (I don't know if I should add ``#pragma once`` or not but just in case).
**Note in (1) we did forward declreation because we want to override ``yylex()`` (as you will see next) which belongs to the class ``parser`` while the class definition will be after the overriding. For more info about Forward Decleration [read this](https://stackoverflow.com/questions/4757565/what-are-forward-declarations-in-c)**.

3. **We defined ``yylex()`` (which belongs to the yy namespace in the C++ API) inside the ``%code requires`` block to make sure that the definition will be before any generated code.** (Also, remember that to override a function inside a name space in C++, you have to put it inside the namespace like I did. aka **defining it like : ``yy::yylex()`` won't work**).<br>
**Note: in (2) we wrote ``parser::`` because ``yylex`` function belongs to the class ``parser`` (not a namespace!!)**
4. As you can remember from the C API, we have ``yyerror`` function but **we don't have to rewrite the signature of it because it will be generated anyway and we just have to write the impelementation as said in (3)**.
5. Now we have to define the tokens and using ``%token`` and rewrite the names of the ones which have a type to specify the type to Bison using ``%type <some_type_name>``. Now as we said in the first point, there will be a constructor for each type with a prameter of its type. (we will use this soon don't worry).
6. In the grammar rules section, I declared an empty rule for ``Program`` so that the parser will have a starting point **(Note that always there should be such rule, otherwise the parser mostly will return a syntax error and won't run from the first place)**
<br>
<br>
Now inside ``Parser.tab.hpp`` there's :
```
static symbol_type make_IDENT (const std::string& v)
{
  return symbol_type (token::IDENT, v);
}
```

* This is a generated function which returns a type called ``symbol_type`` which Bison understands and supports for applying its grammar but as you can see, we should give it a string as a parameter (which is passed by a const refrernce of course since it's faster). Also, notice that it sends something else in addition to the string:
``token::IDENT`` where ``token`` is an enum and ``IDENT`` is one of the values which that enum can take.

* There are similar functions and enum values for the other tokens `` VAR, INT, IS`` but for example we have:
``` 
static symbol_type make_VAR ()
{
  return symbol_type (token::VAR);
}
```
* We don't have a parameter and it only keeps the enum inside the symoble info (calling ``symbol_type`` constructor with 1 parameter) because we don't have a type attached to this token so that's all Bison needs in this case.

Now it's time to integrate our Bison parser with our C++ scanner using the make_ methods in addition to using the correct way to define and override the parser class function.

The scanner will have this code : (comments might be repeated for the final version and for better understanding)

```
#include<iostream>
#include<fstream>
#include<string>
#include "parser.tab.hpp"
// I know that global variables are often bad. Forgive me I wanna just explain the idea ((
std::ifstream fin;

yy::parser::symbol_type get_next_token()
{
    std::string s;
    char c;
    while (true)
    {
        // could be done better but organized it like this to edit only assignments and returns when using Bison API
        if (!fin.eof())
            fin.get(c);// get one character
        else // return the end of the file so the parser will stop
            return yy::parser::make_YYEOF();

        if (c == ' ' || c == '\n' || fin.eof())
        {
            if (s.empty()) // we only have this character
            {
                if (fin.eof())
                    return yy::parser::make_YYEOF();
                    
                //otherwise go and see what's next
                return get_next_token();
            }
            else
            {
                if (!fin.eof())
                {
                   // now we need to put the pointer exactly after the word (undo the last step)
                   fin.unget();
                }

                if (s == "var") // the last word is var
                    return yy::parser::make_VAR();
                if (s == "integer")
                    return yy::parser::make_INT();
                if (s == "is")
                   return yy::parser::make_IS();
                if (!s.empty())// it means it is some identifier name 
                   return yy::parser::make_IDENT(s); // don't forget to pass the identifier name stored in the string
            }
        }
        else if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_') // reading some name
        {
            s += c; // add the char to the string
        }
        else
        {
            // we don't know what's that so return undefined token
            return yy::parser::make_YYUNDEF();
        }
    }
}
int main()
{
    fin.open("input.txt");
    yy::parser p;
	p.parse();
    return 0;
}

namespace yy
{
    parser::symbol_type yylex()
    {
        return get_next_token();
    }
}
```


and for the parser code :

```
%require "3.2"
%language "C++"
%define api.value.type variant
%define api.token.constructor
%define parse.assert

%code requires
{
    #pragma once
    #include <iostream>
    #include <string>
    
    // forward decleration (1)
    namespace yy
    {
        class parser;
    }
}

%code
{    
    namespace yy
    {
        parser::symbol_type yylex(); // override (2)
        // no need to override the error function because it is already waiting for our implementation (3)
    }
    /* 
    because this function is in the main cpp file, we have to tell the compiler that its definition is outside so that Bison won't also generate an implementation by itself.
    */
    extern yy::parser::symbol_type get_next_token();
}

%token IDENT VAR INT IS
%type <std::string> IDENT

%%

Program:
|Program VariableDeclaration
;

VariableDeclaration: VAR IDENT IS INT { std::cout << "defined an int variable " << $2 << "\n"; /* now we will print what we have */ }
;

%%
namespace yy
{
    void parser::error(const std::string& msg) //(3+)
    {
        std::cout<< "syntax error!\n";
    }
}
```

You can find all the final code examples [here](https://github.com/TheSharpOwl/Bison_API_Tutorial/tree/main/Cpp).

**Further Reading:**<br>
1. [Bison Documentation](https://www.gnu.org/software/bison/manual/bison.html).
2. [My parser for an imperative programming language](https://github.com/TheSharpOwl/FoobarCompiler/tree/061d6f544a72dcd4acd509b3b933999f4f63a5d6/compiler) (it is C++ Scanner using Bison's C API but it would be useful to read).
3. [Flex & Bison: Text Processing Tools 1st Edition book
by John Levine](https://www.amazon.com/flex-bison-Text-Processing-Tools/dp/0596155972).