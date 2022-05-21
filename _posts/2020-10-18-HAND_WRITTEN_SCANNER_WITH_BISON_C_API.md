---
title: Using Bison C API With Hand-written Scanner
date: 2020-10-18 12:00:00 +/-0800
categories: [Compilers, Bison]
tags: [C, Bison]
---
You may have a compilers course and wanna learn how to use Bison with your other code. When I had this course I got in a big confusion while trying to use Bison with a Hand 
Written parser (in my opinion documentation isn't easy for a bachelor and the deadline won't wait for you to read all of it). As a result, I had to ask people just to know how to make Bison use my functions instead of Flex's (most code on internet Flex is used). Well, I finished the course (with not much disaster, thank God) and wanted to write here about it !

Let's start with a brief introduction to be sure that we are on the same page. **Compiler Construction consists of the following stages:**

1. Token producer or Scanner
2. Parser
3. Parsing Tree Design (just design)
4. Construction and adding nodes to the tree
5. Semantics (make sure an expression is valid for example)
6. Machine Code generation (LLVM code for example)


**Note : In my case it was a compiled langauge so if you are doing an interpreter, some things will be different for you.**

1. Use linux or linux shell on windows because latest Bison is not supported for windows.<br>

2.  **[Install the latest version from here](https://launchpad.net/bison/head/3.7.2) not from package manager**. because sometimes it doesn't install the latest one for some reason....
3. Let's say we only have this rule/case (any variable name with letters and underscores works)

```
var some_variable is integer
```

4. A dummy Scanner just to explain the idea : 

```
#include<stdio.h>
#include<stdlib.h>
#include<string.h>

FILE* fptr;

// returns 0 when the file is completely read
int get_next_token()
{
    char s[1002], c;
    int len = 0;

    while(1)
    {
        c = getc(fptr);

        if(c == ' ' || c == '\n' || c == EOF)
        {
            if(len == 0) // we only have this character
            {
                if(c == EOF)
                    return 0;

                // we can print it
                printf("%c",c);
                return 1;
            }
            else
            {
            // add NULL termination at the end so that strcmp will know the ending of our array
             s[len] = '\0'; 

             if(strcmp(s,"var") == 0) // the last word is var
                printf("VAR");
             else if(strcmp(s,"integer") == 0)
                printf("INT");
             else if(strcmp(s,"is") == 0)
                printf("IS");
             else if(len > 0)// it means it is some identifier (if there's misplacment the grammar will discover it :D )
                printf("IDENT");
            // now we need to put the pointer exactly after the word (undo last step to reread ONLY the 'non letter or underscore')
             ungetc(c, fptr);
            //reset the char array
             memset(s, 0, sizeof(s));
             len = 0;
             return 1;
            }
        }
        else if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_') // reading some name
        {
            s[len++] = c;
        }
        else
        {
            printf("%s + Unknown symbol! ", s);
            //reset the char array
            memset(s, 0, sizeof(s));
            len = 0;
            // stop everything 
            return 0;
        }    
    }
}
int main()
{
    fptr = fopen("input.txt", "r+");

    while(get_next_token() != 0);

    printf("\n");
    
    return 0;
}
```

5. Now the Bison part. I won't go deep into grammar rules and the way you design them (I learned grammar in Theoretical Computer Science course) but I will talk about the programming aspects. <br> <br>
Bison file consists of 4 parts :
    1. Defines and requirements (statements start with ``%`` usually)
    2. C code part(s) for includes and signatures (will be at the beginning of the generated .h file)
    3. Grammar part
    4. Function defintions part
<br>

6. Let's take a look at this example and I will explain it after that :

```
%require "3.2"
%define api.pure full

%code requires
{
    // note that these are imported at the beginning of parser.tab.h in the generated C file 
    // so you might not need it (in my case I don't but just to clarify)
}
%code
{
    int yylex(YYSTYPE *lvalp);
    void yyerror(const char *error);
    // note that this is added after including parser.tab.h in parser.tab.c
    #include<stdio.h>
    #include<string.h>

    // TODO delete
    int temp = 1;
}

%token IDENT VAR INT IS


%union {
    // put all the types you want to use here with any identical name
    // then in types section put its name such as 'st' below
    char st[1002];
}

%type<st> IDENT

%%

VariableDeclaration: VAR IDENT IS INT {
    /* this is called a semantic action*/
     printf("defined a variable %s with type int\n", $2);
     }
;

%%

void yyerror(const char *error)
{
    printf("Syntax Error\n");
}

int yylex(YYSTYPE *lvalp)
{
    return YYEOF;
}
```
* The first line holds a condition of the least required version of Bison to run our file.
* The second line declares that we want to use our own scanner (pure calling). Take a look at this [link](https://www.gnu.org/software/bison/manual/html_node/Pure-Calling.html) (don't worry about understanding it 100%).
* Code-requires and Code blocks.
* Terminal tokens which don't have a rule so they can be used as the building blocks for other rules (you can put non terminal token in a rule but at the end, a non terminal token should have only terminal tokens in its grammar directly or in the grammar of its grammar non terminal tokens).

* Union, which contains a variable to each type we want to return with a unique name for each and Bison will generate them as data members to ``lvalp`` in ``yylex`` function.

* ``%type`` declares that a token can return that type. The type is named as the identifier name used in the ``%Union``. You don't have to put types to all terminal tokens and you can add a type to a non-terminal token but don't forget to put ``$$ = some_value`` in the semantic action in each rule of its grammar.
* In general, Bison has 3 important functions:

    1. ``yylex(lvalp)`` which is similar in logic to our ``get_next_token`` except that you have to return predefined token (we will see them in a little bit) and if you have a token which corresponds to a value, you have to put its value inside ``lvalp`` member. In our case, if we see an identifier, we should copy its name to ``lvalp->st`` since ``st`` is in the union.
    <br>
    For the defined tokens, if you generate .c and .h files from this bison using :

         ```Bison -d file_name.y```

        then you can see at the beginning :
        ```
        enum yytokentype
        {
        YYEMPTY = -2,
        YYEOF = 0,                     /* "end of   file"  */
        YYerror = 256,                 /* error  */
        YYUNDEF = 257,                 /* "invalid  token"  */
        IDENT = 258,                   /* IDENT  */
        VAR = 259,                     /* VAR  */
        INT = 260,                     /* INT  */
        IS = 261                       /* IS  */
        };
       ```
       Well, ``yylex(YYSTYPE lvalp)`` returns one of these enum values and knows which value to get from ``lvalp`` when we return ``IDENT`` for example because in Bison we wrote that the type of ``IDENT`` is ``st`` which is a char array.

       * **You can add more parameters using:<br>
        ```%param {name_of_parameter}```
        <br> at the beginning.** 
    2. ``yyparse()`` which keeps calling ``yylex()`` until it returns ``YYEOF``, ``YYERROR`` or ``YYUNDEF``.
       *  **You can add more parameters using:<br>
             ```%parse-param {name_of_param}```
          <br>
        at the beginning.** 
    3. ``yyerror(char* error_message)`` which gets called when  ``yylex`` returns **``YYUNDEF``**. To know how to modify the error message, [check this](https://stackoverflow.com/questions/41409423/bison-custom-syntax-error).

* Since some tokens have values (``IDENT`` in our case), we can access these value using 

* I have missed an important thing in Bison, which is not adding a rule for the empty case (because it works for this example only). Therefore, running this bison will print ``syntax error`` (supposing that you will add a main function and call ``yyparse()`` from there). So don't forget to do that when you are writing a real parser.

7. I suggest that you stop here and try to modify our ``get_next_token()`` such that we can do this in Bison :
    ```
    int yylex(YYSTYPE *lvalp)
    {
        return get_next_token(lvalp);
    }
    ```
    **Don't forget to add the line : <br>
    `` int get_next_token(YYSTYPE *lvalp);``
    <br>
    in the code block.**

    And try to make it do the job.

8. After changing ``yylex`` and adding the signature of ``get_next_token`` in the previous step, we will modify our ``scanner.cpp`` file  so that it would work with the parser:

```
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
// include parser header so we can use the enums
#include "parser.tab.h"
FILE* fptr;

// returns 0 when the file is completely read
int get_next_token(YYSTYPE *lvalp)
{
    char s[1002], c;
    int len = 0;

    while(1)
    {
        c = getc(fptr);

        if(c == ' ' || c == '\n' || c == EOF)
        {
            if(len == 0) // we only have this character
            {
                if(c == EOF)
                    return YYEOF;
                //otherwise return the next token because there's no enum for new lines or spaces
                return get_next_token(lvalp);
            }
            else
            {
            // add NULL termination at the end so that strcmp will know the ending of our array
             s[len] = '\0'; 

             if(strcmp(s,"var") == 0) // the last word is var
                return VAR;
             else if(strcmp(s,"integer") == 0)
                return INT;
             else if(strcmp(s,"is") == 0)
                return IS;
             else if(len > 0)// it means it is some identifier (if there's misplacment the grammar will discover it :D )
                {
                    // put the value and return the correct token
                    strcpy(lvalp->st, s);
                    return IDENT;
                }
            // now we need to put the pointer exactly after the word (undo last step to reread ONLY the 'non letter or underscore')
                ungetc(c, fptr);
            //reset the char array
             memset(s, 0, sizeof(s));
             len = 0;
             // go to the next one
             return get_next_token(lvalp);;
            }
        }
        else if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_') // reading some name
        {
            s[len++] = c;
        }
        else
        {
            //reset the char array
            memset(s, 0, sizeof(s));
            len = 0;
            // stop everything 
            return YYUNDEF;
        }    
    }
    return YYUNDEF;
}
int main()
{
    fptr = fopen("input.txt", "r+");

    int x = yyparse();

    printf("\n%d", x);

    return 0;
}
```

For easier compliation use a make file :

```
example: Scanner.c Parser.y 
	bison -d Parser.y
	gcc Scanner.c Parser.tab.c -o program

```

Now running ``make`` then ``./program``, gives the following output:<br>
```
defined a variable x with type int
```

You can try different variable names or mess up the code sample to get a syntax error.

I hope this was a useful tutorial and now you know how to use Bison with your hand written scanner.

You can find all the final code examples [here](https://github.com/TheSharpOwl/Bison_API_Tutorial/tree/main/C).

**Final Note:**
There's a Bison C++ API which allows you to use dynamic types such as ``std::string``. I will write about it in another post. You can use C API with C++ scanner but you have to use a pointer to the ``std::string`` and other things such as ``std::shared_pointer`` will be really difficult to use here.

**Further Reading:**<br>
1. [Bison Documentation](https://www.gnu.org/software/bison/manual/bison.html).
1. [My parser for an imperative programming language](https://github.com/TheSharpOwl/FoobarCompiler/tree/061d6f544a72dcd4acd509b3b933999f4f63a5d6/compiler) (it is C++ Scanner using Bison's C API but it would be useful to read).
2. [Flex & Bison: Text Processing Tools 1st Edition book
by John Levine](https://www.amazon.com/flex-bison-Text-Processing-Tools/dp/0596155972).