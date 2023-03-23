# Class for C+ & C

[toc]

# 编译
## 手动编译
cpp文件夹下有三个文件 `helloworld.c` `file.c` `file.h`.
如果希望编译helloworld，必须在终端执行单个命令如下：

```
gcc -c helloworld.c
gcc -c file.c
gcc -o helloworld helloworld.o file.o
```

**.c**
[clang gcc -e]
*preprocessor*: handle #include, define; strip out comments.
*compiler*: translate C to Assembly.

**.s**
[clang gcc -s]
*assembler*: translate assembly to `object file`.

**.o**
[clang gcc -c] 需要到`object file`,直接这一步
*linker*: brings together `object file` to produce `executable`.

**exec.**

`gcc -c` helloworld.c
`gcc -c` file.c
`gcc -o` **helloworld** helloworld.o file.o

## 自动编译
automating builds with makefiles:

#C

* `include`: 将一个文件的内容导入到这个文件的方式。
C具有使用`.h`扩展名作为头文件的惯例。
头文件中拥有一些函数的列表，这些都是你想在程序中使用的函数。

* 注释:
    * `/* 和 */`: 多行注释，你可以在之间放置任意多行。
    * `//`: 注释的另一种形式，它就像Python或Ruby的注释。以//开头，直到行末结束。

* `int main(int argc, char *argv[])`
    * 操作系统加载完你的程序，之后会运行叫做main的函数，这是C程序的工作方式。
    * 这个函数只需要返回int，并接受两个参数:
    * 一个是int作为命令行参数的数量，
    * 另一个是char*字符串的数组作为命令行参数。

* `{}`: 任何函数都以{字符开始，它表示“程序块”的开始。在Python中用一个`:`来表示。在其它语言中，可能需要用`begin`或者`do`来表示。

* `type name = value;` :
    * 一个变量的*声明*和同时的*赋值*。
    * 使用语法type name = value;来创建变量。
    * 在C的语句中，除了逻辑语句，都以一个;（分号）来结尾。

* `printf("XX", value);`。就像许多语言中的函数调用，使用语法`name(arg1, arg2);`。

* `return 0;`: main函数的返回语句，它会向OS提供退出值。

## 变量类型
* **整数** %d.
　　使用`int name = XX`声明，使用`%d`来打印。
* **浮点**
　　使用`float name = XX`或`double name = XX`声明，使用`%f`来打印。
　　
　　使用`long name = XX`: 来声明一个大的数值，它可以储存比较大的数。使用`%ld`打印出这个变量，我们添加了个修饰符到`%d`上面。添加的`"l"`表示将它当作长整形打印。
　　结果非常小，所以我们要使用`%e`以科学记数法的形式打印它。
　　
* **字符**
　　使用`char name = XX`来声明，以周围带有`'`（单引号）的单个字符来表示，使用`%c`来打印。
* **字符串（字符数组）**
　　使用`char name[]=XX`来声明，以周围带有`"`的一些字符来表示，使用`%s`来打印。
你会注意到C语言中区分单引号的char和双引号的char[]或字符串。

* `'\0'` 特殊的语法
    * '\0'声明了一个字符。
    * 这样创建了一个“空字节”字符，实际上是数字0。


# C++


### 计算
**逻辑运算符**
`&`    **AND**     仅当两项都为真（1）时则为 True

        ```
        0 0 1 1   運算元1
        0 1 0 1   運算元2
        ———-
        0 0 0 1   (運算元1 & 運算元2)    –    回傳結果
        ```

`|`    **OR**       只要其中一项为真时则为 True

        ```
        0 0 1 1   運算元1
        0 1 0 1   運算元2
        ———-
        0 1 1 1   (運算元1 | 運算元2)    –    回傳結果
        ```


`^`  **XOR**     如果表达式中任意一项（但不是两项）为真时则为 True EXCLUSIVE OR
     在 XOR 的運算中，如果 mask 的 bit 是 1 則會顛倒；如果是 0 則不會顛倒，保持原值。

        ```
        0 0 1 1   運算元1
        0 1 0 1   運算元2
        ———-
        0 1 1 0   (運算元1 ^ 運算元2)    –    回傳結果
        ```

`!`  **NOT**    只要其中一项为真时则为 True

```
NOT 0           1
NOT 1           0
```

NOT     将值由 False 变为 True，或者由 True 变为 False

## 循环

**循环类型**

循环类型|描述
---|---
`while` 循环|当条件为真，重复语句或语句组。它会在执行循环主体之前测试条件。
`for` 循环|多次执行一个语句序列，简化管理循环变量的代码。
`do...while` 循环|除了它是在循环主体结尾测试条件外，其他与 while 语句类似。
嵌套循环|您可以在 while、for 或 do..while 循环内使用一个或多个循环。

**循环控制语句**
循环控制语句: 更改执行的正常序列。当执行离开一个范围时，所有在该范围中创建的自动对象都会被销毁。

控制语句 | 描述
---|---
`break` 语句 | 终止 loop 或 switch 语句，程序流将继续执行紧接着 loop 或 switch 的下一条语句。
`continue` 语句 | 引起循环跳过主体的剩余部分，立即重新开始测试条件。
`goto` 语句 | 将控制转移到被标记的语句。但是不建议在程序中使用 goto 语句。

**无限循环**
如果条件永远不为假，则循环将变成无限循环。for 循环在传统意义上可用于实现无限循环。由于构成循环的三个表达式中任何一个都不是必需的，您可以将某些条件表达式留空来构成一个无限循环。

```
#include <iostream>
using namespace std;

int main ()
{

   for( ; ; )
   {
      printf("This loop will run forever.\n");
   }

   return 0;
}
```

当条件表达式不存在时，它被假设为真。您也可以设置一个初始值和增量表达式，但是一般情况下，C++ 程序员偏向于使用 for(;;) 结构来表示一个无限循环。
注意：您可以按 Ctrl + C 键终止一个无限循环。

### C++ while 循环
只要给定的条件为真，`while`会重复执行一个目标语句。

#### 语法
```
while(condition)
{
   statement(s);
}
```

**statement(s)**:
可以是一个单独的语句，也可以是几个语句组成的代码块。
**condition**:
可以是任意的表达式，当为任意非零值时都为真。
当条件为真时执行循环。
当条件为假时，程序流将继续执行紧接着循环的下一条语句。

#### 流程图
* `while`的关键点是循环可能一次都不会执行。
* 当条件被测试且结果为假时，会跳过循环主体，直接执行紧接着 while 循环的下一条语句。
实例

```
include <iostream>
using namespace std;
int main ()
{
   int a = 10;        // 局部变量声明
   while( a < 20 )    // while 循环执行
   {
       cout << "a 的值：" << a << endl;
       a++;
   }
   return 0;
}
```
当上面的代码被编译和执行时，它会产生下列结果：
```
a 的值： 10
a 的值： 11
a 的值： 12
a 的值： 13
a 的值： 14
a 的值： 15
a 的值： 16
a 的值： 17
a 的值： 18
a 的值： 19
```

## C输入输出
C++ 标准库提供了一组丰富的输入/输出功能，我们将在后续的章节进行介绍。本章将讨论 C++ 编程中最基本和最常见的 I/O 操作。
C++ 的 I/O 发生在流中，流是字节序列。如果字节流是从设备（如键盘、磁盘驱动器、网络连接等）流向内存，这叫做输入操作。如果字节流是从内存流向设备（如显示屏、打印机、磁盘驱动器、网络连接等），这叫做输出操作。

### I/O 库头文件
下列的头文件在 C++ 编程中很重要。
头文件	函数和描述
**<iostream>**	该文件定义了 cin、cout、cerr 和 clog 对象，分别对应于标准输入流、标准输出流、非缓冲标准错误流和缓冲标准错误流。
**<iomanip>**	该文件通过所谓的参数化的流操纵器（比如 setw 和 setprecision），来声明对执行标准化 I/O 有用的服务。
**<fstream>**	该文件为用户控制的文件处理声明服务。我们将在文件和流的相关章节讨论它的细节。

### 标准输出

### C：标准输出 [printf("Bobby")]
```
#include <stdio.h>
int main ()
{
    printf("I am 100 years old.\n")

    return 0;
}
```
首先你包含了另一个头文件叫做`stdio.h`。这告诉了编译器你要使用“标准的输入/输出函数”。它们之一就是`printf`。

**NAME**
     `printf` -- formatted output

**SYNOPSIS**
     `printf` format [arguments ...]

**DESCRIPTION**
     The printf utility `formats` to print its `arguments`, after the first, under control of the format.
     The format is a character  string which contains three types of objects:
     **plain characters** 普通字符串, which are simply copied to standard output, 直接拷贝到标准输出STDOUT
     **character escape sequences** 字符转义序列, which are converted and copied to the standard output, 通过字符转义之后输出到标准输出
     and **format specifications** 格式说明符, each of which causes printing of the next successive argument. 每一个格式说明符对应输出相应的argument；
   The arguments after the first are treated as strings if the corresponding format is either c, b or s; otherwise it is evaluated as a C constant, with the following extensions:
如果对应的格式指示符是%c/%b/%s时，相对应的参数都视为字符串，否则它们会被解释为C语言的数字常量: 在其开头可使用正负号标识;如果字符串开头是单引号'或者双引号",那么打印输出的值是紧跟着单引号或者双引号后的那个字符的ASCII值

* A leading plus or minus sign is allowed.
* If the leading character is a single or double quote, the value is the ASCII code of the next character.

#### 格式占位符

**格式占位符**|结果
---|---
**%c** | The first character of argument is printed.打印输出参数中的第一个字符，会忽略其它字符；
**%s** | Characters from the string argument are printed until the end is reached or until the number of characters indicated by the precision specification is reached; however if the precision is 0 or missing, all characters in the string are printed. 打印整个字符串参数，或者打印控制精度控制的字符个数，如果精度值是0或者没有控制精度值，则打印所有的`字符值`。
**%b** | As for s, but interpret character escapes in backslash notation in the string argument. printf会*解析*格式指示符和参数列表中的*转义字符*, 其后的参数列表需要用双引号. 无双引号，简单输出全部字符。

```
$ printf "hello %c, %s. \n" "China" "World"
hello C, World
//%c只头字母

$ printf "hello \t world, welcome to %s. \n" "\tChina"
hello	world, welcome to\tChina
//%s全部简单输出
//%数字s 表示空格大小

$ printf "hello %b see u\n" hjkl\taa.
hello hjkltaa see u

$ printf "hello %b see u\n" "hjkl\taa"
hello hjkl	aa see u
//%b，解析 参数中的 转义字符.

```

#### 转义字符

**转义字符**|结果
---|---
\a | 警告声音输出.
\b | 退格键<backspace>.
`\c` | 忽略该字符串中后面的任何字符(包括**普通字符**、**转义字符**以及**参数**)以及**格式字符串中的字符**，该规则只有在`%b`格式指示符控制下参数字符串中有效
`\f `| 换页<form-feed>. 按前进位置直接切行。
`\n` |输出新的一行.
\r | Enter键，<carriage return> character.
`\t` | 水平制表符[tab].
\v | 垂直制表符.
\' | 单引号.
\\ | 反斜线.

```
$ printf "ab %s c %b def %s" "AAA\n" "JQK\cioio" "China"
ab(AAA\n)c(JQK)
```
//忽略了该参数后的**字符**`ioio`，
//也忽略了**指示符**`%b`后所有格式说明，包括**普通字符**`def`以及**格式占位符**`%s`.

```
$ printf "hello a \f bbbbbkkdkkdk \f lodldidkcmmjjd \n"
hello a
       bbbbbkkdkkdk
                   lodldidkcmmjjd
```

#### 格式说明符

1. 0个或者多个标志位，这里暂且称为位置标识
2. Field Width, 字段宽度。
3. 精度控制
4. 参数格式类型

##### 位置标识
可以直接打在terminal里面
位置标识|结果
---|---
**#**|该字符指示参数应该进行格式转换。对于`%c/%d/%s`格式类型，该选项不产生影响；对于`%o`强制其精度值增加，使其输出字符串的第一个字符为0；对于`%x/%X`,会有0x(0X)加在参数开头; 对于`%e/%E/%f/%g/%G`,在输出结果中总会包含有小数点，即使小数点后没有任何数字.
**-**|该字符指定对应的参数字段使用左对齐方式.
**+**|该字符指定，在输出有符号形式的数字时，在数字前总应该有相应符号.
`.`|空格符，该符号指示在输出有符号形式的数字时，如果该数字是正数，则在该正数前应该有一空格，如果是负数，则在空格位置是-；如果使用了+`位置标识，则该位置标识不起作用.
**0**|该指示符表示，应该使用0填充代替空格填充，如果使用-位置标识，则该位置标识不起作用.

```
$ printf "hello %#s. \n" "world, hey"
hello world, hey.
//#对%s格式类型无影响

$ printf "this is octonary number: %o. \n" "123"
this is octonary number: 173.
$ printf "this is octonary number: %#o. \n" "123"
this is octonary number: 0173.

$ printf "this is octonary number: %x \n" "123"
this is octonary number: 7b
$ printf "this is octonary number: %#x \n" "123"
this is octonary number: 0x7b
//八进制和十六进制数字，使用#后会在参数前添加0或者0x


$ printf "Field width: |%10s| \t |%-10s| \n" "hello" "world"
Field width: |     hello|	|world     |
//-指示符之后，参数实行左对齐
//%数字s 表示空格大小


$ printf "signed number: %+d \n" 3456
signed number: +3456
$ printf "signed number: %+d \n" -3456
signed number: -3456
//使用+指示符之后，输出有符号数字时会在数字前加上+/-

$ printf "padding number:% d\n" -3456
padding number:-3456
$ printf "padding number:%+ d\n" -3456
padding number:-3456
//如果输出负数时同时使用了+指示符，则空格指示符不起作用

$ printf "padding number:|%5d|\n" 23
padding number:|   23|
$ printf "padding number:|%05d|\n" 23
padding number:|00023|
//使用0指示符之后，输出的数字会在左边补0.
$ printf "padding number:|%-05d|\n" 23
padding number:|23   |
//如果同时使用了左对齐-指示符，那么0指示符不起作用.
```

##### 字段宽度控制。
```
//使输出的整数5个字符宽度，
$ printf "padding number:|%5d|\n" -2345678
padding number:|-2345678|
//如果数字宽度大于5，那么使用实际宽度(不进行截断),
$ printf "padding number:|%5d|\n" -23
padding number:|  -23|
//如果数字宽度小于5，那个左边使用空格占位补齐，
$ printf "padding number:|%-5d|\n" -23
padding number:|-23  |
//如果使用的左对齐方式，那么右边使用空格占位补齐
```

##### 精度控制。
精度控制使用`一个小数点`加上`紧接着的数字(精度值)`表示.
对于`%.e`和`%.f`格式类型，精度控制指定的是小数点后可保留的小数点位数；
如果小数点后的精度值没有，那么则是将精度值视为0
对于`字符串参数`则表示可输出的最大字符个数；

```
$ printf "padding number:|%.2f|\n" -23.9876543
padding number:|-23.99|
$ printf "padding number:|%.f|\n" -23.9876543
padding number:|-24|
$ printf "padding number:|%.2s|\n" "hello"
padding number:|he|
$ printf "padding number:|%.s|\n" "hello"
padding number:||
$
```

##### 参数格式类型
使用单一字符标识(one of `diouxXfFeEgGaAcsb`).
对于字段宽度控制和精度控制中，可能会使用`*`代替数字，这种情况下，由参数控制字段的`宽度`和`输出精度`。

参数格式|结果
---|---
**diouXx** | 指示参数会以带有符号的十进制数输出(d或者i), 无符号八进制(o), 无符号十进制(u),以及无符号十六进制(x或X)
**fF** | 指示参数会以`[-]ddd.ddd`的格式输出，其中小数点后的d个数等同于参数的精度控制，如果没有精度控制值，则使用默认值`6`；如果精度值是`0`，那么不会输出小数点及任何小数部分. The values infinity and NaN are printed as inf and nan, respectively.
**eE** | 指示参数以`[-d.ddd+-dd]`的格式输出，在小数点前面会有一位数字，小数点后的数字位数由精度值控制，如果没有精度控制值，则使用默认值`6`；The values infinity and NaN are printed as inf and nan, respectively.
**gG** | The argument is printed in style f (F) or in style e (E) whichever gives full precision in minimum space.
**aA** | The argument is printed in style [-h.hhh+-pd] where there is one digit before the hexadecimal point and the number after is equal to the precision specification for the
argument; when the precision is missing, enough digits are
produced to convey the argument’s exact double-precision floating-point representation. The values infinity and NaN are printed as inf and nan, respectively.
**%** | Print a %; no argument is used.
In no case does a non-existent or small field width cause truncation of a field; padding takes place only if the specified
field width exceeds the actual width.

```
$ printf "float number: %f\n" 1342.78654329
float number: 1342.786543
$ printf "float number: |%.4f|\n" -1342.78654329
float number: |-1342.7865|
$ printf "float number: |%.f|\n" -1342.78654329
float number: |-1343|
//精度控制默认值(小数点后6位)；第二种情形表示保留小数点后4位；第三种情形表示没有精度控制值时忽略小数点及小数。

$ printf "float number: |%.e|\n" -1342.78654329
float number: |-1e+03|
$ printf "float number: |%e|\n" -1342.78654329
float number: |-1.342787e+03|
$ printf "float number: |%.3e|\n" -1342.78654329
float number: |-1.343e+03|

$ printf "significant digit control: %.4G\n" 123456.3456256789
significant digit control: 1.235E+05
$ printf "significant digit control: %.4G\n" 56.3456256789
significant digit control: 56.35

$ printf "|%6.2f|\n"  567.98765442
|567.99|
//%N.nf: 其中N表示浮点数所占有的总的宽度，包括小数点，n表示小数点的个数

```



### C+：标准输出流 [cout << "Bobby" << endl]
预定义的对象 `cout` 是 `ostream` 类的一个实例。
**cout** 对象"连接"到标准输出设备，通常是显示屏。
**cout** 是与流插入运算符 `<<` 结合使用的，
如下所示：

```
#include <iostream>

using namespace std;

int main( )
{
   char str[] = "Hello C++";

   cout << "Value of str is : " << str << endl;
}
```

C++ 编译器根据要输出变量的数据类型，选择合适的流插入运算符来显示值。
`<<` 运算符被重载来输出内置类型`（整型、浮点型、double 型、字符串和指针）`的数据项。
流插入运算符 << 在一个语句中可以多次使用
`endl` 用于在行末添加一个换行符。

### 标准输入流

#### C : 输入 [scanf("%d", &number)]

```
#include <stdio.h>

char name[50];

printf ("Please enter your name:");
scanf("%s", & name);
printf("Hello, %s. \n", name);
```

**printf** ("想要输出的语句");
**scanf** ("**%s**", **&** name);
**printf** ("Hello, **%s**. \n", name);


#### C+ : 标准输入流 [cin >> name]
预定义的对象 cin 是 istream 类的一个实例。
**cin** 对象附属到标准输入设备，通常是键盘。
**cout** 是与流提取运算符 >> 结合使用的，
如下所示：

```
#include <iostream>

using namespace std;

int main( )
{
   char name[50];

   cout << "请输入您的名称： ";
   cin >> name;
   cout << "您的名称是： " << name << endl;
}
```

C++ 编译器根据要输入值的数据类型，选择合适的流提取运算符来提取值，并把它存储在给定的变量中。

流提取运算符 >> 在一个语句中可以多次使用，如果要求输入多个数据，可以使用如下语句：
`cin >> name >> age;`
这相当于下面两个语句：

```
cin >> name;
cin >> age;
```


### 标准错误流（cerr）
预定义的对象 cerr 是 ostream 类的一个实例。cerr 对象附属到标准错误设备，通常也是显示屏，但是 cerr 对象是非缓冲的，且每个流插入到 cerr 都会立即输出。
cerr 也是与流插入运算符 << 结合使用的，如下所示：
/#include <iostream>

using namespace std;

int main( )
{
   char str[] = "Unable to read....";

   cerr << "Error message : " << str << endl;
}
当上面的代码被编译和执行时，它会产生下列结果：
Error message : Unable to read....

### 标准日志流（clog）
预定义的对象 clog 是 ostream 类的一个实例。clog 对象附属到标准错误设备，通常也是显示屏，但是 clog 对象是缓冲的。这意味着每个流插入到 clog 都会先存储在缓冲在，直到缓冲填满或者缓冲区刷新时才会输出。
clog 也是与流插入运算符 << 结合使用的，如下所示：
/#include <iostream>

using namespace std;

int main( )
{
   char str[] = "Unable to read....";

   clog << "Error message : " << str << endl;
}
当上面的代码被编译和执行时，它会产生下列结果：
Error message : Unable to read....
通过这些小实例，我们无法区分 cout、cerr 和 clog 的差异，但在编写和执行大型程序时，它们之间的差异就变得非常明显。所以良好的编程实践告诉我们，使用 cerr 流来显示错误消息，而其他的日志消息则使用 clog 流来输出。
