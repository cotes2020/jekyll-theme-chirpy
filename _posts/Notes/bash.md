-------------

### Bash Scripting
### Bash

--------------

### Bash manual

---------------

### What is a shell?

----------------

- A shell is a macro processor that executes commands.The term macro processor means functionality where text and symbols are expanded to create larger expressions.A Unix shell is both a command interpreter and a programming language. As a command interpreter, the shell provides the user interface to the rich set of GNU utilities. The programming language features allow these utilities to be combined. Files containing commands can be created, and become commands themselves. These new commands have the same status as system commands in directories such as /bin, allowing users or groups to establish custom environments to automate their common tasks.Shells may be used interactively or non-interactively. In interactive mode, they accept input typed from the keyboard. When executing non-interactively, shells execute commands read from a file.

---------------------

### SHELL SYNTAX

---------------------

- When the shell reads input, it proceeds through a sequence of operations. If the input indicates the beginning of a comment, the shell ignores the comment symbol (‘#’), and the rest of that line.

```bash
#script
```

- Shell mode of operation-:

```
- Reads it from a file or if it is passed as an argument with `-c`  or from the user terminal
- Breaks the input into words and operators, obeying the quoting rules described in Quoting. These tokens are separated by metacharacters.
- Parses the tokens into simple and compound commands
- Performs the various shell expansions (see Shell Expansions), breaking the expanded tokens into lists of filenames
- Performs any necessary redirections
- Optionally waits for the command to complete and collects its exit status
```

- Quoting is used to remove the special meaning of certain characters or words to the shell. Quoting can be used to disable special treatment for special characters, to prevent reserved words from being recognized as such, and to prevent parameter expansion.
- Each of the shell metacharacters (see Definitions) has special meaning to the shell and must be quoted if it is to represent itself. When the command history expansion facilities are being used (see History Expansion), the history expansion character, usually ‘!’, must be quoted to prevent history expansion.e.g

![image](https://github.com/user-attachments/assets/36f11f92-d194-48e6-9558-05920acd6711)

- Quoting Mechanisms-:

```
- Escape Character
- Single Quotes
- Double quotes
- Ansi-c Quoting
- Locale-Specific Translation
```

- Escape character `\` is the Bash escape character.It preserves the literal value of the next character that follows, with the exception of newline. If a \newline pair appears, and the backslash itself is not quoted, the \newline is treated as a line continuation (that is, it is removed from the input stream and effectively ignored).
- Enclosing characters in single quotes (‘'’) preserves the literal value of each character within the quotes. A single quote may not occur between single quotes, even when preceded by a backslash.
- Double quotes-: Enclosing characters in double quotes (‘"’) preserves the literal value of all characters within the quotes, with the exception of ‘$’, ‘\`’, ‘\’, and, when history expansion is enabled, ‘!’.The characters ‘$’ and "\`" retain their special meaning within double quotes.A double quote may be quoted within double quotes by preceding it with a backslash. If enabled, history expansion will be performed unless an ‘!’ appearing in double quotes is escaped using a backslash. Backslahing with non-special characters within double quotes won't affect the characters e.g `backslash before a double quotes within 2 double quotes `.The backslash preceding the ‘!’ is not removed.The special parameters ‘*’ and ‘@’ have special meaning when in double quotes.
- ANSI-C Quoting-: Character sequences of the form $’string’ are treated as a special kind of single quotes. The sequence expands to string, with backslash-escaped characters in string replaced as specified by the ANSI C standard.
- Locale-Specific Translation-:Prefixing a double-quoted string with a dollar sign (‘$’), such as `$"hello, world"`, will cause the string to be translated according to the current locale.

----------------

### Comments

----------------

- In a non-interactive shell, or an interactive shell in which the interactive_comments option to the shopt builtin is enabled (see The Shopt Builtin), a word beginning with ‘#’ causes that word and all remaining characters on that line to be ignored. An interactive shell without the interactive_comments option enabled does not allow comments. The interactive_comments option is on by default in interactive shells.

------------------

### SHELL COMMANDS

------------------

- A simple shell command can be "echo a c b",`echo` is the command and `a b c` are the arguments.More complex shell commands are composed of simple commands arranged together in a variety of ways: in a pipeline in which the output of one command becomes the input of a second, in a loop or conditional construct, or in some other grouping. 


------------------

- Reserved words are words that have a special meaning in the shell. They are used to begin and end the shell’s compound commands. The following words are recognized as reserved when unquoted and the first word of a command (see below for exceptions):

```bash
if
fi
then
elseif
time
for
in
until
while
do
done
case
esac
coproc
select
function
{
}
[[
]]
!
```

- `in` is recognized as a reserved word if it is the third word of a case or select command. in and do are recognized as reserved words if they are the third word in a for command.
- A simple command is the kind of command encountered most often. It’s just a sequence of words separated by blanks, terminated by one of the shell’s control operators (see Definitions). The first word generally specifies a command to be executed, with the rest of the words being that command’s arguments.
- A pipeline is a set of simple commands separated by `|` or `|&`.e.g

```bash
ls|base64
```

- If the `|&` is used, the standard error is passed to the other command along with the `stdout`.`|&` is the shorthand for `2>&1`.


