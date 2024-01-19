
- [Work Command](#work-command)
  - [github repo search](#github-repo-search)
  - [reuse token](#reuse-token)
- [My Issue](#my-issue)
  - [Github issue](#github-issue)
    - [Github upload](#github-upload)
  - [VSC Issue](#vsc-issue)
    - [auto save](#auto-save)
  - [My M1 Issue](#my-m1-issue)
    - [Install brew](#install-brew)
    - [install Pyenv](#install-pyenv)
      - [Prerequisites](#prerequisites)
      - [intsall Pyenv](#intsall-pyenv)
      - [Use pyenv](#use-pyenv)
      - [Use Virtual Environments](#use-virtual-environments)
      - [errors you might meet](#errors-you-might-meet)
    - [installation for Conda](#installation-for-conda)
    - [installation](#installation)
      - [pip](#pip)
      - [brew](#brew)
    - [terraform](#terraform)

---

# Work Command

---

## github repo search

```bash
page=1
per_page=100
while true; do
    gh search repos --language=java --visibility=public \
      --sort="updated" --order="asc" \
      --limit $per_page --json url \
      | jq -r '.[].url' \
      | while read -r url; do
        echo "Performing action for repository: $url"
    done

    ((page++))

    if [ $page -gt 200 ]; then
        break
    fi
done > output.txt


gh search repos --language=java --visibility=public \
      --sort="updated" --order="asc" \
      --limit $per_page --json url \
      | jq -r '.[].url'


page=1
per_page=100
while true; do
    gh search repos --language=java --visibility=public \
      --sort="updated" --order="asc" \
      --limit $per_page --json url \
      | jq -r '.[].url' \
      | while read -r url; do
        echo "Performing action for repository: $url"
    done

    ((page++))

    if [ $page -gt 2 ]; then
        break
    fi
done > output.txt
```


```bash
#!/bin/bash

# Read the input file line by line
while IFS= read -r url; do
    # Copy URL from fileA to fileB
    echo "$url" >> fileB

    # Initialize a counter for retries
    retries=0
    size="NA"  # Default value for size

    # Get the repository name from the URL
    repo_name=$(echo "$url" | awk -F'/' '{print $NF}' | tr -d '\r\n')

    # Try to get the size up to 3 times
    while [ $retries -lt 3 ]; do
        # Make an API request to get the repository size
        response=$(curl -s "https://api.github.com/repos/$repo_name" | grep '"size":')

        if [ -n "$response" ]; then
            # Extract the size value from the response
            size=$(echo "$response" | awk -F': ' '{print $2}' | tr -d ',')

            # Break out of the loop if size is obtained successfully
            break
        else
            # Increment the retries counter
            ((retries++))
            # Add a delay before the next retry (optional)
            sleep 1
        fi
    done

    # Append the size to fileB
    echo "Size: $size" >> fileB
done < fileA

# Iterate through fileB to check for URLs without sizes and mark them as NA
# This will start from the top of fileB again
while read -r line; do
    if [[ $line == "Size: NA" ]]; then
        url=$(sed -n '/Size: NA/{n;p}' fileB)  # Get the URL associated with "Size: NA"
        repo_name=$(echo "$url" | awk -F'/' '{print $NF}' | tr -d '\r\n')

        # Try to get the size up to 3 times
        retries=0
        while [ $retries -lt 3 ]; do
            response=$(curl -s "https://api.github.com/repos/$repo_name" | grep '"size":')

            if [ -n "$response" ]; then
                size=$(echo "$response" | awk -F': ' '{print $2}' | tr -d ',')
                sed -i "s|${url}|${url}\nSize: ${size}|" fileB  # Replace URL with URL + size
                break
            else
                ((retries++))
                sleep 1
            fi
        done
    fi
done < fileB

```



## reuse token

```bash
TOKEN=$(echo "hi")
echo $TOKEN

# get token
export TOKEN=$(curl \
  --url https://website/token/generate \
  --data-binary '{
    "context":"12345",
    "ttl":86400
  }' \
  -H 'Content-Type: application/json' | jq '.token' | sed 's/"//g')

echo $TOKEN

# validate token
curl --url https://website/token/validate \
     -H 'Content-Type: application/json' \
     --data-binary '{
        "context":"12345",
        "ttl":86400,
        "token":"'$TOKEN'"
     }'

curl -X POST 'https://website/token/scans' \
  -H 'Authorization: Bearer '$TOKEN'' \
  -H 'Content-Type: application/json' \
  -d '{
    "repo": "'$TARGETREPO'",
    "branch": "'$TARGETBRANCH'"
  }'


curl 'https://checkers.apple.com/api/v2/scans/'$UUID'' \
  -H 'Authorization: Bearer '$TOKEN'' \
  -H 'Content-Type: application/json' | jq



export   -H 'Authorization: Bearer '$TOKEN'' \
  -H 'Content-Type: application/json' | jq



```


---


# My Issue

Hopefully you will never need this

---

## Github issue

### Github upload


```bash
zip -r my_folder.zip ./my_folder

unzip fun-macos.zip

split -b 99m SketchUp.zip SketchUp-

cat SketchUp-* > SketchUp2023.dmg
```


---

## VSC Issue

### auto save

open the Settings editor from the Command Palette (⇧⌘P) with Preferences: Open Settings or use the keyboard shortcut (⌘,).

Settings:

- Commonly used
  - AutoSave: after delay

- Python Docstring Generate Configuration
  - Quote Style: `"` or `'`


--


## My M1 Issue

### Install brew

For what it's worth, before installing Homebrew you will need to install Rosetta2 emulator for the new ARM silicon (M1 chip).

```bash
# installed Rosetta2 via terminal using:
/usr/sbin/softwareupdate --install-rosetta --agree-to-license

# use the Homebrew cmd and install Homebrew for ARM M1 chip:
arch -x86_64 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"

# Once Homebrew for M1 ARM is installed use this Homebrew command to install packages:
arch -x86_64 brew install <package>
```

open a Rosetta shell first.

```bash
% arch -x86_64 zsh
% cd /usr/local && mkdir homebrew
% curl -L https://github.com/Homebrew/brew/tarball/master | tar xz --strip 1 -C homebrew
```


---


### install Pyenv


#### Prerequisites

1. install xcode in app store

```bash
sudo rm -rf /Library/Developer/CommandLineTools
xcode-select --install
```


2. install the Python build dependencies.

```bash
brew install openssl readline sqlite3 xz zlib libxml2 libxslt
```



#### intsall Pyenv


**Install**

```bash
# 1. get latest version of pyenv and make it easy to fork and contribute any changes back upstream.
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
cd ~/.pyenv && src/configure && make -C src

# 2. use brew
brew install pyenv pyenv-virtualenv
```


**environment variable**

1. Define environment variable PYENV_ROOT
   1. to point to the path where pyenv repo is cloned
   2. and add `$PYENV_ROOT/bin` to `$PATH` for access to the pyenv command-line utility.

1. in the ~/.zshrc file add the following:

```bash
# For Zsh
echo '# for pyenv' >> ~/.zshrc
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc

# Restart shell
exec "$SHELL"
```


```bash
export PATH="$HOME/.pyenv/bin:$PATH"
export PATH="/usr/local/bin:$PATH"


export LDFLAGS="-L/usr/local/opt/zlib/lib -L/usr/local/opt/bzip2/lib"
export CPPFLAGS="-I/usr/local/opt/zlib/include -I/usr/local/opt/bzip2/include"
# export LDFLAGS="-L/usr/local/homebrew/opt/zlib/lib"
# export CPPFLAGS="-I/usr/local/homebrew/opt/zlib/include"
# export LDFLAGS="-L/usr/local/homebrew/opt/bzip2/lib"
# export CPPFLAGS="-I/usr/local/homebrew/opt/bzip2/include"


eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```

1. Add pyenv init to shell
   1. to enable shims and autocompletion.
   2. make sure `eval $(pyenv init -)` is placed toward *the end of the shell configuration file* since it manipulates PATH during the initialization.

```bash
echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n  eval "$(pyenv init -)"\nfi' >> ~/.zshrc
```


1. Restart shell

```bash
# Restart shell
exec "$SHELL"
```



**Install python version**

1. Align command-line tools
    1. check the version of command-line tools matches Xcode.
    2. Open Xcode > Preference > Locations > Command Line Tools

2. install python version


```bash
# download the python
sudo CFLAGS="-I$(brew --prefix openssl)/include \
            -I$(brew --prefix bzip2)/include \
            -I$(brew --prefix readline)/include \
            -I$(xcrun --show-sdk-path)/usr/include"

sudo LDFLAGS="-L$(brew --prefix openssl)/lib \
        -L$(brew --prefix readline)/lib \
        -L$(brew --prefix zlib)/lib \
        -L$(brew --prefix bzip2)/lib"

pyenv install --patch 3.7.7 < <(curl -sSL https://github.com/python/cpython/commit/8ea6353.patch\?full_index\=1)

pyenv install --patch 3.8.0 < <(curl -sSL https://github.com/python/cpython/commit/8ea6353.patch\?full_index\=1)

pyenv install --patch 3.7.2 < <(curl -sSL https://github.com/python/cpython/commit/8ea6353.patch\?full_index\=1)




CFLAGS="-I$(brew --prefix openssl)/include \
            -I$(brew --prefix bzip2)/include \
            -I$(brew --prefix readline)/include \
            -I$(xcrun --show-sdk-path)/usr/include"

LDFLAGS="-L$(brew --prefix openssl)/lib \
        -L$(brew --prefix readline)/lib \
        -L$(brew --prefix zlib)/lib \
        -L$(brew --prefix bzip2)/lib"

PYTHON_CONFIGURE_OPTS=--enable-unicode=ucs2

pyenv install -v 3.7.2


CFLAGS="-I$(brew --prefix readline)/include \
        -I$(brew --prefix openssl)/include \
        -I$(xcrun --show-sdk-path)/usr/include" \
LDFLAGS="-L$(brew --prefix readline)/lib \
         -L$(brew --prefix openssl)/lib" \
PYTHON_CONFIGURE_OPTS=--enable-unicode=ucs2 \
pyenv install -v 3.7.2



CFLAGS="-I$(brew --prefix openssl)/include \
            -I$(brew --prefix bzip2)/include \
            -I$(brew --prefix readline)/include \
            -I$(xcrun --show-sdk-path)/usr/include" \
LDFLAGS="-L$(brew --prefix openssl)/lib \
        -L$(brew --prefix readline)/lib \
        -L$(brew --prefix zlib)/lib \
        -L$(brew --prefix bzip2)/lib" \
pyenv install --patch 3.7.7 < <(curl -sSL https://github.com/python/cpython/commit/8ea6353.patch\?full_index\=1)



# install your python version
pyenv install 3.6.5
pyenv install 3.7.7
pyenv install 3.8.2
pyenv global 3.6.5 3.7.7 3.8.2
pyenv local 3.6.5



pyenv install 3.8.13
# python-build: use openssl@1.1 from homebrew
# python-build: use readline from homebrew
# Downloading Python-3.8.13.tar.xz...
# -> https://www.python.org/ftp/python/3.8.13/Python-3.8.13.tar.xz
# Installing Python-3.8.13...
# python-build: use tcl-tk from homebrew
# python-build: use readline from homebrew
# python-build: use zlib from xcode sdk
# Installed Python-3.8.13 to /Users/graceluo/.pyenv/versions/3.8.13
```


**check installation**


```bash
$ python -V
% pyenv --version
# pyenv 1.2.24.1

% pyenv versions
# * system (set by /Users/l/.pyenv/version)
#   3.7.7
# * system (set by /home/grace/.pyenv/version)
#   2.7.15
#   3.6.8
#   3.8-dev


$ which python
# /home/grace/.pyenv/shims/python


# configure brew
# edit .zshrc file
alias brew="env PATH=${PATH//$(pyenv root)\/shims:/} brew"
```


pyenv inserts itself into PATH
* from OS’s perspective is the executable that is getting called.
* to see the actual path, can run the following:

```bash
$ pyenv which python
# /usr/bin/python
```



#### Use pyenv

**Specifying Python Version**

* 3 ways to modify which version of python you’re using.

```bash
# =======================
$ pyenv versions
* system (set by /home/grace/.pyenv/version)
  2.7.15
  3.6.8
  3.8-dev
#  system Python is being used as denoted by the *.

# To exercise the next most global setting, use global command:
$ pyenv global 2.7.15


# create a .python-version file with local:
$ pyenv local 2.7.15


# set the Python version with shell:
$ pyenv shell 3.8-dev

# All this did is set the $PYENV_VERSION environment variable:
$ echo $PYENV_VERSION
3.8-dev
```



#### Use Virtual Environments


**Creating Virtual Environments**

```bash
# Creating Virtual Environments**
$ pyenv virtualenv <python_version> <environment_name>

$ pyenv virtualenv 3.6.8 myproject
# The output includes messages that show a couple of extra Python packages getting installed, namely wheel, pip, and setuptools.
# This is strictly for convenience and just sets up a more full featured environment for each of virtual environments.


# Activating Versions
$ pyenv local myproject

# delete pyenv
$ pyenv uninstall 3.8.13/envs/lmeval
```


**Working With Multiple Environments**

```bash
$ pyenv versions
# * system (set by /home/grace/.pyenv/version)
#   2.7.15
#   3.6.8
#   3.8-dev

# to work on two different, aptly named, projects:
* project1 supports Python 2.7 and 3.6.
* project2 supports Python 3.6 and experiments with 3.8-dev.


# First, create a virtual environment for the first project
$ cd project1/
$ pyenv which python
# /usr/bin/python
$ pyenv virtualenv 3.6.8 project1
$ pyenv local project1
$ python -V
# /home/grace/.pyenv/versions/project1/bin/python



# cd out of the directory, default back to the system Python
$ cd $HOME
$ pyenv which python
# /usr/bin/python


# create a virtual environment for project2
$ cd project2/
$ pyenv which python
# /usr/bin/python
$ pyenv virtualenv 3.8-dev project2
$ pyenv local 3.8-dev
$ pyenv which python
# /home/grace/.pyenv/versions/3.8-dev/bin/python



# These are one time steps for projects.
# cd between the projects, environments will automatically activate:
$ cd project2/
$ python -V
# Python 3.8.0a0

$ cd ../project1
$ python -V
# Python 3.6.8
```






#### errors you might meet



1. got error output

```bash
env PYENV_DEBUG=1 pyenv install -v 3.6.10 2>&1 | tee trace.log
```




1. cannot install other python version
```bash
$ pyenv install 3.9.2
# ...
# Please consult to the Wiki page to fix the problem.
# https://github.com/pyenv/pyenv/wiki/Common-build-problems
# BUILD FAILED (OS X 11.2.1 using python-build 1.2.24.1)
```

> It is usually something wrong with the python version and the python path


```bash
$ which python
# /Users/l/.pyenv/shims/python

$ pyenv versions
#   system
#   3.7.7
#   3.9.1
# * 3.9.2 (set by /Users/l/.python-version)
```



```bash
# solution
arch -x86_64 pyenv install 3.7.2
# BUILD FAILED (OS X 12.6 using python-build 20180424)
arch -x86_64 pyenv install 3.8.9
# BUILD FAILED (OS X 12.6 using python-build 20180424)



# solution
CC=/opt/homebrew/bin/gcc-11 pyenv install 3.8.12



# solution
arch -x86_64 /usr/local/bin/brew install gcc
arch -x86_64 ./configure \
    --with-openssl=/usr/local/opt/openssl@3 \
    --prefix=/Users/devin/.pyenv/versions/3.6.1 \
    --libdir=/Users/devin/.pyenv/versions/3.6.1/lib
CC=/usr/local/Cellar/gcc/11.2.0_3/bin/gcc-11
LDFLAGS="-L/usr/local/opt/bzip2/lib -L/usr/local/opt/zlib/lib -L/usr/local/opt/openssl@3/lib"
CPPFLAGS="-I/usr/local/opt/bzip2/include -I/usr/local/opt/zlib/include -I/usr/local/opt/openssl@3/include"
PKG_CONFIG_PATH="/usr/local/opt/zlib/lib/pkgconfig:/usr/local/opt/openssl@3/lib/pkgconfig"

make && make install




# solution
brew uninstall openssl && brew install openssl
CFLAGS="-I$(brew --prefix openssl)/include" \
LDFLAGS="-L$(brew --prefix openssl)/lib" \
pyenv install 3.7.0
# set the python path
export PYTHONPATH="/Users/l/.pyenv/shims/python3"




# solution
CFLAGS="-I$(brew --prefix openssl)/include \
            -I$(brew --prefix bzip2)/include \
            -I$(brew --prefix readline)/include \
            -I$(xcrun --show-sdk-path)/usr/include"
LDFLAGS="-L$(brew --prefix openssl)/lib \
        -L$(brew --prefix readline)/lib \
        -L$(brew --prefix zlib)/lib \
        -L$(brew --prefix bzip2)/lib"
PYTHON_CONFIGURE_OPTS=--enable-unicode=ucs2
pyenv install -v 3.7.2

CFLAGS="-I$(brew --prefix readline)/include \
        -I$(brew --prefix openssl)/include
        -I$(xcrun --show-sdk-path)/usr/include"
LDFLAGS="-L$(brew --prefix readline)/lib \
        -L$(brew --prefix openssl)/lib" \
PYTHON_CONFIGURE_OPTS=--enable-unicode=ucs2
pyenv install -v 3.7.2




# solution
sudo installer -pkg /Library/Developer/CommandLineTools/Packages/macOS_SDK_headers_for_macOS_10.14.pkg -target /



# solution
# add just this to my .bash_profile
export SDKROOT=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk


# solution
SDKROOT=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.14.sdk MACOSX_DEPLOYMENT_TARGET=10.14 pyenv install 3.7.3



# solution
SDKROOT=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.14.sdk
MACOSX_DEPLOYMENT_TARGET=10.14
pyenv install 3.7.2

# python-build: use openssl@1.1 from homebrew
# python-build: use readline from homebrew
# Downloading Python-3.7.2.tar.xz...
# -> https://www.python.org/ftp/python/3.7.2/Python-3.7.2.tar.xz
# Installing Python-3.7.2...
# python-build: use tcl-tk from homebrew
# python-build: use readline from homebrew
# python-build: use zlib from xcode sdk

# BUILD FAILED (OS X 12.6 using python-build 20180424)

# Inspect or clean up the working tree at /var/folders/hw/52gfb9js43d4xb9g0zypmljr0000gn/T/python-build.20221007110445.69078
# Results logged to /var/folders/hw/52gfb9js43d4xb9g0zypmljr0000gn/T/python-build.20221007110445.69078.log

# Last 10 log lines:
# checking for --with-cxx-main=<compiler>... no
# checking for clang++... no
# configure:

#   By default, distutils will build C++ extension modules with "clang++".
#   If this is not intended, then set CXX on the configure command line.

# checking for the platform triplet based on compiler characteristics... darwin
# configure: error: internal configure error for the platform triplet, please file a bug report
# make: *** No targets specified and no makefile found.  Stop.
```





1. when you see this, means you need to brew install every app it shows below.

```bash
$ brew install pyenv
Error: pyenv dependencies not built for the arm64 CPU architecture:
  openssl@1.1 was built for x86_64
  pkg-config was built for x86_64
  readline was built for x86_64
```



1. sqlite3 error

```bash
# // error
ModuleNotFoundError: No module named '_sqlite3'

# // solutions
brew reinstall openssl
pyenv install 3.9.1 && pyenv global 3.9.1
```



```bash
# =============================
Using terminal got this error, change to Rosetta terminal
# =============================
# configure: error: Unexpected output of 'arch' on OSX
# make: *** No targets specified and no makefile found.  Stop.


$ pyenv install 3.7.2
# python-build: use openssl@1.1 from homebrew
# python-build: use readline from homebrew
# Downloading Python-3.7.2.tar.xz...
# -> https://www.python.org/ftp/python/3.7.2/Python-3.7.2.tar.xz
# Installing Python-3.7.2...
# python-build: use readline from homebrew
# python-build: use zlib from xcode sdk
# BUILD FAILED (OS X 11.2.1 using python-build 1.2.24.1)
# Inspect or clean up the working tree at /tmp/python-build.20210319230240.79971
# Results logged to /tmp/python-build.20210319230240.79971.log
# Last 10 log lines:
# extern int _NSGetExecutablePath(char* buf, uint32_t* bufsize)                 __OSX_AVAILABLE_STARTING(__MAC_10_2, __IPHONE_2_0);
#                                                      ^
# clang -Wno-unused-result -Wsign-compare -Wunreachable-code -DNDEBUG -g -fwrapv -O3 -Wall -I/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include   -I/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include   -std=c99 -Wextra -Wno-unused-result -Wno-unused-parameter -Wno-missing-field-initializers -Wstrict-prototypes -Werror=implicit-function-declaration   -I. -I./Include -I/usr/local/homebrew/opt/readline/include -I/usr/local/homebrew/opt/readline/include -I/Users/l/.pyenv/versions/3.7.2/include  -I/usr/local/homebrew/opt/readline/include -I/usr/local/homebrew/opt/read line/include -I/Users/l/.pyenv/versions/3.7.2/include   -DPy_BUILD_CORE_BUILTIN  -c ./Modules/errnomodule.c -o Modules/errnomodule.o
# ./Modules/posixmodule.c:8409:15: error: implicit declaration of function 'sendfile' is invalid in C99 [-Werror,-Wimplicit-function-declaration]
#         ret = sendfile(in, out, offset, &sbytes, &sf, flags);
#               ^
# 1 error generated.
# make: *** [Modules/posixmodule.o] Error 1
# make: *** Waiting for unfinished jobs....
# 1 warning generated.



# =============================
in terminal : fail
and R terminal: Successfully!!
# =============================
CFLAGS="-I$(brew --prefix openssl)/include \
    -I$(brew --prefix bzip2)/include \
    -I$(brew --prefix readline)/include \
    -I$(xcrun --show-sdk-path)/usr/include" \
    LDFLAGS="-L$(brew --prefix openssl)/lib \
    -L$(brew --prefix readline)/lib \
    -L$(brew --prefix zlib)/lib \
    -L$(brew --prefix bzip2)/lib" \
    pyenv install --patch 3.7.7 < <(curl -sSL https://github.com/python/cpython/commit/8ea6353.patch\?full_index\=1)
# python-build: use openssl@1.1 from homebrew
# python-build: use readline from homebrew
# Downloading Python-3.7.7.tar.xz...
# -> https://www.python.org/ftp/python/3.7.7/Python-3.7.7.tar.xz
# Installing Python-3.7.7...
# patching file Misc/NEWS.d/next/macOS/2020-06-24-13-51-57.bpo-41100.mcHdc5.rst
# patching file configure
# Hunk #1 succeeded at 3374 (offset -52 lines).
# patching file configure.ac
# Hunk #1 succeeded at 490 (offset -20 lines).
# python-build: use readline from homebrew
# python-build: use zlib from xcode sdk
# BUILD FAILED (OS X 11.2.1 using python-build 1.2.24.1)
# Inspect or clean up the working tree at /var/folders/gq/djp7wv2j6rvcgwjwl2c82lv80000gn/T/python-build.20210320190320.26757
# Results logged to /var/folders/gq/djp7wv2j6rvcgwjwl2c82lv80000gn/T/python-build.20210320190320.26757.log
# Last 10 log lines:
# checking size of _Bool... 1
# checking size of off_t... 8
# checking whether to enable large file support... no
# checking size of time_t... 8
# checking for pthread_t... yes
# checking size of pthread_t... 8
# checking size of pthread_key_t... 8
# checking whether pthread_key_t is compatible with int... no
# configure: error: Unexpected output of 'arch' on OSX
# make: *** No targets specified and no makefile found.  Stop.
```



---



### installation for Conda


```bash

$ conda create --name ct-repo python=3.7
$ conda create --name python37 python=3.7
$ conda create --name python38 python=3.8
$ conda create --name python26 python=2.6
$ conda install -n python35


$ conda info --envs
# conda environments:
# base                  *  /opt/homebrew/Caskroom/miniconda/base
# python37                 /opt/homebrew/Caskroom/miniconda/base/envs/python37
# python377                /opt/homebrew/Caskroom/miniconda/base/envs/python377
# python38                 /opt/homebrew/Caskroom/miniconda/base/envs/python38
# python392                /opt/homebrew/Caskroom/miniconda/base/envs/python392


eval "$(conda shell.bash hook)"

conda activate python377


$ source activate python35
$ source activate python37
$ source activate python377

$ python -V
# Python 3.5.4 :: Anaconda, Inc.

conda list -n



# create virtualenv with the created conda env
$ pyenv virtualenv /opt/homebrew/Caskroom/miniconda/base/envs/python377 ct-python377

# set local env (as default env in the current path)
$ pyenv local ct-python377

conda activate ct-python377


```

---


### installation

#### pip

```bash
python3.9 -m pip install --only-binary=:all: numpy==1.18.4
```

So somehow pip is not recognizing the binary wheels and is rebuilding the wheel from source, detecting the buggy local Accelerate library and binding it to NumPy.

Now we need someone to find out why the python you are using decided that the macosx_10_9_x86_64 tag is not acceptable anymore. The installer available via python.org still lists "for OS X 10.9 and later" on the macOSx line. Where did you install python from?


#### brew

```bash
arch -x86_64 brew update & arch -x86_64 brew upgrade
```


---

### terraform

error

```bash
╷
│ Error: Incompatible provider version
│
│ Provider registry.terraform.io/hashicorp/random v2.3.1 does not have a package available for your current platform, darwin_arm64.
│
│ Provider releases are separate from Terraform CLI releases, so not all providers are available for all platforms. Other versions of this provider may have different platforms supported.
╵

# solution:
brew uninstall terraform
brew install tfenv
TFENV_ARCH=amd64 tfenv install 1.3.3
tfenv use 1.3.3

arch -x86_64 brew install terraform
brew unlink tfenv
```






.
