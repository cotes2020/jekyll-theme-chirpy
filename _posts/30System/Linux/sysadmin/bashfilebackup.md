
# bash file back up

```bash
PATH=$JAVA_HOME/bin:$PATH:.
CLASSPATH=$JAVA_HOME/lib/tools.jar:$JAVA_HOME/lib/dt.jar:.
export JAVA_HOME
export PATH
export CLASSPATH

export PATH="/usr/local/sbin:$PATH"

export PATH=$PATH:/usr/local/mysql/bin

##
# Your previous /Users/luo/.bash_profile file was backed up as /Users/luo/.bash_profile.macports-saved_2019-10-04_at_18:50:17
##

# MacPorts Installer addition on 2019-10-04_at_18:50:17: adding an appropriate PATH variable for use with MacPorts.
export PATH="/opt/local/bin:/opt/local/sbin:$PATH"
# Finished adapting your PATH environment variable for use with MacPorts.

export JAVA_HOME=$(/usr/libexec/java_home -v 1.8)

export PATH="$HOME/.gem/ruby/2.6.0/bin:$PATH"

export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"  # This loads nvm bash_completion

export JAVA_11_HOME=/Library/Java/JavaVirtualMachines/adoptopenjdk-11.jdk/Contents/Home
export JAVA_8_HOME=/Library/Java/JavaVirtualMachines/jdk8.jdk/Contents/Home

alias jdk11='export JAVA_HOME=$JAVA_11_HOME'
alias jdk8='export JAVA_HOME=$JAVA_8_HOME'

#default jdk8
export JAVA_HOME=$JAVA_11_HOME

burp2.1()
{
/Library/Java/JavaVirtualMachines/adoptopenjdk-14.jdk/Contents/Home/bin/java -jar /Applications/Burp\ Suite\ Community\ Edition.app/Contents/java/app/burpsuite_pro_v2.1.07.jar
}

# The next line is for HTB
alias firesox='ssh -i id_rsa -D 1337 -f -C -q -N grace@research.cdg.io && /Applications/Firefox.app/Contents/MacOS/firefox &'


# The next line updates PATH for the Google Cloud SDK.
if [ -f '/Users/luo/google-cloud-sdk/path.bash.inc' ]; then . '/Users/luo/google-cloud-sdk/path.bash.inc'; fi


# The next line enables shell command completion for gcloud.
if [ -f '/Users/luo/google-cloud-sdk/completion.bash.inc' ]; then . '/Users/luo/google-cloud-sdk/completion.bash.inc'; fi


# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/Users/luo/opt/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/Users/luo/opt/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/Users/luo/opt/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/Users/luo/opt/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
```

`source ~/.bash_profile`



ref:
- [ref](https://natelandau.com/my-mac-osx-bash_profile/)
