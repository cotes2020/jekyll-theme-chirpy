To get started, set your OPENAI_API_KEY environment variable.

Next, change a few of the prompts in prompts.txt and edit promptfooconfig.yaml.

Then run:
```
promptfoo eval
```

Afterwards, you can view the results by running `promptfoo view`


## check output

### all user in loop

```json
[
    {% for completion in _conversation %}
    {
        "role": "user",
        "content": "{{ system_message }}" # A
    },
    {
        "role": "user",
        "content": "{{ completion.input }}" # B
    },
    {
        "role": "assistant",
        "content": "{{ completion.output }}"
    },
    {% endfor %}
    {
        "role": "user",
        "content": "{{ question }} {{code}}" # C
    }
]

// output 1
[
    {
        "role": "user",
        "content": "" # C
    }
]
// output 2
[
    {
        "role": "user",
        "content": "" # A
    },
    {
        "role": "user",
        "content": "[
            {
                "role": "user",
                "content": "" # C
            }
        ]"
    },
    {
        "role": "assistant",
        "content": "..."
    },
    {
        "role": "user",
        "content": "" # C
    }
]
```

### user outside loop


```json
[
    {
        "role": "user",
        "content": "{{ system_message }}" # A
    },
    {% for completion in _conversation %}
    {
        "role": "user",
        "content": "{{ completion.input }}" # B
    },
    {
        "role": "assistant",
        "content": "{{ completion.output }}"
    },
    {% endfor %}
    {
        "role": "user",
        "content": "{{ question }} {{code}}" # C
    }
]

// output 1
[
    {
        "role": "user",
        "content": "" # A
    },

    {
        "role": "user",
        "content": "" # C
    }
]
// output 2
[
    {
        "role": "user",
        "content": "" # A
    },
    {
        "role": "user",
        "content": "[
            {
                "role": "user",
                "content": "" # A
            }
            {
                "role": "user",
                "content": "" # C
            }
        ]"
    },
    {
        "role": "assistant",
        "content": "..."
    },
    {
        "role": "user",
        "content": "" # C
    }
]
```

### user inside loop

```json
[
    {% for completion in _conversation %}
    {
        "role": "user",
        "content": "{{ completion.input }}" # B
    },
    {
        "role": "assistant",
        "content": "{{ completion.output }}"
    },
    {% endfor %}
    {
        "role": "user",
        "content": "{{ question }} {{code}}" # C
    }
]

// output 1
[
    {
        "role": "user",
        "content": "" # C
    }
]
// output 2
[
    {
        "role": "user",
        "content": "[
            {
                "role": "user",
                "content": "" # C
            }
        ]"
    },
    {
        "role": "assistant",
        "content": "..."
    },
    {
        "role": "user",
        "content": "" # C
    }
]
```



### system outside loop


```json
[
    {
        "role": "system",
        "content": "{{ system_message }}"
    },
    {% for completion in _conversation %}
    {
        "role": "user",
        "content": "{{ completion.input }}" # B
    },
    {
        "role": "assistant",
        "content": "{{ completion.output }}"
    },
    {% endfor %}
    {
        "role": "user",
        "content": "{{ question }} {{code}}" # C
    }
]

// output 1
[
    {
        "role": "system",
        "content": "{{ system_message }}"
    },
    {
        "role": "user",
        "content": "" # C
    }
]
// output 2
[
    {
        "role": "system",
        "content": "{{ system_message }}"
    },
    {
        "role": "user",
        "content": "[
            {
                "role": "system",
                "content": "{{ system_message }}"
            },
            {
                "role": "user",
                "content": "" # C
            }
        ]"
    },
    {
        "role": "assistant",
        "content": "..."
    },
    {
        "role": "user",
        "content": "" # C
    }
]
```





### system outside loop + length-1


```json
[
    {
        "role": "system",
        "content": "{{ system_message }}"
    },
    {% for completion in _conversation %}
    {
        "role": "user",
        "content": "{{ completion.prompt[completion.prompt.length - 1].content }}" # B
    },
    {
        "role": "assistant",
        "content": "{{ completion.output }}"
    },
    {% endfor %}
    {
        "role": "user",
        "content": "{{ question }} {{code}}" # C
    }
]

// output 1
[
    {
        "role": "system",
        "content": "{{ system_message }}"
    },
    {
        "role": "user",
        "content": "" # C
    }
]
// output 2
[
    {
        "role": "system",
        "content": "{{ system_message }}"
    },
    {
        "role": "user",
        "content": ""
    },
    {
        "role": "assistant",
        "content": "..."
    },
    {
        "role": "user",
        "content": "" # C
    }
]
```
