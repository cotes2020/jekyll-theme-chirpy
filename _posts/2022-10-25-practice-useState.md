---
title: "Practice [React-useState]"
author: cotes
categories: practice
classes: wide
tag: [useState]
math: true
mermaid: true
---

## 버튼 클릭시 input value가 리스트 최상단에 추가되는 기능 

![image](https://user-images.githubusercontent.com/105469077/197806447-87ad08da-b560-4c5b-9417-c0b322783b12.png)

![image](https://user-images.githubusercontent.com/105469077/197806578-88050147-a490-4018-a7ae-eb9c5f9a3693.png)

<details>
<summary>정답</summary>
<div markdown='1'>

```javascript
import {useState} from 'react';

function App() {
  const [names, setNames] = useState(['홍길동', '김민수']);
  const [input, setInput] = useState('');

  const onInputChange = (e) => {
    setInput(e.target.value);
  }

  const handleUpload = () => {
    setNames((prevState) => {
      return [input, ...prevState];
    });
  };

  return (
    <div>
      <input type="text" value={input} onChange={onInputChange}/>
      <button onClick={handleUpload}>Upload</button>
      {names.map((v, idx) => {
        return <p key={idx}>{v}</p>
      })}
    </div>
  );
}

export default App;

```

</div>
</details>
