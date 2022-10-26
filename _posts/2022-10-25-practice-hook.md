---
title: "Practice [React-Hook]"
author: cotes
categories: practice
classes: wide
tag: [hook]
math: true
mermaid: true
---

`다음의 요구사항을 충족하는 Printer.js, Calc.js 페이지를 만들어 App.js를 통해 화면에 표시하시오.`

## Printer.js
### 요구사항
![image](https://user-images.githubusercontent.com/105469077/197727828-431b0b2a-c3a1-4647-9974-a0ae7a1a4392.png)

### 실행결과
![image](https://user-images.githubusercontent.com/105469077/197727993-7d129a05-ce76-4b0d-8482-53473fabc087.png)


## Calc.js
### 요구사항
![image](https://user-images.githubusercontent.com/105469077/197728187-db9d44e9-e3c2-4ffb-9dd1-27786ef336fa.png)

### 실행결과
![image](https://user-images.githubusercontent.com/105469077/197728259-e752871f-7f07-4e61-a2b6-45ee55aca151.png)

<details>
<summary>정답</summary>
<div markdown='1'>

```javascript
// PrintStar.js
import React from 'react'

const PrintStar = () => {
  const console = React.useRef();

  const [rowNum, setRowNum] = React.useState(0);

  const onValueChange = (e) => {
    setRowNum(e.currentTarget.value);
  }

  React.useEffect(() => {
    let str = '';
    for (let i = 0; i < rowNum; i++) {
      for (let j = 0; j <= i; j++) {
        str += '*';
      }
      str += '<br/>';
    }
    console.current.innerHTML = str;
  }, [rowNum])


  return (
    <div>
      <h2>PrintStar</h2>
      <p>useState, useEffect, useRef를 사용한 별찍기 구현</p>
      <hr />
      <div>
        <label htmlFor="rowNumInput">rownum: </label>
        <input id='rowNumInput' type="text" value={rowNum} onChange={onValueChange}/>
      </div>
      <hr />
      <div ref={console}></div>
    </div>
  )
}

export default PrintStar;
```

</div>
</details>
