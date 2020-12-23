---
title: programmers algorithm test Level1 첫번째
author: juyoung
date: 2020-10-21 18:46:00 +0800
categories: [algorithm, Level1]
tags: [algorithm]
---

'2020.10.21' 부터 프로그래머스에 있는 알고리즘 연습문제를 풀기 시작했다.  
처음 일주일은 해답을 보지 않으면 2-3시간을 고민해도 안풀렸는데 조금씩 비슷한 풀이가 나오기 시작하자 혼자 힘으로 쉬운 문제는 풀리는 게 신기했다. 

내가 푼 방식과 함께 프로그래머스에 있던 다른 기발한 답들도 함께 기록한다.
  

# 1. 2016년

 
```javascript
function solution(a, b) {
  
    const week = ['SUN','MON','TUE','WED','THU','FRI','SAT'];
   
    return week[new Date(2020,a-1,b).getDay()];
}
console.log(solution(10,23)); 

```


# 2. 같은 숫자는 싫어  


```javascript
//sol1
function solution(arr) {
    var answer = [arr[0]];

    for (let i=1; i<arr.length; i++) {
        if (answer[answer.length - 1] !== arr[i]) {
            answer.push(arr[i]);
        }
    }

    return answer;
}
solution([1,1,3]);

//sol2
function solution(arr)
{
    var answer = [];

    
    console.log(arr)
    var x = arr[0];
    var j = 0;
    var i = 1;
    while(true){
        if(i == arr.length+1){
            break;
        }
        if(x == arr[i]){             
        }else{
            answer[j] = x;
            j++;
            x = arr[i];
        }
        i++;
    } 
    return answer;
}
solution([1,1,3]);

//sol3
function solution(arr)
{
    var answer = [];
    var count = arr.length;

    for(var i=0; i<count; i++) {
        var before = arr[i-1];

        if(before != arr[i]) {
            answer.push(arr[i])
        }
    } 

    return answer;
}

//sol4
function solution(arr)
{
    var answer = [];

    for(var i = 0; i < arr.length; ++i) {
        if(arr[i] != arr[i + 1]) answer.push(arr[i]);
    }

    return answer;
}

console.log(solution([1,1,3,3,0,1,1]))

```  
  

# 3. 3진법 뒤집기    


```javascript
//sol1
function solution(s) {
    let a = s.toString(3).split('');
    let b = [...a];
    for (let i = 0; i < a.length; i++) {
        b[a.length - 1 - i] = a[i];
    }
    return parseInt(b.join(''), 3);

}
const solution = (s) => {
    return parseInt([...s.toString(3)].reverse().join(''), 3);
};

//sol2
const solution = (n) => {
    const arr = [];
    while (n !== 0) {
        arr.unshift(n % 3);
        n = Math.floor(n / 3);
    }
    console.log(arr);
    return arr.reduce((acc, v, i) => {
        console.log(v * Math.pow(3, i));
        return acc + (v * Math.pow(3, i));
    }, 0);
};
console.log(solution(45));

```      


# 4. 두개 뽑아서 더하기    


```javascript
//so1
function solution(n) {
    let a = [];
    n.filter((v, j) => {
        for (let i = 0; i < n.length; i++) {
            if (i !== j) {
                a.push(v + n[i]);
            }
        }
    })
    return a.sort((g, h) => g - h).filter((c, x) => c !== a[x + 1]);

}
//sol2
function solution(n) {
    let a = [];

    for (let i = 0; i < n.length; i++) {
        for (let j = i + 1; j < n.length; j++) {
            a.push(n[i] + n[j]);
        }
    }

    a = [...new Set(a)];
    console.log(a);
    return a.sort((g, h) => g - h);

}
//sol3
function solution(n) {
    let a = [];

    for (let i = 0; i < n.length - 1; i++) {
        for (let j = i + 1; j < n.length; j++) {

            if (a.indexOf(n[i] + n[j] === -1)) a.push(n[i] + n[j]);
            //중복된 값이 없으면 push하기
        }
    }
}
console.log(solution([5, 0, 2, 7]));

```    
  

# 5. 평균 구하기   


```javascript
//sol1
function solution(s) {
    return s.reduce((v, i) => (v + i), 0) / s.length;
}

//sol2
function solution(s) {
    let sum = 0;
    for (let a of s) {
        sum += a;
    }
    return sum / s.length;
}
console.log(solution([5, 5]));

```    




# 6. 문자열을 정수로 바꾸기 


```javascript
function solution(s) {
    console.log(typeof (s + ""));
}
function solution(s) {
    console.log(typeof (s / 1));
}
function solution(s) {
    console.log(typeof (+s));
}
let solution = parseInt;
function solution(s) {

    let a = s - 0;
    console.log(typeof a);
    return a;
}
function solution(s) {

    let a = s * 1;
    console.log(typeof a);
    return a;
}

console.log(solution("-1234"));

```  
  

# 7. 수박수박수박수박수   

```javascript
//sol1
function waterMelon(n) {
    return Array(n).fill().map((v, i) => {
        if (i % 2 === 0) {
            return v = '수';
        } else {
            return v = '박';
        }
    }).join('');
}

//sol2
function waterMelon(n) {
    var result = "";
    for (var i = 0; i < n; i++) {
        result += i % 2 == 0 ? "수" : "박";
    }
    return result;
}

//sol3
const waterMelon = n => {
    console.log('수박'.repeat(n / 2));
    return '수박'.repeat(n / 2) + (n % 2 === 1 ? '수' : '');
}

//sol4
function waterMelon(n) {
    var result = "수박";
    console.log(result.repeat(n - 1).substring(0, 3));
    result = result.repeat(n - 1).substring(0, n);
    //함수를 완성하세요

    return result;
}

console.log(waterMelon(4));

```  
  

# 8. 문자열 내 p와 y의 개수  


```javascript
//sol1
function solution(s) {
    let a = s.split('');
    let b = 0;
    let c = 0;
    a.map((v) => {

        if (v === 'p') b++;
        if (v === 'P') b++;

        if (v === 'y') c++;
        if (v === 'Y') c++;

    });
    if (b === c) {
        return true;
    } else {

        return false;
    }
}

//sol2
function solution(s) {
    let a = s.match(/p/ig);
    let b = s.match(/y/ig);
    console.log(a, b);
    if (a.length === b.length) {
        return true;
    } else {

        return false;
    }

}

//sol3
function solution(s) {
    console.log(!s);
    if (s.match(/p/ig)?.length === s.match(/y/ig)?.length) {

        return true;
    }
}

//sol4
function solution(s) {
    console.log(s.toUpperCase().split('Y'));

    return s.toUpperCase().split('P').length ===s.toUpperCase().split('Y').length;

}
console.log(solution("pPoooyY"));//true
console.log(solution("Pyy"));//false

function solution(s) {
    console.log(s.replace(/p/gi, ''));

    return s.replace(/p/gi, '').length == s.replace(/y/gi, '').length;

}
console.log(solution("abc"));//true

```    


# 9. 두 정수 사이의 합    


```javascript
//sol1
function solution(a, b) {
    var answer = 0;
    let ab = [];
    let sol = [a, b];
    if (a == b) {
        return a;
    } else {
        let c = sol.sort((c, p) => c - p);

        for (let i = 0; i <= c[1]; i++) {
            if (i >= c[0]) {
                ab.push(parseInt(i));
            }
        }
        ab.map((v, i) => {
            return answer += ab[i];
        });
    }
    return answer;
}


//sol2
function solution(a, b) {
    a > b && ([a, b] = [b, a]);
    return Array(b - a + 1).fill().map((v, i) => a + i).reduce((a, c) => a + c);
}


//sol3
function solution(a, b) {
    return (a + b) * ((a > b ? a - b : b - a) + 1) / 2;
}
console.log(solution(3, 6));

```  
  

# 10. 직사각형 별찍기   

```javascript
//sol1
function solution(c, r) {
    return Array(r).fill().map(v => '*'.repeat(c)).join('\n');
}

//sol2
function solution(c, r) {
    // console.log(Array.from(Array(c), () => "*"));
    return Array(r).fill([]).map((_) => Array.from(Array(c), () => "*").join("")).join("\n");
}

```  
  

[프로그래머스](https://programmers.co.kr/learn/challenges?selected_part_id=12079)
