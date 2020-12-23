---
title: programmers algorithm test Level1 두번째
author: juyoung
date: 2020-10-26 18:46:00 +0800
categories: [algorithm, Level1]
tags: [algorithm]
---


# 11. 콜라츠 추측


```javascript
//sol1
function solution(n, count = 0) {
    // let a = 0;
    // console.log(a++, ++a);
    return n === 1 ? (count > 500 ? -1 : count) : solution(n % 2 ? n * 3 + 1 : n / 2, ++count);

}

//sol2
function solution(n) {

    let i = 0;
    while (n > 1) {
        n % 2 === 0 ? n = n / 2 : n = n * 3 + 1;
        i++;
    }
    if (i > 500) i = -1;
    console.log(i);


}
console.log(solution(626331));

```


# 12. 하샤드 수  


```javascript
//sol1
// 
function solution(x, i = 0, sum = 0) {

    return String(x).length == i ? x % sum == 0 : solution(x, i + 1, sum + String(x)[i] * 1);
}
//sol2
function solution(x) {
    let answer = false;
    let n = x.toString().split('').map(v => +v);
    let d = n.reduce((a, c) => a + c);
    if (x % d === 0) answer = true;
    return answer;
}

console.log(solution(18));


```  

# 13. 핸드폰 번호 가리기   

```javascript
//sol1
function solution(phone_number) {


    let a = phone_number.split('');

    a.map((v, i) => {
        if (i < a.length - 4) {
            a[i] = "*";
        }
    });

    return a.join('');

}

//sol2
function solution(s) {
    return s.replace(/\d(?=\d{4})/g, "*");
}

//sol3
function solution(s) {
    let result = '';
    for (let i = 0; i < s.length; i++) {
        result += i < s.length - 4 ? "*" : s.charAt(i);
    }

    return result;


}

//sol4
function solution(s) {
    let a = Array(s.length - 3).join("*");


    let b = s.substring(s.length - 4);
    console.log(b);
    return a + b;
}

//sol5
function solution(s) {
    var _str = '';
    for (var i = 0; i < s.length - 4; i++) {
        _str += '*';
    }
    var result = s.replace(s.substr(0, s.length - 4), _str);
    return result;
}

//sol6
function solution(s) {
    console.log(s.slice(-4));
    return "*".repeat(s.length - 4) + s.slice(-4);
}
console.log(solution("01033334444"));

```    

# 14. 행렬의 덧셈    

```javascript
//sol1
function solution(arr1, arr2) {

    return arr1.map((arr, i) => arr.map((v, j) => v + arr2[i][j]));

}

//sol2
function solution(A, B) {
    var answer = Array();
    for (var i = 0; i < A.length; i++) {
        answer[i] = [];

        for (var j = 0; j < A[i].length; j++) {
            answer[i][j] = A[i][j] + B[i][j];
        }
    }
    return answer;
}

//sol3
function solution(arr1, arr2) {

    let arr = [[], []], a = [[], []];

    for (let i = 0; i <= arr1.length - 1; i++) {
        for (let j = 0; j <= arr1.length - 1; j++) {
            arr[i].push(arr1[i][j] + arr2[i][j]);
        }
        arr[i].filter((v) => {
            if (!isNaN(v)) return a[i].push(v);

        });

    }

return a;

}
console.log(solution([[1], [2]], [[3], [4]]));

```  

# 15. 시저암호  

```javascript
//sol1
function solution(s, n) {

    let a = s.split('').map(v => v.charCodeAt() + n);
    let b = '';

    a.map((v, i) => {

        if (v > 90 && v < 97) {
            a[i] = v - 90 + 64;
        }
        if (v > 122) {
            a[i] = v - 122 + 96;
        }
        if (v < 65) a[i] = 32;
        b += String.fromCharCode(a[i]);

    });
    return b;
}

//sol2
function solution(s, n) {
    return s.split('').map((l) => {
        console.log(l.charCodeAt() <= 90);
        return l === ' '
            ? l
            : l.charCodeAt() + n > 122 || (l.charCodeAt() <= 90 && l.charCodeAt() + n > 90)
                ? String.fromCharCode((l.charCodeAt() + n) - 26)
                : String.fromCharCode(l.charCodeAt() + n);
    }).join('');
}
console.log(solution("a B z", 4));

```  

# 16. 이상한 문자 만들기  

```javascript
//sol1
function solution(s) {
    let arr = [], p = [];
    s.split(' ').forEach((el) => {
        arr = el.split('').map((v, i) => {
            if (i % 2 === 0) {
                return v.toUpperCase();
            } else {
                return v.toLowerCase();
            }
        });
        p.push(arr.join(''));
    });
    return p.join(' ');


}

//sol2
const solution = (s) => {
    return s.toUpperCase().replace(/(\w)(\w)/g, function (a) { return a[0].toUpperCase() + a[1].toLowerCase(); })
};

//sol3
function solution(s) {
    return s.split(' ').map(w => (
        w.split('').map((v, i) => (i % 2 ? v.toLowerCase() : v.toUpperCase())).join('')
    )).join(' ');
}

//sol4
function solution(s) {
    let answer = '';
    for (let word of s.split(' ')) {
        console.log(word);
        for (let i in word) {
            console.log(i);
            answer += word[i][parseInt(i) % 2 == 0 ? 'toUpperCase' : 'toLowerCase']();
        }
        // console.log(answer);
        answer += ' ';
    };
    // console.log(answer.split(''));
    return answer.slice(0, -1);
}

//sol5
function solution(s) {
    let result = '', num = 0;
    for (let i in s) {
        if (s.charAt(i) == " ") {
            num = 0;
            result += ' ';
            continue;
        } else if (num % 2 == 0) {
            result += (s.charAt(i)).toUpperCase();
            num++;
        } else {
            result += (s.charAt(i)).toLowerCase();
            num++;
        }

    }
    return result;
}

console.log(solution("try hello world"));

```  
  
  
# 17. 짝수와 홀수 

```javascript
function solution(n) {
    console.log(0 % 2 == true);
    return n % 2 ? "Odd" : "Even";
}
console.log(solution(0));
```
  

# 18. 문자열 다루기 기본  

```javascript
//sol1
function solution(s) {
    if (isNaN(parseInt(s)) && [4, 6].includes(s.length)) {
        return false;
    } else {
        return true;
    }
}

//sol2
function solution(s) {
    function solution(s) {
        return /^[0-9]+$/.test(s) && [4, 6].includes(s.length);
    }
}
console.log(solution('ee22'));
console.log(/^[0-9]+$/.test('22'));//true

```  
  

# 19. 가운데 글자 가져오기  

```javascript
//sol1
function solution(s) {
   
  return s.substr(Math.ceil(s.length/2)-1, s.length%2 !== 0 ? 1 : 2);

}

console.log(solution("qwer"));

//sol2
function solution(s) {
    var length = s.length;
    var answer = '';

    if(!(s.length >0 && s.length<100)) {
        return;
    }

    if(length % 2 != 0) {
        answer = s.slice(length/2, length/2 +1);
        console.log(length/2 , length/2 +1);
    }else {
        answer = s.slice(length/2 -1, length/2 +1);
        console.log(length/2 -1, length/2 +1);
    }

    return answer;

}

let s = 'abcd';
console.log(solution(s));

```    


# 20. 자연수 뒤집어 배열로 만들기  

```javascript
//sol1
function solution(n) {
    var arr = [];
    do {
        console.log(n % 10);
        arr.push(n % 10);
        n = Math.floor(n / 10);
    }
    while (n > 0);
    return n.toString().split('').reverse().map(v => parseInt(v));

}

//sol2
function solution(n) {
    console.log(n.toString().split(''));
    return n.toString().split('').reverse().map(v => +v);
}

//sol3
function solution(n) {
    let arr = [];
    n.toString().split('').map((v, i) => arr.push(parseInt(v))).reverse();
    console.log(arr);
}
console.log(solution(19875));

```  

[프로그래머스](https://programmers.co.kr/learn/challenges?selected_part_id=12079)
