---
title: programmers algorithm test Level1 세번째
author: juyoung
date: 2020-11-12 19:06:00 +0800
categories: [algorithm, Level1]
tags: [algorithm]
---


# 21. 최대공약수와 최소공배수


```javascript
//sol1
function solution(n, m) {
    const greatestCommonDivisor = (a, b) => {
        if (b === 0) return a;
        return greatestCommonDivisor(b, a % b);
    }
    const leastCommonMultiple = (a, b) => (a * b) / greatestCommonDivisor(a, b);
    return [greatestCommonDivisor(n, m), leastCommonMultiple(n, m)];

}

//sol2
function solution(a, b) {
    let r, ab;
    for (ab = a * b; r = a % b; a = b, b = r) {
    }
    console.log(ab, a, b);
    console.log(r);
    return [b, ab / b];
}

console.log(solution(2, 5));


```


# 22. 나누어 떨어지는 숫자 배열


```javascript
//sol1
function solution(arr, divisor) {
    var answer = [], temp=[], save=0,fir,sec;
    
    for(let i=0;i<arr.length;i++){
     if( arr[i]% divisor == 0 ){
        temp.push(arr[i]);
     }
    
     
    }
    for(let i=0;i<temp.length-1;i++){
        fir = i;
        for( sec=fir+1;sec<temp.length;sec++){
           
            if(temp[fir]>temp[sec]){
                fir = sec;
            }
        }
        save = temp[fir];
        temp[fir] = temp[i];
        temp[i]=save;
    }
    console.log(temp);
    if(temp.length != 0){
        answer = temp;
    }else{
        answer = [-1];
    }
    return answer;
}
//console.log(solution([5, 9, 7, 10],4));
solution([5, 9, 7, 10],4)

function sol(arr,divisor){
const ans = arr.filter(el => el% divisor === 0);
return ans.length ? ans.sort((p,c) => p - c) : [-1];
}
console.log(sol([5, 9, 7, 10],5));


```    


# 23. 문자열 내 마음대로 정렬하기

```javascript
//sol1
function solution(strings, n) {
    strings.sort((p, c) => {
        return strings.sort((p, c) => p[n] === c[n] ? p.localeCompare(c) : p[n].localeCompare(c[n]));
    });

};

//sol2
function solution(strings, n) {
    return strings.sort((a, b) => {
        const chr1 = a.charAt(n);

        const chr2 = b.charAt(n);
        console.log('chr1', chr1, chr2, (chr1 > chr2));
        console.log('chr2', chr1, chr2, (chr1 < chr2));
        // console.log('(a > b)', a, b, (a > b));
        // console.log('(a < b)', a, b, (a < b));
        if (chr1 == chr2) {
            return (a > b) - (a < b);
        } else {
            return (chr1 > chr2) - (chr1 < chr2);
        }
    })
}

//sol3
function solution(strings, n) {
    var answer = [];
    for (var i = 0; i < strings.length; i++) {
        var chu = strings[i][n];
        strings[i] = chu + strings[i];
    }

    strings.sort();
    console.log(strings);
    for (var j = 0; j < strings.length; j++) {
        strings[j] = strings[j].replace(strings[j][0], "");

        answer.push(strings[j])
    }

    return answer;
}
console.log(solution(['bed', 'aun', 'car'], 1));

```     


# 24. 문자열 내림차순으로 배치하기  

```javascript
function solution(s) {
  return s.split('').sort((prev, cur) => cur.charCodeAt() - prev.charCodeAt()).join('');
}
```    


# 25. 서울에서 김서방 찾기  

```javascript
//sol1
function solution(s) {
    let a = s.findIndex((v) => v === 'Kim');

    console.log(b);
}

//sol2
function findKim(seoul){
    var idx = 0;
          for (var i = 0; i < seoul.length; i++){
         if (seoul[i] === 'Kim'){
           idx += i;
           break;
        }
        }

    return "김서방은 " + idx + "에 있다";
  }

  //sol3
function solution(s) {
    return `김서방은 ${s.indexOf("Kim")}에 있다`;
}

//sol4
function solution(s) {
    var idx = 0;
    var findSize = s.length;
    for (var i = 1; findSize > i; i++) {
        if (s[i] == "Kim") {
            idx = i;
        }
    }
    return "김서방은 " + idx + "에 있다";
}
console.log(solution(['Kim', 'Jane', 'Cho',]));

```  
  

# 26. 소수 찾기

```javascript
//sol1
function solution(s) {
    let a = [], b = [];

    for (let i = 0; i <= s; i++) {
        a.push(i);

        if (a[i] % 2 !== 0 && a[i] % 3 !== 0 && a[i] > 3) {

            b.push(a[i]);


        }
    }
    return b.length + 2;
}

//sol2
function solution(n) {
    const primes = [];
    for (let j = 2; j <= n; j++) {
        let isPrime = true;
        const sqrt = Math.sqrt(j);
        console.log('sqrt', sqrt);
        for (let i = 0; primes[i] <= sqrt; i++) {
            console.log(i, primes[i], 'j', j);
            if (j % primes[i] === 0) {
                isPrime = false;
                break;
            }

        }
        if (isPrime) {
            primes.push(j);
        };

    }
    return primes.length;
}

//sol3
  function solution(n, start = 2, primes = [], count = 0) {
    if (start > n) return count;
    const sqrt = Math.sqrt(start);
    const isPrime = primes.every(v => start % v);
    if (isPrime) primes.push(start);
    return solution(n, start + 1, primes, count + (isPrime ? 1 : 0));
}

//sol4
function solution(n) {
    let range = Array(n - 1).fill().map((v, i) => i + 2);
    for (let i = 0; i < range.length; i++) {
        range = range.filter(v => v === range[i] || v % range[i]);
    }
    return range.length;
}

//sol5
const solution = (n) => {
    let arr = [];
    for (let i = 1; i <= n; i++) arr.push(i);

    for (let i = 1; i * i < n; i++) {
        console.log(i, 'arr[i]', arr[i]);
        if (arr[i]) {
            let num = arr[i];
            console.log('num', num);
            for (let j = 2 * num; j <= n; j += num) {
                console.log('j', j);
                arr[j - 1] = 0;
            }
        }
    }
    let answer = arr.filter((number) => {
        console.log(number, 'number');
        return number;
    });
    console.log('answer', answer);
    answer.shift();
    console.log('shift', answer);
    return answer.length;
}

//sol6
const solution = (n) => {
    let arr = [];
    for (let i = 1; i <= n; i++) arr.push(i);

    for (let i = 1; i * i < n; i++) {
        console.log(i, 'arr[i]', arr[i]);
        if (arr[i]) {
            let num = arr[i];
            console.log('num', num);
            for (let j = num * num; j <= n; j += num) {
                console.log('j', j);
                arr[j - 1] = 0;
            }
        }
    }
    let answer = arr.filter((number) => {
        console.log(number, 'number');
        return number;
    });
    console.log('answer', answer);
    answer.shift();
    console.log('shift', answer);
    return answer.length;
}

console.log(solution(100));

```  
  

# 27. x만큼 간격이 있는 n개의 숫자
 

```javascript
//sol1
function solution(x, n) {
    return [...Array(n).keys()].map(v => (v + 1) * x);
}

//sol2
function solution(x, n) {
    return Array(n).fill(x).map((v, i) => (i + 1) * v);
}

//sol3
function solution(x, n) {
    let arr = [];
    for (let i = 1; i <= n; i++) {
        arr.push(x * i);
    }
    return arr;
}
console.log(solution(4, 3));
```
  

# 28. 제일 작은 수 제거하기 

```javascript
//sol1
function solution(s) {
    if (s.length === 1) return [-1];
    let arr = [];
    s.forEach((v, i) => {
        if (v < s[i - 1]) {
            arr.push(v);
        }
    });
    let a = arr.sort((p, c) => p - c).splice(0, 1)[0];
    return s.filter((v) => v !== a);

};

//sol2
function solution(arr) {
    // const minValue = Math.min.apply(null, arr);
    // console.log(minValue);
    const min = Math.min(...arr);
    const r = arr.filter(v => v !== min);
    return r.length ? r : [-1];

}

console.log(solution([1]));

```  
  

# 29. 정수 제곱근 판별  

```javascript
//sol1
function solution(s) {
    let a = Math.sqrt(s);

    return Number.isInteger(a) ? (Math.pow(a + 1, 2)) : -1;

}

//sol2
function solution(s) {
    let a = Math.sqrt(s);

    return Number.isInteger(a) ? (a + 1) ** 2 : -1;

}
console.log(solution(121));

```

# 30. 정수 내림차순으로 배치하기


```javascript
//sol1
function solution(n) {
    const newN = n + "";
    const newArr = newN.split('').sort().reverse().join('');
    console.log(typeof +newArr);
    return +newArr;
}

//sol2
function solution(n) {
    var r = 0, e = 0, arr = [];

    do {
        e = n % 10;
        //console.log(e);
        // 정렬
        if (arr.length == 0) {
            arr.push(e);
            console.log('1', arr);
        }
        else for (var i = 0, len = arr.length; i < len; i++) {
            if (arr[i] <= e) {
                console.log('a', arr[i], 'e', e, arr.splice(i, 0, e)); break;
            }
            if (i == len - 1) {

                arr.push(e);
                console.log('3', arr);
            }
        }
    } while (n = Math.floor(n / 10), n > 0);

    return parseInt(arr.join(""));
}

//sol3
function solution(n) {
    return parseInt(n.toString().split('').sort((a, b) => b - a).join(''));
    return n.toString().split('').sort((a, b) => b - a).join('').toNumber();

}

//sol4
function solution(n) {
    var answer = "";
    n = n + "";
    var emptyArray = [];
    for (var i = 0; i < n.length; i++) {
        emptyArray.push(n[i]);
    }
    for (var j = 0; j < emptyArray.length; j++) {
        if (emptyArray[j] < emptyArray[j + 1]) {
            var temp = emptyArray[j];
            emptyArray[j] = emptyArray[j + 1];
            emptyArray[j + 1] = temp;
            j = -1;
        }
        console.log(emptyArray);
    }
    for (var k = 0; k < emptyArray.length; k++) {
        answer += emptyArray[k];
    }
    answer = Number(answer);
    return answer;
}

console.log(solution(118372));

}

```  
  
   
[프로그래머스](https://programmers.co.kr/learn/challenges?selected_part_id=12079)
