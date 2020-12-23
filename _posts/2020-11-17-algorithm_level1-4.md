---
title: programmers algorithm test Level1 네 번째
author: juyoung
date: 2020-11-17 19:06:00 +0800
categories: [algorithm, Level1]
tags: [algorithm]
---




# 31. 약수의 합


```javascript
//sol1
function solution(n) {
    console.log(Array(Math.floor(n / 2)).fill().map((v, i) => i + 1).reduce((s, c) => n % c ? s : s + c) + n);

}

//sol2
function solution(n) {
    console.log(Array(n).fill().map((v, i) => i + 1).reduce((s, c) => n % c ? s : s + c));

}

//sol3
function solution(n, a = 0, b = 0) {
    console.log(n % a);
    return n <= a / 2 ? b : solution(n, a + 1, b += n % a ? 0 : a);
}

//sol4
function solution(n) {

    let a = 0;
    for (let i = 0; i <= n / 2; i++) {
        if (n % i === 0) a += i;
    }
    console.log(a);

    return a + n;
}
console.log(solution(12));

```


# 32. 체육복


```javascript
//sol1
function solution(n, lost, reserve) {
    let answer = n;
    let student = Array(n).fill(1);
    for (let i = 0; i < student.length; i++) {
        if (lost.includes(i + 1)) {
            student[i] -= 1;
        }
        if (reserve.includes(i + 1)) {
            student[i] += 1;
        }
    }
    for (let i in student) {
        //i=0~4
        if (student[i] == 2 && student[i - 1] == 0) {
            student[i] -= 1;
            student[i - 1] += 1;
        }
        if (student[i] == 0 && student[i - 1] == 2) {
            student[i] += 1;
            student[i - 1] -= 1;
        }
    }
    for (let s of student) { // s=[1,1,1,1,2]
        if (s == 0) answer--;
    }
    return answer;
};

//sol2
function solution(n, lost, reserve) {
    console.log(lost.filter(a => {
        const b = reserve.find(r => {

            return Math.abs(r - a) <= 1
        })
        console.log(b);
        if (!b) return true

        reserve = reserve.filter(r => r !== b)
    }).length)

    return n - lost.filter(a => {
        const b = reserve.find(r => Math.abs(r - a) <= 1)
        if (!b) return true
        reserve = reserve.filter(r => r !== b)
    }).length
}

//sol3
function solution(n, lost, reserve) {      
    return n - lost.filter(a => {
        const b = reserve.find(r => Math.abs(r-a) <= 1)
        if(!b) return true
        reserve = reserve.filter(r => r !== b)
    }).length
}

//sol4
function solution(x, lost, reserve) {
    const students = Array(x).fill(1);
    for (let i of lost) {
        students[i - 1] -= 1;
    }
    for (let i of reserve) {
        students[i - 1] += 1;
    }
    for (let i in students) {
        if (students[i] === 2 && students[i + 1] === 0) {
            students[i] -= 1;
            students[i + 1] += 1;
        }
        if (students[i - 1] === 0 && students[i] === 2) {
            students[i] -= 1;
            students[i - 1] += 1;
        }
    }
    let answer = 0;
    for (let s of students) {
        if (s > 0) {
            answer += 1;
        };
    }
    return answer;
}
console.log(solution(3, [3], [1]));

```  
 

# 33. 완주하지 못한 선수

```javascript
//sol1
function solution(participant, completion) {
    completion.sort().map((c, j) => {
        participant.sort().map((p, i) => {
            if (p == c) {
                participant.splice(i, 1);
            }
        })
    })
    let answer = participant[0];
    return answer;
}

//sol2
function solution(participant, completion) {
    participant = participant.sort();
    completion = completion.sort();
    for (let i in participant) {
        if (participant[i] !== completion[i]) {
            return participant[i];
        }
    }
}

//sol3
function solution(participant, completion) {
    var dic = completion.reduce((obj, t) => {

        return obj[t] = obj[t] ? obj[t] + 1 : 1, obj
    }
    );
    return participant.find(t => {
        console.log(t);
        if (dic[t])
            dic[t] = dic[t] - 1;
        else
            return true;
    });
}

//sol4
const solution = (p, c) => {
    p.sort()
    c.sort()
    while (p.length) {
        let pp = p.pop()

        if (pp !== c.pop()) return pp
    }
}
console.log(solution(["a", "b", "c", "a"], ["b", "a", "c"]));

```    

# 34. K번째 수

```javascript
//sol1
function solution(a, c) {
    let arr = [], i = 0, j = 0, k = 0, b = 0;
    c.forEach((e, h) => {
        i = e[0], j = e[1], k = e[2];
        b = a.slice(i - 1, j).sort((a, b) => a - b)[k - 1];
        arr.push(b);
    });
    return arr;
}


//sol2
function solution(g, h) {
    return h.map(([i, j, k]) => {
        return g.slice(i - 1, j).sort((a, b) => a - b)[k - 1];
    })
}


//sol3
function solution(g, h) {
    let arr = [];
    for (let v of h) {
        arr.push(g.slice(v[0] - 1, v[1]).sort((a, b) => a - b)[v[2] - 1]);
    }
    return arr;
}
console.log(solution([1, 5, 2, 6, 3, 7, 4], [[2, 5, 3], [4, 4, 1], [1, 7, 3]]));

```    

# 35. 모의고사

```javascript
function solution(n) {
    let answer = [];
    let a = [1, 2, 3, 4, 5], a1 = 0;
    let b = [2, 1, 2, 3, 2, 4, 2, 5], b1 = 0;
    let c = [3, 3, 1, 1, 2, 2, 4, 4, 5, 5], c1 = 0;


    for (let i in n) {
        console.log([i % b.length]);
        if (n[i] === a[i % a.length]) a1++;
        if (n[i] === b[i % b.length]) b1++;
        if (n[i] === c[i % c.length]) c1++;
    }
    let max = Math.max(a1, b1, c1);
    if (max === a1) answer.push(1);
    if (max === b1) answer.push(2);
    if (max === c1) answer.push(3);
    return answer;
}

console.log(solution([1, 3, 2, 4, 2, 1, 3, 2, 4, 2]));

```  

# 36. 자릿수 더하기   


```javascript
//sol1
function solution(n) {

    return (n + "").split("").reduce((acc, c) => acc + parseInt(c), 0);

}

//sol2
function solution(n) {
    let a = 0;
    n.toString().split('').map(v => a += parseInt(v));
    console.log(a);

}
console.log(solution(987));

```  

[프로그래머스](https://programmers.co.kr/learn/challenges?selected_part_id=12079)
