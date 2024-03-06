
# JS

```
    __∧_∧__    ~~~~~
　／(*´O｀)／＼
／|￣∪∪￣|＼／
　|＿＿ ＿|／
```

---


**class object inheritance**

```js
// OOJS 1
// OOJS 2
function Shape(name, sides, sideLength) {
    this.name = name;
    this.sides = sides;
    this.sideLength = sideLength;
    this.calcPerimeter = function(){
        let perimeter = this.sides*this.sideLength
        console.log(perimeter);
    };
}
let square = new Shape('square',4,5);
let triangle = new Shape('triangle',3,3);


// OOJS 3
class Shape {
    constructor (name, sides, sideLength) {
        this.name = name;
        this.sides = sides;
        this.sideLength = sideLength;
        this.calcPerimeter = function(){
            let perimeter = this.sides*this.sideLength
            console.log(perimeter);
        };
    };
}

class Square extends Shape {
    constructor (sideLength) {
        super();
        this.name = 'square';
        this.sides = 4;
        this.sideLength = sideLength;
    };
    calaArea() {
        console.log(this.sides*this.sideLength);
    };
}

let square = new Square(5);
square.calaArea()
square.calcPerimeter()
```
