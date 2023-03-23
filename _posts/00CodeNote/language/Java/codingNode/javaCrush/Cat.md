// package JAVA.javaCrush;

// public class Cat extends Dog {
//     // extends : Cat inherit all from dog

//     public int food;

//     public Cat(String name, int age, int food) {
//         super(name, age);  // super call the parent method
//         this.food = food;
//     }


//     public Cat(String name, int age) {
//         super(name, age);
//         this.food = 1000; // asume the food to 500, if no input
//     }


//     public Cat(String name) {
//         super(name, 1);  // asume the age to 0, if no input
//         this.food = 1000; // asume the food to 500, if no input
//     }


//     public void speak() {
//         System.out.println("meow! "+ this.name + " " + this.age + ". Food amount: " + this.food);
//     }


//     public void eat(int x) {
//         this.food -= x;
//         System.out.println("Fed: " + x + " Food amount: " + this.food);
//     }
// }
