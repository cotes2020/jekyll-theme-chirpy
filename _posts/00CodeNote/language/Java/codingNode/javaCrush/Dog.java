package JAVA.javaCrush;

public class Dog {

    // private : means only accessible within this class
    protected String name;   // only this package can see it
    protected int age;

    // static all change together
    protected static int count = 0;

    // this: look for the attribute

    public Dog(String name, int age) {
        this.name = name;
        this.age = age;

        // static variable
        // this.count +=1; all well works
        Dog.count += 1;
        Dog.display();        // static can be called directlly
        // Dog.display2();    // not static need to be called by a instance
        this.display2();

        // add2();
        // speak();
    }

    public static void display() {
      System.out.println("I'm a dog!");
      // but static can call this.age
      // cannot specific a instance
    }

    public void display2() {
        System.out.println("I'm a dog!");
    }


    public void speak() {
        System.out.println("I am " + this.name + " age: "+ this.age);
    }


    public int getAge() {
        return this.age;
    }


    public void setAge(int age) {
        this.age = age;
    }


    // private int add2() {
    //     return this.age + 2;
    // }

    // private void notouch() {
    //     // private: no other class or package able to use it, but this class
    // }
}
