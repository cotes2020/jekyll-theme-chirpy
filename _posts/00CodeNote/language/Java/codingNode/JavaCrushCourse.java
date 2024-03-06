
package JAVA.javaCrush;

// import java.util.ArrayList;
// import java.util.Arrays;
// import java.util.HashMap;
// import java.util.HashSet;
// import java.util.LinkedHashMap;
// import java.util.LinkedHashSet;
// import java.util.LinkedList;
// import java.util.Map;
// import java.util.Scanner;
// import java.util.Set;
// import java.util.TreeMap;
// import java.util.TreeSet;

/**
 * JavaCrushCourse
 */
public class JavaCrushCourse {

    public static void main(String[] args) {

        // constructor
        // specifies how to create objects of this class.
        // code that gets run when an object is created to initialize that object.
        // constructor looks like a function but has no return type.
        // And is named the same as the class.

        // abstraction.
        // The details of how a object works can remain hidden from us as long as we know what it does.

        // 0. array
        // int[] arr = {1,2,3,4,5};
        // String[] names = new String[5];


        // 0. string
        // String s1 = "Hello";
        // String s2 = " World!";
        // String s3 = s1.concat(s2);
        // System.out.println(s3);       // prints "Hello World!"
        // System.out.println(s1.equals("Hello"));     // prints true
        // System.out.println(str.indexOf("l"));       // prints 2
        // System.out.println(str.charAt(0));          // prints 'T'
        // String str = "Hello World!";
        // String uppercase = str.toUpperCase();
        // // uppercase = "HELLO WORLD!"
        // String lowercase = str.toLowerCase();
        // lowercase = "hello world!"
        // s.substring(startindex, endindex)


        // 1. simple for loop
        // for (int i = 0; i < arr.length; i++) {
        //     System.out.println(arr[i]);
        // }


        // 2. usual count method
        // int count = 0;
        // for (int element:arr) {
        //     System.out.println(count + " " + element);
        //     count++;
        // }


        // 3. use break
        // for (int i = 0; i < names.length; i++) {
        //     System.out.print("Input: ");
        //     String input = sc.nextLine();
        //     names[i] = input;
        //     break
        // }
        // for (String n:names) {
        //     System.out.println(n);
        //     if (n.equals("tim")) {
        //         break
        //     }
        // }



        // 4. while: when dont know the number of times to loop
        // Scanner sc = new Scanner(System.in);
        // System.out.println("Type number: ");
        // int x = sc.nextInt();
        // int count =0
        // while (x != 10) {
        //     System.out.println("Type 10");
        //     System.out.println("Type number: ");
        //     int x = sc.nextInt();
        //     count++;
        // }
        // System.out.println("you tried"+count+"times");


        // 5. do while:
        // Scanner sc = new Scanner(System.in);
        // int x;
        // do {
        //     System.out.println("Type number: ");
        //     x = sc.nextInt();
        // } while (x != 10);



        // 6. set: unorder collection of unique element, fast, just want to know if it exist
        // Set<Integer> t = new HashSet<Integer>();
        // t.add(5);
        // t.add(17);
        // t.add(5);
        // t.add(9);
        // t.remove(9);
        // t.clear();  // remove everything
        // boolean y = t.contains(5);
        // t.isEmpty();
        // int x = t.size();
        // System.out.println(y);


        // 7. treeset: ordered collection of unique element
        // Set<Integer> t2 = new TreeSet<Integer>();
        // t2.add(5);
        // t2.add(17);
        // t2.add(5);
        // t2.add(9);
        // System.out.println(t2);


        // 7. LinkedHashSet
        // Set<Integer> t2 = new LinkedHashSet<Integer>();
        // t2.add(5);
        // t2.add(17);
        // t2.add(5);
        // t2.add(9);
        // System.out.println(t2);


        // 8. lists:
        // ArrayList<Integer> t = new ArrayList<Integer>();  // no size need
        // t.add(5);
        // t.remove(5)
        // t2.get(index);
        // t.get(0);
        // t2.set(index.value);
        // t.size();
        // t.isEmpty();
        // t.subList(fromIndex, toIndex)
        // System.out.println(t.subList(1,3));



        // 8. LinkedList:
        // LinkedList<Integer> t = new LinkedList<Integer>();
        // t.add(5);
        // t.remove(5);
        // t2.get(index);
        // t.get(0);
        // t2.set(index, value);
        // t.size();
        // t.isEmpty();
        // t.subList(fromIndex, toIndex)
        // System.out.println(t.subList(1,3));


        // 9. HashMap: unique key, no order, so fast.
        // Map m = new HashMap();
        // HashMap<String, integer> map = new HashMap<String, integer>();
        // m.put("tim", 5);
        // m.put(11, 5);
        // m.get("tim");
        // map.containsKey("tim");
        // System.out.println(m);
        // System.out.println(m.get("tim"));
        // for(String s : map.keySet());
        // {sd=5, df=5, tim=5}


        // 10. TreeMap: key has to be same type, has order.
        // Map m = new TreeMap();
        // m.put("tim", 5);
        // m.put("sd", 5);
        // System.out.println(m);
        // System.out.println(m.get("tim"));
        // {sd=5, tim=5}


        // 10. LinkedHashMap: same as input order.
        // Map m = new LinkedHashMap();
        // m.put("tim", 5);
        // m.put("sd", 5);
        // System.out.println(m);
        // System.out.println(m.get("tim"));
        // {tim=5, sd=5}


        // 11. HashMap: unique key, no order, so fast.
        // Map m = new HashMap();
        // m.put("tim", 5);
        // m.put("sd", 5);
        // m.put("a", "b");
        // m.clear();
        // m.isEmpty();
        // m.size();
        // m.keySet();
        // m.containsValue(5);
        // m.containsKey("a");
        // m.remove(key)
        // System.out.println(m);
        // System.out.println(m.get("tim"));
        // System.out.println(m.values());



        // 12. map example
        // Map m = new HashMap();
        // String wrds = "hello";
        // for (char x:wrds.toCharArray()) {
        //     if (m.containsKey(x)) {
        //         int old = (int)m.get(x);
        //         m.put(x, old+1);
        //     }
        //     else {
        //         m.put(x, 1);
        //     }
        // }
        // System.out.println(m);


        // 13. sort
        // int[] x = {1,2,3,4,6,3,2,5,3,1,4,6,0};
        // Arrays.sort(x, 3,6);
        // for (int i:x) {
        //     System.out.print(i+",");
        // }


        // tim("Tim!!!!!", 4);
        // System.out.println(add2(6));
        // System.out.println(strth("bob"));


        // 15. class
        // create public class Dog
        // Dog tim = new Dog("tim", 9);
        // Dog bill = new Dog("bill", 2);
        // tim.speak();
        // tim.setAge(5);
        // tim.add2();  private method is not usable
        // System.out.println(tim.getAge());



        // 16. inherit
        // Cat tim2 = new Cat("tim", 3, 500);
        // Cat joe = new Cat("joe", 2);
        // Cat bob = new Cat("bob");
        // tim2.speak();
        // joe.speak();
        // bob.speak();
        // tim2.eat(2)
        // class Animal { // Parent Class: Animal class members}
        // class Dog extends Animal { // Child Class: Dog inherits traits from Animal }
        // class Animal {    // Parent class
        //     String sound;
        //     Animal(String snd){this.sound = snd;}
        //     protected double gpa;    // any child class of Student can access gpa
        //     final protected boolean isStudent() {
        //         return true;      // any child class of Student cannot modify isStudent()
        //     }
        // }
        // class Dog extends Animal {    // Child class
        //     // super() method can act like the parent constructor inside the child class constructor.
        //     Dog() {
        //         super("woof");
        //     }
        //     // alternatively, we can override the constructor completely by defining a new constructor.
        //     Dog() {
        //         this.sound = "woof";
        //     }
        // }



        // 16.
        // Main() method in Java
        // In simple Java programs, you may work with just one class and one file. However, as your programs become more complex you will work with multiple classes, each of which requires its own file. Only one of these files in the Java package requires a main() method, and this is the file that will be run in the package.



        // 17. static variable
        // System.out.println(Dog.count);  // 2
        // Dog.count = 7;
        // System.out.println(tim.count);  // 7


        // 18. static method


        // 19. overloading methods & object comparisons
        // check quality between objects.
        // Student joe = new Student("Joe");
        // Student bill = new Student("Bill");
        // Student tim = new Student("Tim");
        // System.out.println(joe.equals(bill));    // the actual object is 2 different object in memory.
        // System.out.println(joe.compareTo(tim));  // compare 2 object
        // System.out.println(tim);   // the memory location address of the object  JAVA.javaCrush.Student@36aa7bc2
        // System.out.println(tim.toString());



        // 20. inner classes
        // 2 way to access an inner class
        // OuterClass out = new OuterClass();
        // OuterClass.InnerClass in = out.new InnerClass();   // if is public
        // out.inner();
        // System.out.println();
        // in.display();



        // 21. interface
        // Car ford = new Car();
        // ford.speedUp(10);
        // ford.changeGear(2);
        // ford.display();
        // static method
        // int x = Vehicle.math(9);
        // System.out.println(x);
        // like create a Math interface with calculate
        // then when use, just Math.yourmethod


        // 22. enums
        // Level lvl = Level.LOW;
        // Level[] arr = Level.values();
        // for (Level e:arr) {
        //     System.out.println(e);
        // }
        // System.out.println(lvl.values());    // memory location
        // String en = lvl.toString();
        // if (lvl == Level.LOW) {
        //     System.out.println(lvl);
        // } else if (lvl == Level.HIGH) {
        //     System.out.println(lvl);
        // } else {
        //     System.out.println(lvl);
        // }
        // System.out.println(lvl.getLvl());
        // System.out.println(Level.valueOf("LOW"));
        // lvl.setLvl(5);
        // System.out.println(lvl.getLvl());

    }


    // 14. Object
    // void : not return anything but do something

    // public static void tim() {
    //     System.out.println("Tim!!");
    // }
    // public static void tim(String str, int x) {
    //     for (int i = 0; i < x; i++) {
    //         System.out.println(str);
    //     }
    // }
    // public static int add2(int x) {
    //     return x + 2;
    // }
    // public static String strth(String x) {
    //     return "hello" + x;
    // }


    // DirectoryResource dr = new DirectoryResource();
    // for(File f : dr.selectedFile()){
    //     FileResource fr = new FileResource(f);
    //     CSVParser parser = fr.getCSVParser();
    //     for(CSVRecord record : parser){
    //         double currenttemp = current.get("TemperatureF");
    //     }
    // }


    return (Collection<Car>) CollectionUtils.select( listOfCars, (arg0) -> {return Car.SEDAN == ((Car)arg0).getStyle();} );
    // is equivalent to
    return (Collection<Car>) CollectionUtils.select(listOfCars, new Predicate() {
        public boolean evaluate(Object arg0) {
            return Car.SEDAN == ((Car)arg0).getStyle();
        }
    });

    Runnable r = ()-> System.out.print("Run method");
    // is equivalent to
    Runnable r = new Runnable() {
            @Override
            public void run() {
                System.out.print("Run method");
            }
    };





}
