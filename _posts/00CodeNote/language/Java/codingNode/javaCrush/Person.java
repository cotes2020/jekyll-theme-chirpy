
// public class Person {

//     public enum Sex {
//         MALE, FEMALE
//     }

//     String name;
//     LocalDate birthday;
//     Sex gender;
//     String emailAddress;


//     public int getAge() {
//         // ...
//     }

//     public void printPerson() {
//         // ...
//     }

// }


// // members are stored in a List<Person> roster instance.

// // Approach 1: Create Methods That Search for Members That Match One Characteristic

// public static void printPersonsOlderThan(List<Person> roster, int age) {
//     for (Person p : roster) {
//         if (p.getAge() >= age) {
//             p.printPerson();
//         }
//     }
// }


// // Approach 2: Create More Generalized Search Methods
// // prints members within a specified range of ages:

// public static void printPersonsWithinAgeRange(List<Person> roster, int low, int high) {
//     for (Person p : roster) {
//         if (low <= p.getAge() && p.getAge() < high) {
//             p.printPerson();
//         }
//     }
// }


// // Approach 3: Specify Search Criteria Code in a Local Class
// // prints members that match search criteria that you specify:
// interface CheckPerson {
//     boolean test(Person p);
// }

// public static void printPersons(List<Person> roster, CheckPerson tester) {
//     for (Person p : roster) {
//         if (tester.test(p)) {
//             p.printPerson();
//         }
//     }
// }

// class CheckPersonEligibleForSelectiveService implements CheckPerson {
//     public boolean test(Person p) {
//         return p.gender == Person.Sex.MALE &&
//             p.getAge() >= 18 &&
//             p.getAge() <= 25;
//     }
// }

// printPersons(
//     roster,
//     new CheckPersonEligibleForSelectiveService()
// );




// // Approach 4: Specify Search Criteria Code in an Anonymous Class
// // One of the arguments of the following invocation of the method printPersons is an anonymous class that filters members that are eligible for Selective Service in the United States: those who are male and between the ages of 18 and 25:

// printPersons(
//     roster,
//     new CheckPerson() {
//         public boolean test(Person p) {
//             return p.getGender() == Person.Sex.MALE
//                 && p.getAge() >= 18
//                 && p.getAge() <= 25;
//         }
//     }
// );



// // Approach 5: Specify Search Criteria Code with a Lambda Expression
// // The CheckPerson interface is a functional interface. A functional interface is any interface that contains only one abstract method. (A functional interface may contain one or more default methods or static methods.) Because a functional interface contains only one abstract method, you can omit the name of that method when you implement it. To do this, instead of using an anonymous class expression, you use a lambda expression, which is highlighted in the following method invocation:

// printPersons(
//     roster,
//     (Person p) -> p.getGender() == Person.Sex.MALE
//         && p.getAge() >= 18
//         && p.getAge() <= 25
// );










// // Syntax of Lambda Expressions


// new CheckPerson() {
//     public boolean test(Person p) {
//         return p.getGender() == Person.Sex.MALE
//             && p.getAge() >= 18
//             && p.getAge() <= 25;
//     }
// }


// p -> p.getGender() == Person.Sex.MALE
//     && p.getAge() >= 18
//     && p.getAge() <= 25


// p -> {
//     return p.getGender() == Person.Sex.MALE
//         && p.getAge() >= 18
//         && p.getAge() <= 25;
// }

// // A return statement is not an expression; in a lambda expression, you must enclose statements in braces ({}). However, you do not have to enclose a void method invocation in braces. For example, the following is a valid lambda expression:

// email -> System.out.println(email)


// // consider lambda expressions as anonymous methods—methods without a name.
// // The following example, Calculator, is an example of lambda expressions that take more than one formal parameter:

// public class Calculator {

//     interface IntegerMath {
//         int operation(int a, int b);
//     }

//     public int operateBinary(int a, int b, IntegerMath op) {
//         return op.operation(a, b);
//     }

//     public static void main(String... args) {
//         Calculator myApp = new Calculator();
//         IntegerMath addition = (a, b) -> a + b;
//         IntegerMath subtraction = (a, b) -> a - b;
//         System.out.println("40 + 2 = " + myApp.operateBinary(40, 2, addition));
//         System.out.println("20 - 10 = " + myApp.operateBinary(20, 10, subtraction));
//     }
// }


// // Accessing Local Variables of the Enclosing Scope
// // Like local and anonymous classes, lambda expressions can capture variables; they have the same access to local variables of the enclosing scope. However, unlike local and anonymous classes, lambda expressions do not have any shadowing issues (see Shadowing for more information). Lambda expressions are lexically scoped. This means that they do not inherit any names from a supertype or introduce a new level of scoping. Declarations in a lambda expression are interpreted just as they are in the enclosing environment. The following example, LambdaScopeTest, demonstrates this:


// import java.util.function.Consumer;

// public class LambdaScopeTest {

//     public int x = 0;

//     class FirstLevel {

//         public int x = 1

//         void methodInFirstLevel(int x) {
//             // The following statement causes the compiler to generate
//             // the error "local variables referenced from a lambda expression
//             // must be final or effectively final" in statement A:
//             // x = 99;

//             Consumer<Integer> myConsumer = (y) -> {
//                 System.out.println("x = " + x); // Statement A
//                 System.out.println("y = " + y);
//                 System.out.println("this.x = " + this.x);
//                 System.out.println("LambdaScopeTest.this.x = " + LambdaScopeTest.this.x);
//             };

//             myConsumer.accept(x);
//         }
//     }

//     public static void main(String... args) {
//         LambdaScopeTest st = new LambdaScopeTest();
//         LambdaScopeTest.FirstLevel fl = st.new FirstLevel();
//         fl.methodInFirstLevel(23);
//     }
// }
// // This example generates the following output:
// // x = 23
// // y = 23
// // this.x = 1
// // LambdaScopeTest.this.x = 0


// // If you substitute the parameter x in place of y in the declaration of the lambda expression myConsumer, then the compiler generates an error:

// Consumer<Integer> myConsumer = (x) -> {
//     // ...
// }
// // The compiler generates the error "variable x is already defined in method methodInFirstLevel(int)" because the lambda expression does not introduce a new level of scoping. Consequently, you can directly access fields, methods, and local variables of the enclosing scope. For example, the lambda expression directly accesses the parameter x of the method methodInFirstLevel. To access variables in the enclosing class, use the keyword this. In this example, this.x refers to the member variable FirstLevel.x.

// // However, like local and anonymous classes, a lambda expression can only access local variables and parameters of the enclosing block that are final or effectively final. For example, suppose that you add the following assignment statement immediately after the methodInFirstLevel definition statement:

// void methodInFirstLevel(int x) {
//     x = 99;
//     // ...
// }
// // Because of this assignment statement, the variable FirstLevel.x is not effectively final anymore. As a result, the Java compiler generates an error message similar to "local variables referenced from a lambda expression must be final or effectively final" where the lambda expression myConsumer tries to access the FirstLevel.x variable:

// System.out.println("x = " + x);
