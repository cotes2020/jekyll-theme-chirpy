package JAVA.javaCrush;

public class OuterClass {


    // private class InnerClass {
    // public class InnerClass {

    //     public void display() {
    //         System.out.println("This is an inner class");
    //     }

    // }


    public void inner() {


        class InnerClass {

            public void display() {
                System.out.println("This is an inner class");

            }
        }

        InnerClass in = new InnerClass();
        in.display();
    }

}
