package JAVA.javaCrush;

public interface Vehicle {

    // fianl means it is a constant and it cannot be changed
    final int gears = 5;

    void changeGear(int a);
    void speedUp(int a);
    void slowDown(int a);

    default void out() {
        System.out.println("Default method");
    }

    static int math(int b) {
        return b + 9;
    }
}
