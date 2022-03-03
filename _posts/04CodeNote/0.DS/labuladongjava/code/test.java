import java.util.Scanner;

public class test {
    public static void main(String[] args) {
        Scanner inputint = new Scanner(System.in); 
        System.out.print("Please enter an integer: ");
        while (!inputint.hasNextInt()) {
                inputint.nextLine( );
                System.out.print("Invalid integer; please enter an integer: ");
        } 
        // System.out.println(inputint);
        int i = inputint.nextInt();
        System.out.println("i: "+i);
    }
    
}
