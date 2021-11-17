package labuladongjava;

import java.util.LinkedList;

public class run {
     
    // public static void main(String[ ] args) { 
    //     CreditCard[] wallet = new CreditCard[3]; 
    //     wallet[0] = new CreditCard("John Bowman", "California Savings", "5391 0375 9387 5309", 5000); 
    //     wallet[1] = new CreditCard("John Bowman", "California Federal", "3485 0399 3395 1954", 3500); 
    //     wallet[2] = new CreditCard("John Bowman", "California Finance", "5391 0375 9387 5309", 2500, 300); 
    //     for (int val = 1; val <= 16; val++) { 
    //         wallet[0].charge( 3*val ); 
    //         wallet[1].charge( 2*val); 
    //         wallet[2].charge( val );
    //     }
    //     for (CreditCard card : wallet) { 
    //         CreditCard.printSummary(card); 
    //         while (card.getBalance() > 200.0) {
    //             card.makePayment(200);
    //             // calling static method
    //             System.out.println("New balance = " + card.getBalance( )); 
    //         }
    //     }
    // }


    public static void main(String[] args) {
        Labu test = new Labu();

        // initialize the first elements of the array
        // ListNode anw = Labu.mergeTwoLists(l1, l2);

        // // accessing the elements of the specified array
        // for (int i = 0; i < arr.length; i++)
        //     System.out.println("Element at " + i + " : " +
        //                 arr[i].roll_no +" "+ arr[i].name);

        // String[] equations = {"c==c"};
        // String[] equations = {"b==d"};
        // // String[] equations = {"c==c","b==d","x!=z"};
        // System.out.println(test.equationsPossible(equations));


        // int[] arr = {1,2,3,4,5};
        // int n = arr.length, index = 0;
        // for(int i=0; i<7; i++) {
        //     System.out.println(arr[index % n]);
        //     index++;
        // }

        LinkedList<Integer> q = new LinkedList<>();

        System.out.println(q.isEmpty());
        q.addLast(1);
        q.addLast(2);
        q.addLast(3);
        q.addLast(4);
        q.addLast(5);
        System.out.println(q);
        System.out.println(q.getLast());
        System.out.println(q.pollLast()); 
        System.out.println(q);
    }
}
