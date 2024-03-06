import queue.*;

public class Josephus {

    // /∗∗ Computes the winner of the Josephus problem using a circular queue. ∗/
    public static <E> E Josephus (CircularQueue<E> queue, int k) {
        if(queue.isEmpty()) return null;
        while(queue.size()>2){
            for(int i=0; i<k; i++) queue.rotate();
            E e = queue.dequeue();
            System.out.println("" + e + " is out");
            k--;
        }
        return queue.dequeue();
    }

    public static <E> CircularQueue<E> buildQueue(E[] a) {
        CircularQueue<E> queue = new LinkedCircularQueue<>();
        for(E data: a) queue.enqueue(data);
        return queue;
    }

    public static void main(String[ ] args) {
        String[] a1 = {"Alice", "Bob", "Cindy", "Doug", "Ed", "Fred"};
        // String[] a1 = {"Alice", "Bob", "Cindy", "Ed", "Fred"};
        // String[] a1 = {"Bob", "Cindy", "Ed", "Fred"};
        // String[] a1 = {"Bob", "Ed", "Fred"};
        // String[] a1 = {"Fred"};
        String[] a2 = {"Gene", "Hope", "Irene", "Jack", "Kim", "Lance"};
        String[] a3 = {"Mike", "Roberto"};
        System.out.println("First winner is " + Josephus(buildQueue(a1), 3));
        System.out.println("Second winner is " + Josephus(buildQueue(a2), 10));
        System.out.println("Third winner is " + Josephus(buildQueue(a3), 7));
    }
}
