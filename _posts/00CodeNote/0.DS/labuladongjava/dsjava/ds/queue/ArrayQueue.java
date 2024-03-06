package queue;

public class ArrayQueue<E> implements Queue<E> {

    private E[] data;
    private int f=0;
    private int sz=0;

    // constructors
    public ArrayQueue(){ this(CAPACITY); }
    public ArrayQueue(int capacity){
        data = new Object[capacity];
    }

    public int size() {return sz;}
    public boolean isEmpty() {return sz==0;}

    public void enqueue(E e) throws IllegalStateException {
        if(sz==data.length) throw new IllegalStateException("Queue is full");
        int avail = (f+sz)%data.length;
        data[avail] = e;
        sz++;
    }

    public E first() {
        return isEmpty()? null: data[f];
    }

    // /∗∗ Removes and returns the first element of the queue (null if empty). ∗/
    public E dequeue() {
        if(isEmpty()) return null;
        E ans = data[f];
        data[f] = null;
        f = (f+1)% data.length;
        sz--;
        return ans;
    }
}
