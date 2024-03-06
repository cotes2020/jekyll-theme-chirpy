package queue;

import array.*;

public class LinkedCircularQueue<E> implements CircularQueue<E> {
    private CircularlyLinkedList<E> list = new CircularlyLinkedList<>();

    public LinkedCircularQueue(){}

    public int size(){return list.size();}
    public boolean isEmpty(){return list.isEmpty();}
    public void enqueue(E e){list.addLast(e);}
    public E first(){return list.first();}
    public E dequeue(){return list.removeFirst();}
    public void rotate(){list.rotated();}
}
