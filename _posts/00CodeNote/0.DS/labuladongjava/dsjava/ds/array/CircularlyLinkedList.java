package array;

public class CircularlyLinkedList<E> {

    private Node<E> tail = null;
    private int size = 0;

    public CircularlyLinkedList(){}

    // access methods
    public int size() {return size;}
    public boolean isEmpty() {return size == 0;}

    public E first() {
        return isEmpty()? null:tail.getNext().getElement();
    }
    public E last() {
        return isEmpty()? null:tail.getElement();
    }

    // update methods
    public void addfirst(E e) {
        if(size==0) {
            tail = new Node<>(e, null);
            tail.setNext(tail);
        }
        else {
            Node<E> newest = new Node<>(e, tail.getNext());
            tail.setNext(newest);
        }
        size++;
    }
    public void addLast(E e) {
        addfirst(e);
        tail = tail.getNext();
    }

    public E removeFirst() {
        if(size<=1) return null;
        Node<E> ans = tail.getNext();
        tail.setNext(ans.getNext());
        size--;
        return ans.getElement();
    }

    public void rotated() {
        Node<E> temp = tail.getNext().getNext();
        tail = tail.getNext();
        tail.setNext(temp);
    }
}
