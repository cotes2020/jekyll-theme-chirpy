package array;

public class Node<E> {

    private E element;
    private Node<E> next;

    public Node(E e, Node<E> n){
        element = e;
        next = n;
    }

    public E getElement() {return element;}
    public Node<E> getNext() {return next;}
    public void setNext(Node<E> m) {next = m;}

}
