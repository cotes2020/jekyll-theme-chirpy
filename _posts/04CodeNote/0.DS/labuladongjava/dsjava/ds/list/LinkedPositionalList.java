package list;

import javax.swing.text.Position;

public class LinkedPositionalList<E> implements PositionList<E>  {
    
    private static class Node<E> implements Position<E>{
        private E element;
        private Node<E> prev;
        private Node<E> next;
        public Node(E e, Node<E> p, Node<E> n){
            element = e;
            prev = p;
            next = n;
        }
        public E getElememnt() throws IllegalStateException{
            if(next==null) throw new IllegalStateException("Position no longer valid");
            return element;
        }
        public Node<E> getPrev(){
            return prev;
        }
        public Node<E> getNext(){
            return next;
        }

        public E setElememnt(E e) {
            element = e;
        }
        public E setPrev(Node<E> p) {
            prev = p;
        }
        public E setNext(Node<E> n) {
            next = n;
        }
    } 

    private Node<E> header;
    private Node<E> trailer;
    private int size = 0;

    // /∗∗ Constructs a new empty list. ∗/
    public LinkedPositionalList(){
        header = new Node<>(null,null,null);
        trailer = new Node<>(null,header,null);   
        header.setNext(trailer);
    }

    // private utilities
    // /∗∗ Validates the position and returns it as a node. ∗/
    private Node<E> validate(Position<E> p) throws IllegalArgumentException {
        if (!(p instanceof Node)) throw new IllegalArgumentException("Invalid p"); 
        Node<E> node = (Node<E>) p; // safe cast 
        // convention for defunct node
        if (node.getNext() == null) throw new IllegalArgumentException("p is no longer in the list"); 
        return node;
    }

    // /∗∗ Returns the given node as a Position (or null, if it is a sentinel). ∗/
    private Position<E> position(Node<E> node){
        if(node == header || node == trailer) return null;
        return node;
    }
}
