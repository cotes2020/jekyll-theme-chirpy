package array;

public class DoublyLinkedList<E> {

    private static class Node<E> {
        private E element;
        private Node<E> prev;
        private Node<E> next;
        public Node(E e, Node<E> p, Node<E> n){
            element = e;
            prev = p;
            next = n;
        }
        public E getElement() {return element;}
        public Node<E> getPrev() {return prev;}
        public Node<E> getNext() {return next;}
        public void setNext(Node<E> m) {next = m;}
        public void setPrev(Node<E> m) {prev = m;}
    }

    private Node<E> head;
    private Node<E> tail;
    private int size = 0;

    public class DoublyLinkedList<E> {

        public DoublyLinkedList(){
            head = new Node<>(null, null, null);
            tail = new Node<>(null, head, null);
            head.setNext(tail);
        }

        // access methods
        public int size() {return size;}
        public boolean isEmpty() {return size == 0;}

        public E first() {
            return isEmpty()? null:head.getNext().getElement();
        }
        public E last() {
            return isEmpty()? null:tail.getPrev().getElement();
        }

        // update methods
        private void addBetween(E e, Node<E> pre, Node<E> suc){
            Node newest = new Node<>(e, pre, suc);
            pre.setNext(newest);
            suc.setPrev(newest);
            size++;
        }
        private E remove(Node<E> e){
            Node<E> pre = e.getPrev();
            Node<E> suc = e.getNext();
            pre.setNext(suc);
            suc.setPrev(pre);
            size--;
            return e.getElement();
        }


        public void addfirst(E e) {
            addBetween(e, head, head.getNext());
        }
        public void addLast(E e) {
            addBetween(e, tail.getPrev(), tail);
        }
        public E removeFirst() {
            return isEmpty?null: remove(head.getNext());
        }
        public E removeLast() {
            return isEmpty?null: remove(tail.getPrev());
        }
    }
}
