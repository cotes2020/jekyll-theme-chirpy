package array;


public class test<E> {

    private static class Node<E> {
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

    public class test<E> {
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
                Node newest = new Node<>(e, tail.getNext());
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
            Node ans = tail.getNext();
            tail.setNext(ans.getNext());
            size--;
            return ans.getElement();
        }
    }

    public int size() {
        return 0;
    }

    public boolean isEmpty() {
        return false;
    }

    public void addLast(E e) {
    }
}
