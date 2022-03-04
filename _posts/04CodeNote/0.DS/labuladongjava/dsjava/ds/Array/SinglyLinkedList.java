public class SinglyLinkedList<E> {
    
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

    public class SinglyLinkedList<E> {
        private Node<E> head = null;
        private Node<E> tail = null;
        private int size = 0;
        public SinglyLinkedList(){}

        // access methods
        public int size() {return size;}
        public boolean isEmpty() {return size == 0;}

        public E first() {
            return isEmpty()? null:head.getElement(); 
        }
        public E last() {
            return isEmpty()? null:tail.getElement();
        }

        // update methods
        public void addFrist(E e) {
            head = new Node<>(e, head);
            if(size==0) tail = head;
            size++;
        }
        public void addLast(E e) {
            Node newest = new Node<>(e, null);
            if(size==0) tail = head;
            tail.setNext(newest);
            tail = newest;
            size++;
        }
        public E removeFirst() {
            if(size<=1) return null;
            Node ans = head.getElement();
            head = head.next;
            size--;
            return ans;
        } 

        public boolean equals(Object o) {
            if(o==null) return false;
            if(getClass() != o.getClass()) return false;

            SinglyLinkedList other = (SinglyLinkedList) o;
            if(size != other.size) return false;
            while(head != null) {
                if( !head.getElement().equals(other.head.getElement()) ) return false;
                head = head.getNext();
                other.head = other.head.getNext();
            }
            return true;
        }
    }
}
