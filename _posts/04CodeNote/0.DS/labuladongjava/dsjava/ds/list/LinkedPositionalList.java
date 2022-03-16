package list;

// public class LinkedPositionalList<E> implements PositionalList<E>  {
//     private static class Node<E> implements Position<E>{ }
//     private Node<E> header;
//     private Node<E> trailer;
//     private int size = 0;
//     public LinkedPositionalList(){}
//     private Node<E> validate(Position<E> p) throws IllegalArgumentException {}
// }
 
public class LinkedPositionalList<E> implements PositionalList<E>  {
    
    private static class Node<E> implements Position<E>{
        private E element;
        private Node<E> prev;
        private Node<E> next;
        public Node(E e, Node<E> p, Node<E> n){
            element = e;
            prev = p;
            next = n;
        }
        public E getElement() throws IllegalStateException {
            if(next==null) throw new IllegalStateException("Position no longer valid");
            return element;
        }
        public Node<E> getPrev() {
            return prev;
        }
        public Node<E> getNext() {
            return next;
        }

        public void setElement(E e) {
            element = e;
        }
        public void setPrev(Node<E> p) {
            prev = p;
        }
        public void setNext(Node<E> n) {
            next = n;
        }
    } 
    // ----------- end of nested Node class ----------

    private Node<E> header;
    private Node<E> trailer;
    private int size = 0;

    /** Constructs a new empty list. */
    public LinkedPositionalList(){
        header = new Node<>(null,null,null);
        trailer = new Node<>(null,header,null);   
        header.setNext(trailer);
    }

    // private utilities
    // ** Validates the position and returns it as a node. */
    private Node<E> validate(Position<E> p) throws IllegalArgumentException {
        if (!(p instanceof Node)) throw new IllegalArgumentException("Invalid p"); 
        Node<E> node = (Node<E>) p; // safe cast 
        // convention for defunct node
        if (node.getNext() == null) throw new IllegalArgumentException("p is no longer in the list"); 
        return node;
    }
    // ** Returns the given node as a Position (or null, if it is a sentinel). */
    private Position<E> position(Node<E> node){
        if(node == header || node == trailer) return null;
        return node;
    }

    // public accessor methods
    public int size() {return size;}
    public boolean isEmpty() {return size==0;}
    public Position<E> first() {return header.getNext();}
    public Position<E> last() {return trailer.getPrev();}
    public Position<E> before(Position<E> p) throws IllegalStateException {
        Node<E> node = validate(p);
        return node.getPrev();
    }
    public Position<E> after(Position<E> p) throws IllegalStateException {
        Node<E> node = validate(p);
        return node.getNext();
    }

        
    // private utilities
    public Position<E> addBetween(E e, Node<E> prev, Node<E> succ) throws IllegalStateException {
        Node<E> node = new Node<>(e, prev, succ);
        prev.setNext(node);
        succ.setPrev(node);
        size++;
        return node;
    }
    public Position<E> addLast(E e) {
        return addBetween(e, trailer.getPrev(), trailer);
    }
    public Position<E> addFirst(E e) {
        return addBetween(e, header, header.getNext());
    }
    public Position<E> addBefore(Node<E> p , E e) throws IllegalStateException {
        Node<E> node = validate(p);
        return addBetween(e, node.getPrev(), node);
    }
    public Position<E> addAfter(Node<E> p , E e) throws IllegalStateException {
        Node<E> node = validate(p);
        return addBetween(e, node, node.getNext());
    }
    public E set(Node<E> p , E e) throws IllegalStateException {
        Node<E> node = validate(p);
        E ans = node.getElement();
        node.setElement(e);
        return ans;
    }
    public E remove(Position<E> p) throws IllegalStateException {
        Node<E> node = validate(p);
        Node<E> prev = node.getPrev();
        Node<E> next = node.getNext();
        E ans = node.getElement();
        prev.setNext(next);
        next.setPrev(prev);
        node.setElement(null);
        node.setNext(null);
        node.setPrev(null);
        size--;
        return ans;
    }
}
