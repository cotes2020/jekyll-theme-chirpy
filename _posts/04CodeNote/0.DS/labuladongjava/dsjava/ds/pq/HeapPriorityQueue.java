package pq;


/** An implementation of a priority queue using an array-based heap. */
public class HeapPriorityQueue<K,V> extends AbstractPriorityQueue<K,V> {
    
    /** primary collection of priority queue entries */
    protected ArrayList<Entry<K,V>> heap = new ArrayList<>();
    
    /** Creates an empty priority queue based on the natural ordering of its keys. */ 
    public HeapPriorityQueue() { super(); }
    /** Creates an empty priority queue using the given comparator to order keys. */ 
    public HeapPriorityQueue(Comparator<K> comp) { super(comp); }
    
    // protected utilities
    protected int parent(int j) { return (j-1) / 2; } // truncating division protected int left(int j) { return 2*j + 1; }
    protected int right(int j) { return 2*j + 2; }
    protected boolean hasLeft(int j) { return left(j) < heap.size(); }
    protected boolean hasRight(int j) { return right(j) < heap.size(); }
    /** Exchanges the entries at indices i and j of the array list. */
    protected void swap(int i, int j) {
    Entry<K,V> temp = heap.get(i); heap.set(i, heap.get(j)); heap.set(j, temp);
    }
    /** Moves the entry at index j higher, if necessary, to restore the heap property. */ protected void upheap(int j) {
    while (j > 0) { // continue until reaching root (or break statement) int p = parent(j);
    if (compare(heap.get(j), heap.get(p)) >= 0) break; // heap property verified swap(j, p);
    j = p; // continue from the parent's location 
}
}