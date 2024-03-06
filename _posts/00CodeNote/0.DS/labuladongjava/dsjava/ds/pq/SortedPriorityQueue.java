package pq;
import list.*;

// ** An implementation of a priority queue with an unsorted list. */
public class SortedPriorityQueue<K,V> extends AbstractPriorityQueue<K,V> {

    // ∗∗ primary collection of priority queue entries ∗/
    private PositionalList<Entry<K,V>> list = new LinkedPositionalList<>();

    /** Creates an empty priority queue based on the natural ordering of its keys. */
    public SortedPriorityQueue() { super(); }
    /** Creates an empty priority queue using the given comparator to order keys. */
    public SortedPriorityQueue(Comparator<K> comp) { super(comp); }

    /** Inserts a key-value pair and returns the entry created. */
    public Entry<K,V> insert(K key, V value) throws IllegalArgumentException {
        checkKey(key); // auxiliary key-checking method (could throw exception)
        Entry<K,V> newest = new PQEntry<>(key, value);
        Position<Entry<K,V>> walk = list.last();
        while(walk != null && compare(newest, walk.getElement())<0 ) walk = list.before(walk);
        if(walk==null) list.addFirst(newest);
        else list.addAfter(walk, newest);
        return newest;
    }

    /** Returns the Position of an entry having minimal key. */
    // private Position<Entry<K,V>> findMin() { // only called when nonempty
    //     return list.first();
    // }

    /** Returns (but does not remove) an entry with minimal key. */
    public Entry<K,V> min() {
        if (list.isEmpty()) return null;
        return list.first().getElement();
    }

    /** Removes and returns an entry with minimal key. */
    public Entry<K,V> removeMin() {
        if (list.isEmpty()) return null;
        return list.remove(list.first()); }

    /** Returns the number of items in the priority queue. */
    public int size() { return list.size(); }
}
