package pq;




// This provides a nested PQEntry class that composes a key and a value into a single object, and support for managing a comparator.
public abstract class AbstractPriorityQueue<K,V> implements PriorityQueue<K,V> {

    //---------------- nested PQEntry class ----------------
    protected static class PQEntry<K,V> implements Entry<K,V> {
        private K k;
        private V v;
        public PQEntry(K key, V value){
            k=key;
            v=value;
        }
        // methods of the Entry interface
        public K getKey(){return k;}
        public V getV(){return v;}
        // utilities not exposed as part of the Entry interface
        protected void setKey(K key){k=key;}
        protected void setValue(V value){v=value;}

    }

    // instance variable for an AbstractPriorityQueue
    private Comparator<K> comp;
    protected AbstractPriorityQueue(Comparator<K> c){comp = c;}
    protected AbstractPriorityQueue(){ this(new DefaultComparator<K>()); }

    // /∗∗ Method for comparing two entries according to key ∗/
    protected int compare(Entry<K,V> a, Entry<K,V> b) { return comp.compare(a.getKey(), b.getKey());}

    // /∗∗ Determines whether a key is valid. ∗/
    protected boolean checkKey(K key) throws IllegalArgumentException {
        try {
            return comp.compare(key, key)==0;
        } catch (ClassCastException e) {
            throw new IllegalArgumentException("Incompatible key.");
        }
    }

    // /∗∗ Tests whether the priority queue is empty. ∗/
    public boolean isEmpty(){return size()==0;}
}
