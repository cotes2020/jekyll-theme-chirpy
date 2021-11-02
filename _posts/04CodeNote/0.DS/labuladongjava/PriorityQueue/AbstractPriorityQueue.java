package labuladongjava;

import java.util.Comparator;

import labuladongjava.PriorityQueue.Entry;
import labuladongjava.PriorityQueue.PriorityQueue;

// This provides a nested PQEntry class that composes a key and a value into a single object, and support for managing a comparator.
public abstract class AbstractPriorityQueue<K,V> implements PriorityQueue<K,V> {

    protected static class PQEntry<K,V> implements Entry<K,V>{
        private K k;
        private V v;
        public PQEntry(K key, V value){
            k = key;
            v = value;
        }
        public K getKey(){return k;}
        public V getValue(){return v;}
        protected void setKey(K key){k=key;} 
        protected void setValue(V value){V=value;}
    }
    
    // instance variable for an AbstractPriorityQueue
    // /∗∗ The comparator defining the ordering of keys in the priority queue. ∗/ 
    private Comparator<K> comp;

    // /∗∗ Creates an empty priority queue using the given comparator to order keys. ∗/ 
    protected AbstractPriorityQueue(Comparator<K> c) { 
        comp = c; 
    }

    // /∗∗ Creates an empty priority queue based on the natural ordering of its keys. ∗/ 
    protected AbstractPriorityQueue() { 
        this(new DefaultComparator<K>()); 
    } 

    // /∗∗ Method for comparing two entries according to key ∗/
    protected int compare(Entry<K,V> a, Entry<K,V> b) {
        return comp.compare(a.getKey(), b.getKey()); 
    }

    // /∗∗ Determines whether a key is valid. ∗/
    protected boolean checkKey(K key) throws IllegalArgumentException {
    try {
        return (comp.compare(key,key) == 0); // see if key can be compared to itself
        } catch (ClassCastException e) {
            throw new IllegalArgumentException("Incompatible key");
        } 
    }

    // /∗∗ Tests whether the priority queue is empty. ∗/ 
    public boolean isEmpty() { 
        return size() == 0; 
    }
        
}
