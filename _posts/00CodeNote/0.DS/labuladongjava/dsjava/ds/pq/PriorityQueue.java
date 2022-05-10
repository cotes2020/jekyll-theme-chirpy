package pq;

// ∗∗ Interface for the priority queue ADT. ∗/
public interface PriorityQueue<K, V> {
    int size();
    boolean isEmpty();
    Entry<K,V> insert(K key, V value) throws IllegalArgumentException;
    Entry<K,V> mim();
    Entry<K,V> removeMim();
}
