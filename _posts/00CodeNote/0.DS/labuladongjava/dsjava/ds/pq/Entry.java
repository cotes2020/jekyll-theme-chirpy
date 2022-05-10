package pq;

// ∗ Interface for a key-value pair. ∗/
public interface Entry<K, V> {
    K getKey();
    V getV();
}
