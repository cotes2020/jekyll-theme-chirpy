package labuladongjava.PriorityQueue; 
import java.util.Comparator;

public class SortedPriorityQueue<K,V> extends AbstractPriorityQueue<K,V> {
    // /∗∗ primary collection of priority queue entries ∗/
    private PositionalList<Entry<K,V>> list = new LinkedPositionalList<>();

    // /∗∗ Creates an empty priority queue based on the natural ordering of its keys. ∗/
    public SortedPriorityQueue(){
        super();
    }
    // /∗∗ Creates an empty priority queue using the given comparator to order keys. ∗/
    public SortedPriorityQueue(Comparator<K> comp){
        super(comp);
    }

    // /∗∗ Inserts a key-value pair and returns the entry created. ∗/
    public Entry<K,V> insert(K key, V value) throws IllegalArgumentException {
        checkKey(key);
        Entry<K,V> newset = new PQEntry<>(key, value);
        Position<Entry<K,V>> walk = list.last();

        // walk backward, looking for smaller key
        // whill newset small
        while(walk!=null && compare(newset, walk.getElement())<0){
            walk = list.before(walk);
        }
        if(walk == null) list.addFirst(newset);
        else list.addAfter(walk, newset);
        return newset;
    }
    
    public Entry<K, V> min() {
        if(list.isEmpty()) return null;
        return list.first().getElement();
    }

    public Entry<K, V> removeMin() {
        if(list.isEmpty()) return null;
        return list.remove(list.first());
    }


    public int size() {
        return list.size();
    }
    
}
