
package labuladongjava.PriorityQueue;

import labuladongjava.PriorityQueue.*;
import java.util.Comparator;

public class UnsortedPriorityQueue<K,V> extends AbstractPriorityQueue<K,V> {
    // /∗∗ primary collection of priority queue entries ∗/
    private PositionalList<Entry<K,V>> list = new LinkedPositionalList<>();

    // /∗∗ Creates an empty priority queue based on the natural ordering of its keys. ∗/
    public UnsortedPriorityQueue(){
        super();
    }
    // /∗∗ Creates an empty priority queue using the given comparator to order keys. ∗/
    public UnsortedPriorityQueue(Comparator<K> comp){
        super(comp);
    }

    // /∗∗ Returns the Position of an entry having minimal key. ∗/
    private Position<Entry<K,V>> findMin(){
        Position<Entry<K,V>> small = list.first();
        for(Position<Entry<K,V>> walk:list){
          if(compare(walk.getElement(), small.getElement()) < 0 ){
              small = walk;
          }
        }
    }
    @Override
    public int size() {
        // TODO Auto-generated method stub
        return 0;
    }
    @Override
    public Entry<K, V> insert(K key, V value) throws IllegalArgumentException {
        // TODO Auto-generated method stub
        return null;
    }
    @Override
    public Entry<K, V> min() {
        // TODO Auto-generated method stub
        return null;
    }
    @Override
    public Entry<K, V> removeMin() {
        // TODO Auto-generated method stub
        return null;
    }
}
