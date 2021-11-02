
package labuladongjava.PriorityQueue;

import java.util.Comparator;

import javax.swing.plaf.nimbus.AbstractRegionPainter;
import javax.swing.text.Position;

import labuladongjava.AbstractPriorityQueue;
import labuladongjava.LinkedPositionalList;
import labuladongjava.PositionalList;

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
        for( PositionalList<Entry<K,V>> walk:list.positions() ){
          if(compare(walk.getElement(), small.getElement()) < 0 ){
              small = walk;
          }
        }
    }
}
