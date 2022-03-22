package list;

import javax.naming.spi.DirStateFactory.Result;

// Maintains a list of elements ordered with move-to-front heuristic. ∗/
public class FavoritesListMTF<E> extends FavoritesList<E> {
    // Moves accessed item at Position p to the front of the list. ∗/
    protected void moveUp(Position<Item<E>> p){
        if(p!=list.first()) list.addFirst(list.remove(p));
    }

    // /∗∗ Returns an iterable collection of the k most frequently accessed elements. ∗/
    public Iterable<E> getFavorites(int k) throws IllegalArgumentException{
        if(k<0 || k>size()) throw new IllegalArgumentException("invalid k");
        PositionalList<Item<E>> temp = new LinkedPositionalList<>();
        for(Item<E> item : list){
            temp.addLast(item);
        }

        PositionalList<Item<E>> res = new LinkedPositionalList<>();
        for(int j=0; j<k; j++){
            Position<Item<E>> high = temp.first();
            Position<Item<E>> walk = temp.after(high);
            while(walk!=null){
                if(count(walk) > count(high)) high=walk;
                walk = temp.after(walk);
            }
            res.addLast(value(high));
            temp.remove(high);
        }
    }
}
