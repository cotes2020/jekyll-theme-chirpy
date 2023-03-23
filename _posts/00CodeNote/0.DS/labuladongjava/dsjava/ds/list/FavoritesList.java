package list;

/** Maintains a list of elements ordered according to access frequency. */

public class FavoritesList<E> {
    // ---------------- nested Item class ----------------
    protected static class Item<E> {
        private E value;
        private int count = 0;
        // /∗∗ Constructs new item with initial count of zero. ∗/
        public Item(E val) {value = val;}
        public int getCount() {return count;}
        public E getValue() {return value;}
        public void increment() {count++;}
    }

    PositionalList<Item<E>> list = new LinkedPositionalList<>(); // list of Items

    public FavoritesList() {} // constructs initially empty favorites list

    // nonpublic utilities
    /** Provides shorthand notation to retrieve user's element stored at Position p. */
    protected E value(Position<Item<E>> p) {return p.getElement().getValue();}
    /** Provides shorthand notation to retrieve count of item stored at Position p. */
    protected int count(Position<Item<E>> p) {return p.getElement().getCount();}

    /** Returns Position having element equal to e (or null if not found). */
    protected Position<Item<E>> findPosition(E e) {
        Position<Item<E>> walk  = list.first();
        while(walk!=null && e.equals((walk))) walk = list.after(walk);
        return walk;
    }

    // /∗∗ Moves item at Position p earlier in the list based on access count. ∗/
    protected void moveUp(Position<Item<E>> p) {
        int cnt = count(p);
        Position<Item<E>> walk = p;
        while(walk!=null && count(list.before(walk))<cnt) walk = list.before(walk);
        if(walk != p) list.addBefore(walk, list.remove(p));
    }

    // public methods
    public int size(){return list.size();}
    public Boolean isEmpty(){return list.isEmpty();}

    // /∗∗ Accesses element e (possibly new), increasing its access count. ∗/
    public void access(E e){
        Position<Item<E>> p = findPosition(e);
        if (p == null) list.addLast(new Item<E>(e));
        p.getElement().increment();
        moveUp(p);
    }

    // /∗∗ Removes element equal to e from the list of favorites (if found). ∗/
    public void remove(E e){
        Position<Item<E>> p = findPosition(e);
        if (p != null) {
            list.remove(p);
        }
    }

    // /∗∗ Returns an iterable collection of the k most frequently accessed elements. ∗/
    public Iterable<E> getFavorites(int k) throws IllegalArgumentException {
        if (k < 0 || k > size()) throw new IllegalArgumentException("Invalid k");

        PositionalList<E> result = new LinkedPositionalList<>();
        Iterator<Item<E>> iter = list.iterator();
        for (int j=0; j < k; j++) result.addLast(iter.next().getValue());
        return result;
    }
}
