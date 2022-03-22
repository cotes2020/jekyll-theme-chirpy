package tree; 
import list.*; 

// /∗∗ An abstract base class providing some functionality of the Tree interface. ∗/
public abstract class AbstractTree<E> implements Tree<E> {
    public boolean isInternal(Position<E> p) {return numChildren(p)>0;}
    public boolean isExternal(Position<E> p) {return numChildren(p)<0;}
    public boolean isRoot(Position<E> p) {return p==root();}
    public boolean isEmpty(){return size()==0;}

    // ∗ Returns the number of levels separating Position p from the root. ∗/
    public int depth(Position<E> p) {
        if(isRoot(p)) return 0;
        return depth(parent(p)) + 1;
    }

    private int heightBad(){
        int h=0;
        for(Position<E> p: positions()) h=Math.max(h, depth(p));
        return h;
    }

    private int height(Position<E> p){
        int h=0;
        for(Position<E> c: children(p)) h=Math.max(h, 1 + height(c));
        return h;
    }
}