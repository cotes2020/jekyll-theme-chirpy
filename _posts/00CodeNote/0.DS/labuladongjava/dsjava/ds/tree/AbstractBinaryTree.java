package tree;
import list.*;
import array.*;

public class AbstractBinaryTree<E> extends AbstractTree<E> implements BinaryTree<E> {

    public Position<E> sibling(Position<E> p){
        Position<E> parent = parent(p);
        if(parent == null) return null;
        if(p==left(parent)) return right(p);
        else return left(p);
    }

    public int numChildren(Position<E> p){
        int count=0;
        if(left(p)!=null) count++;
        if(right(p)!=null) count++;
        return count;
    }

    public Iterable<Position<E>> children(Position<E> p){
        Iterable<Position<E>> snapshot = new ArrayList<>(2);
        if(left(p)!=null) snapshot.add(left(p));
        if(right(p)!=null) snapshot.add(right(p));
        return snapshot;
    }

}
