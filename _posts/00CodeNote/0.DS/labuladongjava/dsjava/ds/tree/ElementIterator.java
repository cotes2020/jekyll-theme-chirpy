package tree;
import java.util.Iterator;
import list.*;
import array.*;

private class ElementIterator<E> implements Iterator<E> {

    Iterator<Position<E>> posIterator = position().iterator();
    public boolean hasNext() {return posIterator.hasNext();}
    public E next() {return posIterator.next().getElement();}
    public void remove() {return posIterator.remove();}

}
