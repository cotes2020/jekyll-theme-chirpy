package tree;
import java.util.List;

import javax.swing.text.Position;

import list.*;
import queue.*;

// /** An abstract base class providing some functionality of the Tree interface. */
public abstract class AbstractTree<E> implements Tree<E> {
    public boolean isInternal(Position<E> p) {return numChildren(p)>0;}
    public boolean isExternal(Position<E> p) {return numChildren(p)<0;}
    public boolean isRoot(Position<E> p) {return p==root();}
    public boolean isEmpty(){return size()==0;}

    // * Returns the number of levels separating Position p from the root. */
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

    public Iterator<E> iterator() { return new ElementIterator(); }

    public Iterable<Position<E>> positions() { return inorder(); }



    // ** Returns an iterable collection of positions of the tree, reported in preorder. */
    public Iterable<Position<E>> preorder() {
        List<Position<E>> snapshot = new ArrayList<>();
        if (!isEmpty()) preorderSubtree(root(), snapshot); // fill the snapshot recursively
        return snapshot;
    }

    // ** Adds positions of the subtree rooted at Position p to the given snapshot. */
    private void preorderSubtree(Position<E> p, List<Position<E>> snapshot) {
        snapshot.add(p); // for preorder, we add position p before exploring subtrees
        for (Position<E> c : children(p)) preorderSubtree(c, snapshot);
    }


    // ** Returns an iterable collection of positions of the tree, reported in postorder. */
    public Iterable<Position<E>> postorder() {
        List<Position<E>> snapshot = new ArrayList<>();
        if (!isEmpty()) postorderSubtree(root(), snapshot); // fill the snapshot recursively
        return snapshot;
    }

    // ** Adds positions of the subtree rooted at Position p to the given snapshot. */
    private void postorderSubtree(Position<E> p, List<Position<E>> snapshot) {
        for (Position<E> c : children(p)) postorderSubtree(c, snapshot);
        snapshot.add(p); // for preorder, we add position p before exploring subtrees
    }



    // /** Returns an iterable collection of positions of the tree in breadth-first order. */
    public Iterable<Position<E>> breadthfirst() {
        List<Position<E>> snapshot = new ArrayList<>();
        if (!isEmpty()) {
            Queue<Position<E>> fringe = new LinkedQueue<>();
            fringe.enqueue(root());
            while(!fringe.isEmpty()){
                Position<E> p = fringe.dequeue();
                snapshot.add(p);
                for(Position<E> c: children(p)){
                    fringe.enqueue(c);
                }
            }
        }
        return snapshot;
    }



    // ** Returns an iterable collection of positions of the tree, reported in inorder. */
    public Iterable<Position<E>> inorder() {
        List<Position<E>> snapshot = new ArrayList<>();
        if (!isEmpty()) inorderSubtree(root(), snapshot); // fill the snapshot recursively
        return snapshot;
    }

    // ** Adds positions of the subtree rooted at Position p to the given snapshot. */
    private void inorderSubtree(Position<E> p, List<Position<E>> snapshot) {
        if (left(p) != null) inorderSubtree(left(p), snapshot);
        snapshot.add(p);
        if (right(p) != null) inorderSubtree(right(p), snapshot);
    }




    public static <E> void printPreorderIndent(Tree<E> T, Position<E> p, int d) {
        System.out.println(spaces(2*d) + p.getElement()); // indent based on d
        for (Position<E> c : T.children(p)) printPreorderIndent(T, c, d+1); // child depth is d+1
    }


    // /∗∗ Prints parenthesized representation of subtree of T rooted at p. ∗/
    public static <E> void parenthesize(Tree<E> T, Position<E> p) {
        System.out.print(p.getElement());
        if (T.isInternal(p)) {
            boolean firstTime = true;
            for(Position<E> c : T.children(p)){
                System.out.print( (firstTime ? " (" : ", ") ); // determine proper punctuation
                firstTime = false;
                parenthesize(T, c);
            }
            System.out.print(")");
        }
    }


    public static <E> int layout(BinaryTree<E> T, Position<E> p, int d, int x) {
        if (T.left(p) != null) x = layout(T, T.left(p), d+1, x);
        p.getElement().setX(x++);
        p.getElement().setY(d);
        if (T.left(p) != null) x = layout(T, T.left(p), d+1, x);
        return x;
    }

}
