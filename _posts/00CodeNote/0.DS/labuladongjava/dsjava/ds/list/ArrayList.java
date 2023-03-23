package list;

import java.util.Iterator;
import java.util.NoSuchElementException;

public class ArrayList<E> implements List<E> { // instance variables {

    // default array capacity
    public static final int CAPACITY=16;
    // generic array used for storage
    private E[ ] data;
    private int size = 0;

    // constructors
    public ArrayList() { this(CAPACITY); }

    // constructs list with given capacity
    public ArrayList(int capacity) {
        data = (E[]) new Object[capacity];
    }

    // public methods
    public int size() { return size; }
    public boolean isEmpty() { return size == 0; }

    // utility method
    // /** Checks whether the given index is in the range [0, n-1]. */
    protected void checkIndex(int i, int n) throws IndexOutOfBoundsException {
        if (i < 0 || i >= n) throw new IndexOutOfBoundsException("Illegal index: " + i);
    }

    public E get(int i) throws IndexOutOfBoundsException {
        checkIndex(i, size);
        return data[i];
    }

    // /** Replaces the element at index i with e, and returns the replaced element. */
    public E set(int i, E e) throws IndexOutOfBoundsException {
        checkIndex(i, size);
        E temp = data[i];
        data[i] = e;
        return temp;
    }
    // /** Inserts element e to be at index i, shifting all subsequent elements later. */
    public void add(int i, E e) throws IndexOutOfBoundsException, IllegalStateException {
        checkIndex(i, size + 1);
        // not enough capacity
        // if (size == data.length) throw new IllegalStateException("Array is full");
        if (size == data.length) resize(size*2);
        // start by shifting rightmost
        for (int k=size - 1; k >= i; k--) {
            data[k+1] = data[k];
        }
        // ready to place the new element
        data[i] = e;
        size++;
    }

    // /** Removes/returns the element at index i, shifting subsequent elements earlier. */
    public E remove(int i) throws IndexOutOfBoundsException {
        checkIndex(i, size);
        E temp = data[i];
        for (int k=i; k < size-1; k++){
            data[k] = data[k+1];
        }
        data[size-1] = null;
        size--;
        return temp;
    }

    // /** Resizes internal array to have given capacity >= size. */
    protected void resize(int capacity) {
        E[] temp = (E[]) new Object[capacity];
        for(int k=0; k<size; k++) temp[k] = data[k];
        data = temp;
    }

    public Iterator<E> iterator(){
        return new ArrayIterator();
    }

    //---------------- nested ArrayIterator class ----------------
    /**
    * A (nonstatic) inner class. Note well that each instance contains an implicit
    * reference to the containing list, allowing it to access the list's members.
    */
    private class ArrayIterator implements Iterator<E> {
        private int j=0;
        private boolean removable = false;

        /**
        * Tests whether the iterator has a next object.
        * @return true if there are further objects, false otherwise
        */
        public boolean hasNext() {return j<size;}
        public E next() throws NoSuchElementException{
            if(j==size) throw new NoSuchElementException("No next element.");
            removable=true;
            return data[j++];
        }
        public void remove() throws IllegalStateException{
            if(!removable) throw new IllegalStateException("nothing to remove");
            ArrayList.this.remove(j-1);
            j--;
            removable = false;
        }
    }
}
