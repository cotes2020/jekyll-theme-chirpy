package array;


public class DynamicArrayList<E> implements List<E> { // instance variables {

    // default array capacity
    public static final int CAPACITY=16;
    // generic array used for storage
    private E[ ] data;
    // current number of elements
    private int size = 0;

    // constructors
    // constructs list with default capacity
    public DynamicArrayList() {
        this(CAPACITY);
    }

    // constructs list with given capacity
    public DynamicArrayList(int capacity2) {
    }

    public void ArrayList(int capacity) {
        // safe cast; compiler may give warning
        data = (E[ ]) new Object[capacity];
    }
    // // public methods
    // /∗∗ Returns the number of elements in the array list. ∗/
    public int size() { return size; }

    // /∗∗ Returns whether the array list is empty. ∗/
    public boolean isEmpty() { return size == 0; }

    // // utility method
    // /∗∗ Checks whether the given index is in the range [0, n-1]. ∗/
    protected void checkIndex(int i, int n) throws IndexOutOfBoundsException {
        if (i < 0 || i >= n) throw new IndexOutOfBoundsException("Illegal index: " + i);
    }

    // /∗∗ Returns (but does not remove) the element at index i. ∗/
    public E get(int i) throws IndexOutOfBoundsException {
        checkIndex(i, size);
        return data[i];
    }
    // /∗∗ Replaces the element at index i with e, and returns the replaced element. ∗/
    public E set(int i, E e) throws IndexOutOfBoundsException {
        checkIndex(i, size);
        E temp = data[i];
        data[i] = e;
        return temp;
    }
    // /∗∗ Inserts element e to be at index i, shifting all subsequent elements later. ∗/
    // public void add(int i, E e) throws IndexOutOfBoundsException, IllegalStateException {
    //     checkIndex(i, size + 1);
    //     // not enough capacity
    //     if (size == data.length) throw new IllegalStateException("Array is full");
    //     // start by shifting rightmost
    //     for (int k=size-1; k >= i; k--) {
    //         data[k+1] = data[k];
    //     }
    //     // ready to place the new element
    //     data[i] = e;
    //     size++;
    // }
    public void add(int i, E e) throws IndexOutOfBoundsException {
        checkIndex(i, size + 1);
        // not enough capacity
        if (size == data.length) resize(2*data.length);
        // so double the current capacity
    }


    // /∗∗ Removes/returns the element at index i, shifting subsequent elements earlier. ∗/
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

    // /∗∗ Resizes internal array to have given capacity >= size. ∗/
    protected void resize(int capacity) {
        // safe cast; compiler may give warning
        E[] temp = (E[]) new Object[capacity];
        for (int k=0; k < size; k++) temp[k] = data[k];
        data = temp; // start using the new array
    }


}
