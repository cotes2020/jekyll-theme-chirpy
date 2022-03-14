package list;

public interface Position<E> {
    E getElement() throws IllegalStateException;
}
