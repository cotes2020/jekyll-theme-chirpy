package pq;

// /∗∗ Compares two strings according to their lengths. ∗/
public class DefaultComparator<E> implements Comparator<String> {
    public int compare(String a, String b) throws ClassCastException {
      return ( (Comparable<E>) a ).compareTo(b);
    }
}
