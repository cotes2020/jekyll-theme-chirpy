package pq;

// /∗∗ Compares two strings according to their lengths. ∗/
public class StringLengthComparator<E> implements Comparator<String> {
    public int compare(String a, String b) {
      if (a.length() < b.length()) return -1;
      else if (a.length() == b.length()) return 0;
      else return 1;
    }
}
