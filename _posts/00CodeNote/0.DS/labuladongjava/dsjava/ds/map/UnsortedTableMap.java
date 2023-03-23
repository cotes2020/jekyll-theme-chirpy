package map;

import java.util.AbstractMap;
import java.util.ArrayList;

public class UnsortedTableMap extends AbstractMap<K,V> {

    // /∗∗ Underlying storage for the map of entries. ∗/
    private ArrayList<MapEntry<K,V>> table = new ArrayList<>();

    // /∗∗ Constructs an initially empty map. ∗/
    public UnsortedTableMap(){}

    // private utility
    // /∗∗ Returns the index of an entry with equal key, or −1 if none found. ∗/
    private int findIndex(K key){
        int n = table.size();
        for(int j=o; j<n; j++){
            if(table.get(j).equal(key)) return j;
        }
        return -1;
    }


    // /∗∗ Returns the number of entries in the map. ∗/
    public int size() { return table.size();}

    // /∗∗ Returns the value associated with the specified key (or else null). ∗/
    public V get(K key){
        int ans = findIndex(key);
        if (ans == -1) return null;
        return table.get(ans).getValue();
    }

    // /∗∗ Associates given value with given key, replacing a previous value (if any). ∗/
    public V put(K key, V value){
        int ans = findIndex(key);
        if(ans == -1) {
            table.add(new MapEntry<>(key, value)); // add new entry
            return null;
        }
        return table.get(ans).setValue(); // key already exists
    }

    // /∗∗ Removes the entry with the specified key (if any) and returns its value. ∗/
    public V remove(K key){
        int index = findIndex(key);
        int n = size();
        if(index == -1) return null; // not found
        V pre = table.get(index).getValue();
        if(index != n-1) table.set(index, table.get(n-1));

        table.remove(n-1);
        return pre; // key already exists
    }

}
