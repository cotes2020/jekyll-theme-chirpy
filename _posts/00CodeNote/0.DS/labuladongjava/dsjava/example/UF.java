package labuladongjava.other;

public class UF{
    private int count;
    private int[] parent;
    private int[] size;

    public UF(int n){
        parent = new int[n];
        size = new int[n];
        for(int i=0; i<n;i++){
            parent[i]=i;
            size[i]=1;
        }
        this.count = n;
    }

    public int find(int x) {
        while(parent[x]!=x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    }

    public int count() {
        return count;
    }

    public void union(int p, int q) {
        int rootP = find(p);
        int rootQ = find(q);
        if(size[rootP]>size[rootQ]){
            parent[q] = rootP;
            size[rootP] += size[rootQ];
        }
        if(size[rootQ]>size[rootP]){
            parent[p] = rootQ;
            size[rootQ] += size[rootP];
        }
    }

    public boolean connected(int p, int q) {
        int rootP = find(p);
        int rootQ = find(q);
        return rootP == rootQ;
    }

}
