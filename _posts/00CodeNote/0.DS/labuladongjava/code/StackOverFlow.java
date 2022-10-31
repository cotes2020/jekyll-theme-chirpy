import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Scanner;

public class StackOverFlow {

    static Map<Integer, List<Integer>> map = new HashMap<Integer, List<Integer>>();

    static int[] parent = new int[100];
    static boolean[] visited = new boolean[100];

    public static void findingDistanceBetweenTwoNodes(int nodea, int nodeb){
        Queue<Integer> q = new LinkedList<Integer>();
        q.add(nodea);
        parent[nodea] = -1;
        visited[nodea] = true;
        boolean loop = true;
        while(!q.isEmpty() && loop){
            int element = q.remove();
            Integer s =null;
            while((s = getChild(element))!=null) {
                parent[s] = element;
                visited[s] = true;
                if(s == nodeb) {
                    loop= false;
                    break;
                }
                q.add(s);
            }
        }
        int x = nodeb;
        int d = 0;
        if(!loop) {
            while(parent[x] != -1) {
                d +=1;
                x = parent[x];
            }
            System.out.println("\nThe distance from node "+nodea+ " to node "+nodeb+" is "   +d);
        }
        else System.out.println("Can't reach node "+nodeb);
    }


    private static Integer getChild(int element) {
        ArrayList<Integer> childs = (ArrayList<Integer>) map.get(element);
        for (int i = 0; i < childs.size(); i++) {
            if( !visited[childs.get(i)] ) return childs.get(i);
        }
        return null;
    }


    public static void main(String[] args) {

        Scanner in = new Scanner(System.in);
        int n = in.nextInt();

        for (int i=1; i<=n; i++) {
            List<Integer> l1= new ArrayList<Integer>();
            map.put(i,l1);
        }
        for (int i=0; i<n-1; i++) {
            int u = in.nextInt();
            int v = in.nextInt();
            map.get(u).add(v);
            map.get(v).add(u);
        }
        findingDistanceBetweenTwoNodes(1, 3);
    }

}
