import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

public class Amazon {
    // value
    private HashMap<Integer, int[]> map = new HashMap<>();
    private int offset = 0;
    private int index = 0;

    public void receive(int offsetnumber, int[] data) {
        // check the parameter
        // start it
        map.put(offsetnumber, data);
        System.out.println("adding: key: " + offsetnumber + ", value: " + Arrays.toString(data));
    }

    public void read(){
        Object ans;
        ArrayList<Object> res = readdata();
        if((Boolean) res.get(0)) ans = res.get(1);
        else ans = null;
        System.out.println(ans);
    }

    public ArrayList<Object> readdata(){
        ArrayList<Object> res = new ArrayList<Object>();
        if(map.containsKey(offset)) {
            res.add(true);
            int[] data = map.get(offset);
            int n = data.length;
            if(n==1){
                res.add(data[0]);
                // System.out.println("offset: " + offset + ", one value: " + ans);
                offset++;
                return res;
            }
            res.add(data[index]);
            if(index == n-1) {
                // System.out.println("offset: " + offset + ", last value: " + ans);
                index=0;
                offset++;
                return res;
            }
            // System.out.println("offset: " + offset + ", current value: " + ans);
            index++;
            return res;
        }
        // System.out.println("The offset is not stored: " + offset);
        res.add(false);
        return res;
    }

    public static void main(String[] args) {
        Amazon run = new Amazon();
        run.read();

        int offsetnumber = 1;
        int[] data = {12, 4};
        run.receive(offsetnumber, data);

        run.read();

        int offsetnumber2 = 0;
        int[] data2 = {100};
        run.receive(offsetnumber2, data2);

        run.read();
        run.read();
        run.read();
        run.read();
        run.read();
        run.read();
    }
}
