
public class Solution {  

    public Boolean isPalindrome(String s) {
        s = s.replaceAll("[^A-Za-z0-9]", "").toLowerCase(); 
        System.out.println(s); 

        int r=0, l=s.length()-1; 

        while(r<l){

            System.out.println("r: " + r + ", l: " +l);
            System.out.println("r: " + s.charAt(r) + ", l: " +s.charAt(l));

            if(s.charAt(r)==s.charAt(l)){
                r++;
                l--;
            }
            else return Boolean.FALSE;
        } 
        return Boolean.TRUE;
    } 

    public static void main(String[] args) {
        Solution res = new Solution(); 
        // String s = "A man, a plan, a canal: Panama";
        String s = "amanaplanacanalpanama";
        Boolean ans = res.isPalindrome(s);
        System.out.println("ans:" + ans);
    } 
}
