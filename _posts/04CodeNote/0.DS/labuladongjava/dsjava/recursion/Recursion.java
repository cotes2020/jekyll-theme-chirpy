public class Recursion {

    // 5! = 5 · 4 · 3 · 2 · 1
    // 2! = 2 · 1
    // 1! = 1
    public static int factorial(int n) throws IllegalArgumentException {
        if(n<0) throw new IllegalArgumentException();
        else if(n==0) return 1;
        else return factorial(n-1) * n;
    }


    // Drawing an English Ruler
    public static void drawRuler(int nlnches, int majorLength) {
        drawLine(majotLength, 0);
        for(int j=1; j<=nlnches; j++){
            drawInterval(majorLength - 1);
            drawLine(majorLength, j);
        }
    }
}
