public class Progression {

    // instance variable
    protected long current;

    // /∗∗ Constructs a progression starting at zero. ∗/
    public Progression() { this(0); }

    // /∗∗ Constructs a progression with given start value. ∗/
    public Progression(long start) { current = start; }

    // /∗∗ Returns the next value of the progression. ∗/

    public long nextValue() {
        long answer = current;
        advance(); // this protected call is responsible for advancing the current value
        return answer;
    }

    // /∗∗ Advances the current value to the next value of the progression. ∗/
    protected void advance() {
        current++;
    }

    // /∗∗ Prints the next n values of the progression, separated by spaces. ∗/
    public void printProgression(int n) {
        // print first value without leading space
        System.out.print(nextValue());
        for (int j=1; j < n; j++) System.out.print(" " + nextValue( ));
        // print leading space before others
        System.out.println();
    }
}
