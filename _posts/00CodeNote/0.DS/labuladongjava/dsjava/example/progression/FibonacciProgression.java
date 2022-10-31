public class FibonacciProgression extends Progression {

    protected long pre;

    // /∗∗ Constructs traditional Fibonacci, starting 0, 1, 1, 2, 3, ... ∗/
    public FibonacciProgression() {
        this(0, 1);
    }

    // /∗∗ Constructs generalized Fibonacci, with give first and second values. ∗/
    public FibonacciProgression(long fir_num ,long sec_num) {
        super(fir_num);
        pre = sec_num - fir_num;
    }

    // /∗∗ Advances the current value to the next value of the progression. ∗/
    protected void advance() {
        long temp = pre;
        pre = current;
        current += temp;
    }
}
