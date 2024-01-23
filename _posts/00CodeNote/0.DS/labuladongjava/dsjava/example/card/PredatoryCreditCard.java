package labuladongjava.dsjava.card;

public class PredatoryCreditCard extends CreditCard{
    // Additional instance variable
    private double apr;

    // Constructor for this class
    public PredatoryCreditCard(String cust, String bk, String acnt, int lim, double initialBal, double rate) {
        super(cust, bk, acnt, lim, initialBal); // initialize superclass attributes
        apr = rate;
    }

    // A new method for assessing monthly interest charges
    public void processMonth() {
        if (balance > 0) { // only charge interest on a positive balance
            double monthlyFactor = Math.pow(1 + apr, 1.0/12);
            // This is permitted precisely because the balance attributed was declared with protected visibility in the original CreditCard class.
            balance *= monthlyFactor;
        }
    }

    // Overriding the charge method defined in the superclass
    public boolean charge(double price) {
        boolean isSuccess = super.charge(price);
        if (!isSuccess)
            balance += 5;
        return isSuccess;
    }

}
