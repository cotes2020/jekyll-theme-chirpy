package labuladongjava.dsjava.card;

// import java.lang.System;

public class CreditCard {

    private String customer;
    private String bank;
    private String account;
    private int limit;
    protected double balance;

    // The first version requires five parameters,
    // including an explicit initial balance for the account.
    public CreditCard(String cust, String bk, String acnt, int lim, double initialBal) {
        customer = cust;
        bank = bk;
        account = acnt;
        limit = lim;
        balance = initialBal;
    }

    // The second constructor accepts only four parameters;
    // it relies on use of the special this keyword to invoke the five-parameter version, with an explicit initial balance of zero (a reasonable default for most new accounts).
    public CreditCard(String cust, String bk, String acnt, int lim) {
        this(cust, bk, acnt, lim, 0.0);
    }

    // Accessor methods:
    public String getCustomer() { return customer; }
    public String getBank() { return bank; }
    public String getAccount() { return account; }
    public int getLimit() { return limit; }
    public double getBalance() { return balance; } // Update methods:

    public boolean charge(double price) {
        if (price + balance > limit) return false;
        // at this point, the charge is successful
        balance += price; return true;
    }

    public void makePayment(double amount) {
        balance = balance - amount;
    }

    public static void printSummary(CreditCard card) {
        System.out.println("Customer = " + card.customer);
        System.out.println("Bank = " + card.bank);
        System.out.println("Account = " + card.account);
        System.out.println("Balance = " + card.balance);
        System.out.println("Limit = " + card.limit);
    }
}
