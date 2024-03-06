
// /∗∗ Interface for objects that can be sold. ∗/
public interface Sellable {

    // /∗∗ Returns a description of the object. ∗/
    public String description();
    // /∗∗ Returns the list price in cents. ∗/
    public int listPrice();
    // /∗∗ Returns the lowest price in cents we will accept. ∗/
    public int lowestPrice();

}
