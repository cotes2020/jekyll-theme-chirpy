public class GameEntry {
    private String name; // name of the person earning this score private
    int score; // the score value

    // /∗∗ Constructs a game entry with given parameters.. ∗/
    public GameEntry(String n, int s) {
        name = n;
        score = s;
    }
    public String getName() { return name; }
    public int getScore() { return score; }
    public String toString() {return"("+ name +","+ score +")";}
}
