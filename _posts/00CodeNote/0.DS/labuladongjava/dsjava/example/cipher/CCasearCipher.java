

public class CCCasearipher {

    static String encrypt(String planTest, int key){
        StringBuffer sb = new StringBuffer();
        for(int i=0; i < planTest.length(); i++){
            char ch;
            int charvalue = (int)planTest.charAt(i) + key;
            if(Character.isWhitespace(planTest.charAt(i))){
                ch = planTest.charAt(i);
            }
            else if (Character.isUpperCase(planTest.charAt(i))){
                ch = (char)( (charvalue - 65) % 26 + 65 );
                // ch = (char)((charvalue-65)%26);
                // System.out.println("ch" + ch);
            }
            // else ch = (char)(charvalue);
            else ch = (char)( (charvalue - 97) % 26 + 97 );
            sb.append(ch);
        }
        return sb.toString();
    }

    static String decrypt(String cipherText, int key){
        StringBuffer sb = new StringBuffer();
        for(int i=0; i < cipherText.length(); i++){
            char ch;
            int charvalue = (int)cipherText.charAt(i) - key;
            if(Character.isWhitespace(cipherText.charAt(i))){
                ch = cipherText.charAt(i);
            }
            else if (Character.isUpperCase(cipherText.charAt(i))){
                if (65 > charvalue){
                    ch = (char)(charvalue + 26);
                    System.err.println(charvalue);
                    System.out.println("ch: " + ch);
                }
                else ch = (char)(charvalue);
                // ch = (char)( (charvalue) % 26 + 65 );
            }
            else {
                if (97 > charvalue){
                    ch = (char)(charvalue + 97);
                    System.err.println(charvalue);
                    System.out.println("ch: " + ch);
                }
                else ch = (char)(charvalue);
            }
                // ch = (char)( (charvalue - 97) % 26 + 97 );
            sb.append(ch);
        }
        return sb.toString();
    }

    // Driver code
    public static void main(String[] args) {
        String text = "ATTACKATONCEYy";
        int s = 4;
        String cipherTxt = encrypt(text, s);
        String decryptTxt = decrypt(cipherTxt, s);
        System.out.println("Text  : " + text);
        System.out.println("Shift : " + s);
        System.out.println("Cipher: " + cipherTxt);
        System.out.println("decrpty: " + decryptTxt);

        String text2 = "I am studying Data Encryption";
        String cipherTxt2 = encrypt(text2, s);
        String decryptTxt2 = decrypt(cipherTxt2, s);
        System.out.println("Text  : " + text2);
        System.out.println("Shift : " + s);
        System.out.println("Cipher: " + cipherTxt2);
        System.out.println("decrpty: " + decryptTxt2);



    }
}
