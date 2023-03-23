

public class CVigenereCipher {

    // convert the lower case character to Upper case
    static String LowerToUpper(String s) {
        StringBuffer str =new StringBuffer(s);
        for(int i = 0; i < s.length(); i++) {
            if(Character.isLowerCase(s.charAt(i))) {
                str.setCharAt(i, Character.toUpperCase(s.charAt(i)));
            }
        }
        s = str.toString();
        return s;
    }

    // generates the key in a cyclic manner until
    // it's length is equal to the length of original text
    // ABCABCABC
    static String generateKey(String str, String key) {
        int x = str.length();
        int i=0;
        while(key.length() != str.length()) {
            if (x == i) i = 0;
            key+=(key.charAt(i++));
        }
        return key;
    }

    // This function returns the encrypted text
    // generated with the help of the key
    static String cipherText(String str, String key) {
        String cipher_text="";
        for (int i = 0; i < str.length(); i++) {
            // converting in range 0-25
            int x = (str.charAt(i) + key.charAt(i)) %26;
            // convert into alphabets(ASCII)
            x += 'A';
            cipher_text+=(char)(x);
        }
        return cipher_text;
    }

    // This function decrypts the encrypted text
    // and returns the original text
    static String originalText(String cipher_text, String key) {
        String orig_text="";
        for (int i = 0 ; i < cipher_text.length() && i < key.length(); i++) {
            // converting in range 0-25
            int x = (cipher_text.charAt(i) - key.charAt(i) + 26) %26;
            // convert into alphabets(ASCII)
            x += 'A';
            orig_text+=(char)(x);
        }
        return orig_text;
    }

    // Driver code
    public static void main(String[] args) {
        String Str = "GEEKSFORGEEKS";
        String Keyword = "AYUSH";

        String str = LowerToUpper(Str);
        String keyword = LowerToUpper(Keyword);

        String key = generateKey(str, keyword);
        String cipher_text = cipherText(str, key);

        System.out.println("Ciphertext: " + cipher_text + "\n");
        System.out.println("Original/Decrypted Text: " + originalText(cipher_text, key));
        }
    }

// This code has been contributed by 29AjayKumar
