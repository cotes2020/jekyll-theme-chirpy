

public class CHillCipher {

    // Following function generates the key matrix for the key string
    static void getKeyMatrix(String key, int keyMatrix[][]){
        int k = 0;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                keyMatrix[i][j] = (key.charAt(k)) % 65;
                k++;
            }
        }
    }

    // Following function encrypts the message
    static void encrypt(int cipherMatrix[], int keyMatrix[][], int messageVector[]) {
        int x, i;
        for (i = 0; i < 3; i++) {
            cipherMatrix[i] = 0;
            for (x = 0; x < 3; x++) {
                cipherMatrix[i] += keyMatrix[i][x] * messageVector[x];
            }
            cipherMatrix[i] = cipherMatrix[i] % 26;
        }
    }

    // Function to implement Hill Cipher
    static void HillCipher(String message, String key) {
        // Get key matrix from the key string
        int[][] keyMatrix = new int[3][3];
        getKeyMatrix(key, keyMatrix);

        // Generate vector for the message
        int[] messageVector = new int[3];
        for (int i = 0; i < 3; i++) messageVector[i] = (message.charAt(i)) % 65;

        // Following function generates
        // the encrypted vector
        int[] cipherMatrix = new int[3];
        encrypt(cipherMatrix, keyMatrix, messageVector);

        // Generate the encrypted text from the encrypted vector
        String CipherText="";
        for (int i = 0; i < 3; i++)  CipherText += (char)(cipherMatrix[i] + 65);
        // Finally print the ciphertext
        System.out.print(" Ciphertext: " + CipherText);
    }

    // Driver code
    public static void main(String[] args) {
        // Get the message to be encrypted
        String message = "ACT";

        // Get the key
        String key = "GYBNQKURP";

        HillCipher(message, key);

    }
}
