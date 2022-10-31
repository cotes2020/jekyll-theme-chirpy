

import java.io.*;
import java.util.*;

public class CPlayfair {
	String key;
	String plainText;
	LinkedHashSet<Character> set;
	String newKey;
	char[][] matrix = new char[5][5];

	public CPlayfair(String key){
		// convert all the characters to lowercase
		this.key = key.toLowerCase();
		this.set = new LinkedHashSet<Character>();
	}

	// function to remove duplicate characters from the key
	// function to generate playfair cipher key table
	// 只需要存储不重复的key，并不需要存储映射的value，
	public void generateCipherKey(){
		newKey = "";
		for (int i = 0; i < key.length(); i++) {
			if (key.charAt(i) != 'j') set.add(key.charAt(i));
		}
		Iterator<Character> it = set.iterator();
		while (it.hasNext()) newKey += (Character)it.next();

		// ABCxxxxxx
		for (int i = 0; i < 26; i++){
			String s = String.valueOf((char)(i + 97));
			if (s.equals("j")) continue;
			if (!newKey.contains(s)) newKey += s;
		}

		// create cipher key table
		for (int i = 0, idx = 0; i < 5; i++)
			for (int j = 0; j < 5; j++)
				matrix[i][j] = newKey.charAt(idx++);

		System.out.println("Playfair Cipher Key Matrix:");
		for (int i = 0; i < 5; i++) System.out.println(Arrays.toString(matrix[i]));
	}

	// function to preprocess plaintext
	public String formatPlainText(String plainText){
		plainText = plainText.toLowerCase();
		String message = "";
		int len = plainText.length();

		// if plaintext contains the character 'j',
		// replace it with 'i'
		for (int i = 0; i < len; i++){
			if (plainText.charAt(i) == 'j') message += 'i';
			else message += plainText.charAt(i);
		}

		// if two consecutive characters are same, then
		// insert character 'x' in between them
		for (int i = 0; i < message.length(); i += 2){
			if (message.charAt(i) == message.charAt(i + 1))
				message = message.substring(0, i + 1) + 'x' + message.substring(i + 1);
		}

		// make the plaintext of even length
		if (len % 2 == 1) message += 'x'; // dummy character

		return message;
	}

	// function to preprocess plaintext
	public String formatCipherText(String cipherText){
		String message = "";
		int len = cipherText.length();
		// if cipherText contains the character 'i',
		// replace it with 'j'
		for (int i = 0; i < len; i++){
			if (cipherText.charAt(i) == 'i') message += 'j';
			else message += cipherText.charAt(i);
		}
		return message;
	}

	// function to group every two characters
	public String[] formPairs(String message){
		int len = message.length();
		String[] pairs = new String[len / 2];

		for (int i = 0, cnt = 0; i < len / 2; i++)
			pairs[i] = message.substring(cnt, cnt += 2);

		return pairs;
	}

	// function to get position of character in key table
	public int[] getCharPos(char ch){
		int[] keyPos = new int[2];
		for (int i = 0; i < 5; i++){
			for (int j = 0; j < 5; j++) {
				if (matrix[i][j] == ch) {
					keyPos[0] = i;
					keyPos[1] = j;
					break;
				}
			}
		}
		return keyPos;
	}

	public CPlayfair(String key, String plainText){
		// convert all the characters to lowercase
		this.key = key.toLowerCase();
		this.plainText = plainText.toLowerCase();
		this.set = new LinkedHashSet<Character>();
	}

	public String encryptMessage(String plainText){
		String message = formatPlainText(plainText);
		String[] msgPairs = formPairs(message);
		String encText = "";

		// each pair
		for (int i = 0; i < msgPairs.length; i++){
			char ch1 = msgPairs[i].charAt(0);
			char ch2 = msgPairs[i].charAt(1);
			int[] ch1Pos = getCharPos(ch1);
			int[] ch2Pos = getCharPos(ch2);

			// if both the characters are in the same row
			if (ch1Pos[0] == ch2Pos[0]) {
				ch1Pos[1] = (ch1Pos[1] + 1) % 5;
				ch2Pos[1] = (ch2Pos[1] + 1) % 5;
			}

			// if both the characters are in the same column
			else if (ch1Pos[1] == ch2Pos[1]) {
				ch1Pos[0] = (ch1Pos[0] + 1) % 5;
				ch2Pos[0] = (ch2Pos[0] + 1) % 5;
			}

			// if both the characters are in different rows
			// and columns
			else {
				int temp = ch1Pos[1];
				ch1Pos[1] = ch2Pos[1];
				ch2Pos[1] = temp;
			}

			// get the corresponding cipher characters from
			// the key matrix
			encText = encText + matrix[ch1Pos[0]][ch1Pos[1]] + matrix[ch2Pos[0]][ch2Pos[1]];
		}
		return encText;
	}


	public String decryptMessage(String cipherText){
		String message = formatCipherText(cipherText);
		String[] msgPairs = formPairs(message);
		String encText = "";

		// each pair
		for (int i = 0; i < msgPairs.length; i++){
			char ch1 = msgPairs[i].charAt(0);
			char ch2 = msgPairs[i].charAt(1);
			int[] ch1Pos = getCharPos(ch1);
			int[] ch2Pos = getCharPos(ch2);

			// if both the characters are in the same row
			if (ch1Pos[0] == ch2Pos[0]) {
				ch1Pos[1] = (ch1Pos[1] +4) % 5;
				ch2Pos[1] = (ch2Pos[1] +4) % 5;
			}

			// if both the characters are in the same column
			else if (ch1Pos[1] == ch2Pos[1]) {
				ch1Pos[0] = (ch1Pos[0] +6) % 5;
				ch2Pos[0] = (ch2Pos[0] +6) % 5;
			}

			// if both the characters are in different rows
			// and columns
			else {
				int temp = ch1Pos[1];
				ch1Pos[1] = ch2Pos[1];
				ch2Pos[1] = temp;
			}

			// get the corresponding cipher characters from
			// the key matrix
			encText = encText + matrix[ch1Pos[0]][ch1Pos[1]] + matrix[ch2Pos[0]][ch2Pos[1]];
		}
		return encText;
	}


	public static void main(String[] args){
		System.out.println("Example-1\n");

		String key1 = "Problem";
		String plainText1 = "Playfair";
		System.out.println("Key: " + key1);
		System.out.println("PlainText: " + plainText1);

		CPlayfair pfc1 = new CPlayfair(key1);
		pfc1.generateCipherKey();
		String encText1 = pfc1.encryptMessage(plainText1);
		System.out.println("Cipher Text is: " + encText1);

		String decText1 = pfc1.decryptMessage(encText1);
		System.out.println("Plain Text is: " + decText1);
	}
}
