package labuladongjava;

import java.lang.module.ModuleDescriptor.Builder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Random;

import javax.sound.sampled.Mixer;
import javax.swing.plaf.synth.SynthSpinnerUI;

import labuladongjava.other.ListNode;

public class Solution {
    public String reverseOnlyLetters(String s) {
        StringBuilder res = new StringBuilder();
        for(int i=s.length()-1 ; i>=0 ; i--){
            if(Character.isLetter(s.charAt(i))) res.append(s.charAt(i)); 
        }
        for(int i=0 ; i<s.length() ; i++) {
        	if(!Character.isLetter(s.charAt(i))) res.insert(i, s.charAt(i));
        }
		return res.toString();
    }
    public static void main(String[] args) {
        String s = new String("7_28]");
        Solution res = new Solution();
        res.reverseOnlyLetters(s);
    } 
}
