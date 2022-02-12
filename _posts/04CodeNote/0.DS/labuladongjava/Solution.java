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
    public boolean backspaceCompare(String s, String t) {
        var pointerS = s.length() - 1;
        var pointerT = t.length() - 1;
    
        while (pointerS >= 0 || pointerT >= 0) {
            pointerS = movePointer(s, pointerS);
            pointerT = movePointer(t, pointerT);
    
            if (pointerS < 0 && pointerT < 0) // run out on both strings
                return true;
            if (pointerS < 0 || pointerT < 0) // run out on only one string
                return false;
            if (s.charAt(pointerS--) != t.charAt(pointerT--)) // character mismatch
                return false;
        }
        return true;
    }
    
    private int movePointer(String str, int pointer) {
        var backspaceCount = 0;
        while (pointer >= 0) {
            if (str.charAt(pointer) == '#') { // backspace seen
                backspaceCount++;
                pointer--;
            } else if (backspaceCount > 0) { // letter seen and there were backspaces before
                backspaceCount--;
                pointer--;
            } else {
                break; // letter seen and there were no backspaces before. We're done here
            } 
        }
        return pointer;
    }

    public static void main(String[] args) {
        Solution res = new Solution();
        String s = new String("aaaaab#c");
        String t = new String("baaaab#c");
        boolean anw = res.backspaceCompare(s,t);
        System.out.println(anw);
    } 
}
