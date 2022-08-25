package com.company;

public class RANGECACHING {
    /*
    Input: s = "abc", shifts = [[0,1,0],[1,2,1],[0,2,1]]
    Output: "ace"
    Explanation: Firstly, shift the characters from index 0 to index 1 backward. Now s = "zac".
    Secondly, shift the characters from index 1 to index 2 forward. Now s = "zbd".
    Finally, shift the characters from index 0 to index 2 forward. Now s = "ace".
     */
    //The algo is based on RANGE CACHING/LINE SWEEP
    public String shiftingLetters(String s, int[][] shifts) {

        int[] count = new int[s.length() + 1];

        for (int[] shift : shifts) {
            int start = shift[0];
            int end = shift[1];
            int direction = shift[2];
            if (direction == 1) {
                count[start]++;
                count[end + 1]--;
            } else {
                count[start]--;
                count[end + 1]++;
            }
        }

        int temp = 0;
        StringBuilder sb = new StringBuilder();
        int idx = 0;
        for (int c : count) {
            if (idx >= s.length()) break;
            temp += c;
            int newChar = ((s.charAt(idx++) - 'a') + temp) % 26;
            sb.append((char) ('a' + (newChar < 0 ? newChar + 26 : newChar)));
        }

        return sb.toString();
    }
}
