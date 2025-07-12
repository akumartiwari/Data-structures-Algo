package com.company;

public class PractiseProblems {

    /*
    Check if str2 is subsequence of str1 after increment str1[i] to the next character cyclically at most once.
    Example1-
    Input: str1 = "abc", str2 = "ad"
    Output: true
    Explanation: Select index 2 in str1.
    Increment str1[2] to become 'd'.
    Hence, str1 becomes "abd" and str2 is now a subsequence. Therefore, true is returned.

    Example 2:
    Input: str1 = "zc", str2 = "ad"
    Output: true
    Explanation: Select indices 0 and 1 in str1.
    Increment str1[0] to become 'a'.
    Increment str1[1] to become 'd'.
    Hence, str1 becomes "ad" and str2 is now a subsequence. Therefore, true is returned.

     */
    public boolean canMakeSubsequence(String str1, String str2) {
        int i,j;
        for (i = 0, j = 0; i < str1.length() && j < str2.length(); i++) {
            char curr = str1.charAt(i);
            char next = (char) ('a' + (((str1.charAt(i) - 'a') + 1) % 26));
            if (curr == str2.charAt(j) || next == str2.charAt(j)) {
                j++;
            }
        }
        return j == str2.length();
    }
}
