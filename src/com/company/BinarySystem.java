package com.company;

/*
 * ============================================================
 * Problem: Add Binary (LC 67) — Easy
 * ============================================================
 * DESCRIPTION:
 *   Given two binary strings s1 and s2, return their sum as a binary string.
 *   You must NOT convert them to integers directly.
 *
 * EXAMPLE:
 *   s1 = "110"  (= 6)
 *   s2 = "1000" (= 8)
 *   Output: "1110"  (= 14)
 *
 *   s1 = "1011"  (= 11)
 *   s2 = "100"   (= 4)
 *   Output: "1111"  (= 15)
 *
 * TWO APPROACHES IMPLEMENTED BELOW:
 *   1. Recursive  — processes digits from LSB to MSB via call stack
 *   2. Iterative  — processes digits from LSB to MSB using a loop + StringBuilder
 * ============================================================
 */
class BinarySystem {

    public static void main(String[] args) {
        String s1 = "110";// "0000100";// "110";// "1011"; // "101";
        String s2 = "1000";// "101";// "1000"; // "100";

        String res = "";
        res = add(s1.toCharArray(), s2.toCharArray(), s1.length(), s2.length(), 0, res);
        System.out.println(s1 + " + " + s2 + " = " + res);
    }

    /*
     * APPROACH 1 — Recursive
     * ---------------------------------------------------------------
     * ALGORITHM:
     *   1. Base case: both indices exhausted → prepend any remaining carry.
     *   2. Pick digit from A[i] and B[j] (treat out-of-bounds as 0).
     *   3. sum = a + b + carry
     *   4. Prepend (sum % 2) to result string, pass carry = sum / 2 recursively.
     *
     * TC: O(max(N, M))  — one recursive call per digit position
     * SC: O(max(N, M))  — call stack depth + result string
     * ---------------------------------------------------------------
     */
    private static String add(char[] A, char[] B, int i, int j, int carry, String result) {

        if (i < 0 && j < 0) {
            return (carry > 0 ? carry : "") + result;
        }

        int a = (i >= 0 ? A[i] - '0' : 0);
        int b = (j >= 0 ? B[j] - '0' : 0);
        int sum = a + b + carry;
        result = sum % 2 + result;

        return add(A, B, i - 1, j - 1, sum / 2, result);
    }

    /*
     * APPROACH 2 — Iterative (preferred)
     * ---------------------------------------------------------------
     * ALGORITHM:
     *   1. Use two pointers starting at the LSB (rightmost) of each string.
     *   2. For each step, extract digits (0 if pointer out of bounds).
     *   3. subsum = s1Val + s2Val + carryOver
     *   4. Append (subsum % 2) to StringBuilder; carry = subsum / 2.
     *   5. Reverse the StringBuilder at the end (built right-to-left).
     *
     * TC: O(max(N, M))  — single pass over both strings
     * SC: O(max(N, M))  — result StringBuilder
     * ---------------------------------------------------------------
     */
    public static String add(String s1, String s2) {
        int len = Math.max(s1.length(), s2.length());

        final StringBuilder result = new StringBuilder(len);
        int carryOver = 0;

        for (int s1Iter = s1.length() - 1, s2Iter = s2.length() - 1; s1Iter >= 0 || s2Iter >= 0; s1Iter--, s2Iter--) {
            final int s1Val = (s1Iter >= 0 ? s1.charAt(s1Iter) - '0' : 0), s2Val = s2Iter >= 0 ? s2.charAt(s2Iter) - '0' : 0;
            final int subsum = s1Val + s2Val + carryOver;
            result.append(subsum % 2);
            carryOver = subsum / 2;
            if (carryOver == 1) result.append(carryOver);
        }

        return result.reverse().toString();
    }

}
