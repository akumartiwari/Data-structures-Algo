package com.company;

class BinarySystem {

    public static void main(String[] args) {
        String s1 = "110";// "0000100";// "110";// "1011"; // "101";
        String s2 = "1000";// "101";// "1000"; // "100";

        String res = "";
        res = add(s1.toCharArray(), s2.toCharArray(), s1.length(), s2.length(), 0, res);
        System.out.println(s1 + " + " + s2 + " = " + res);

    }

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
