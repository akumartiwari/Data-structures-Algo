package com.company;

import java.util.Scanner;

// TODO: Complete this function
public class MinorReduction {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int t = sc.nextInt();
        while (t-- > 0) {
            long number = sc.nextLong();
            reductions(number);
        }
    }

    private static void reductions(Long number) {

        String str = number.toString();
        int n = str.length();

        if (n == 1) {
            System.out.println(number);
        } else {
            long max = Long.MIN_VALUE;
            int len = n - 1;
            while (len >= 0) {
                String result = len >= 2 ? str.substring(0, len - 2) : "" +
                        ((int) str.charAt(len) + (len >= 1 ? (int) str.charAt(len - 1) : 0)) +
                        str.substring(len + 1);

                long newN = Long.MIN_VALUE;
                if (!result.isEmpty()) {
                    newN = Long.parseLong(result);
                }

                len--;
                max = Math.max(max, newN);
            }

            System.out.println(max);
        }
    }
}
