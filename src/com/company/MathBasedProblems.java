package com.company;

public class MathBasedProblems {
        /*
    Input: num = 58, k = 9
    Output: 2
    Explanation:
    One valid set is [9,49], as the sum is 58 and each integer has a units digit of 9.
    Another valid set is [19,39].
    It can be shown that 2 is the minimum possible size of a valid set.
     */

    //Author: Anand
    public int minimumNumbers(int num, int k) {
        if (num == 0) return 0;
        int ans = -1;
        int c = (num - k) / 10;

        while (c >= 0) {
            int elem = k + 10 * c;
            int i = 1;
            if (elem == 0) return -1;
            while ((num - (elem * i)) >= 0) {
                if (((num - elem * i) % 10) == 0) return 1;
                if (((num - elem * i) % 10) == k) return i + 1;
                i++;
            }
            c--;
        }
        return ans;
    }


    /*
    Input: brackets = [[3,50],[7,10],[12,25]], income = 10
    Output: 2.65000
    Explanation:
    The first 3 dollars you earn are taxed at 50%. You have to pay $3 * 50% = $1.50 dollars in taxes.
    The next 7 - 3 = 4 dollars you earn are taxed at 10%. You have to pay $4 * 10% = $0.40 dollars in taxes.
    The final 10 - 7 = 3 dollars you earn are taxed at 25%. You have to pay $3 * 25% = $0.75 dollars in taxes.
    You have to pay a total of $1.50 + $0.40 + $0.75 = $2.65 dollars in taxes.
     */
    public double calculateTax(int[][] brackets, int income) {
        int idx = 0;
        double tt = 0.0000;
        while (income > 0) {
            int ai = -1;
            if (idx == 0) {
                ai = income - brackets[idx][0] > 0 ? brackets[idx][0] : income;
            } else {
                ai = brackets[idx][0] - brackets[idx - 1][0] > 0 ? brackets[idx][0] - brackets[idx - 1][0] : income;
            }

            ai = Math.min(income, ai);
            int precent = brackets[idx][1];
            double nptax = Double.parseDouble(String.format("%.4f", (double) precent * ai / 100));
            tt += nptax;
            income -= ai;
            idx++;
        }

        return tt;
    }


    /*
    Input: current = "02:30", correct = "04:35"
    Output: 3
    Explanation:
    We can convert current to correct in 3 operations as follows:
    - Add 60 minutes to current. current becomes "03:30".
    - Add 60 minutes to current. current becomes "04:30".
    - Add 5 minutes to current. current becomes "04:35".
    It can be proven that it is not possible to convert current to correct in fewer than 3 operations.
   */
    // Author: Anand
    public int convertTime(String current, String correct) {
        long curMin = Integer.parseInt(current.split(":")[0]) * 60L + Integer.parseInt(current.split(":")[1]);

        long correctMin = Integer.parseInt(correct.split(":")[0]) * 60 + Integer.parseInt(correct.split(":")[1]);

        long diff = correctMin - curMin;

        int op = 0;
        while (diff != 0) {
            if (curMin + 60 <= correctMin) {
                curMin += 60;
            } else if (curMin + 15 <= correctMin) {
                curMin += 15;
            } else if (curMin + 5 <= correctMin) {
                curMin += 5;
            } else if (curMin + 1 <= correctMin) {
                curMin += 1;
            } else break;
            op++;
            diff = correctMin - curMin;
        }

        return op;

    }
}
