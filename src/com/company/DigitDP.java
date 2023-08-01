package com.company;

import java.util.HashMap;
import java.util.Map;


public class DigitDP {


    /*
        Input: low = "1", high = "11"
        Output: 10
        Explanation: The stepping numbers in the range [1,11] are 1, 2, 3, 4, 5, 6, 7, 8, 9 and 10.
         There are a total of 10 stepping numbers in the range. Hence, the output is 10.
     */
    final int mod = 1_000_000_007;

    public int countSteppongStone(String low, String high) {
        Map<String, Integer> dp = new HashMap<>();
        return helper(low, high, new StringBuilder(), high.length() - 1, dp);
    }

    private int helper(String low, String high, StringBuilder sb, int ind, Map<String, Integer> dp) {
        // base case
        if (ind < 0) {
            if (sb.length() > 0) return 1;
            return 0;
        }

        int cnt = 0;
        String key = sb.toString() + "-" + ind;
        if (dp.containsKey(key)) return dp.get(key);

        // Traverse through all digits of high
        for (int i = ind; i >= 0; i--) {
            //take the curr digit
            int curr = high.charAt(i);
            for (int j = 0; j < curr; j++) {
                if (sb.length() == 0) {
                    sb.append(j);
                    cnt = (cnt + helper(low, high, sb, ind - 1, dp)) % mod;
                    //backtrack
                    sb.deleteCharAt(sb.length() - 1);
                } else {
                    if (Math.abs(sb.charAt(sb.length() - 1) - j) == 1) {
                        sb.append(j);
                        cnt = (cnt + helper(low, high, sb, ind - 1, dp)) % mod;
                        //backtrack
                        sb.deleteCharAt(sb.length() - 1);
                    }
                }

                //skip i.e. not-take the current digit
                cnt = (cnt + helper(low, high, sb, ind - 1, dp)) % mod;
            }
        }

        dp.put(key, cnt);
        return cnt;
    }
}
