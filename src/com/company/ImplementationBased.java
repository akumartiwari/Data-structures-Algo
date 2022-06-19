package com.company;

import java.util.HashMap;
import java.util.Map;

public class ImplementationBased {

    //Author: Anand
    public int minimumCardPickup(int[] cards) {
        int ans = Integer.MAX_VALUE;
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < cards.length; i++) {
            if (map.containsKey(cards[i])) {
                ans = Math.min(Math.abs((i - map.get(cards[i]) + 1)), ans);
            }
            map.put(cards[i], i);
        }
        return ans == Integer.MAX_VALUE ? -1 : ans;
    }


    // Author: Anand
    public String minimizeResult(String expression) {
        int n = expression.length();
        int idx = expression.indexOf('+');
        int mini = Integer.MAX_VALUE;
        String ans = "";
        for (int i = idx + 1; i < n; i++) {
            int e1 = Integer.parseInt(expression.substring(idx + 1, i + 1));
            for (int j = idx - 1; j >= 0; j--) {
                int e2 = Integer.parseInt(expression.substring(j, idx));
                int addition = e1 + e2;
                int left = 1, right = 1;
                if (!expression.substring(0, j).equals("")) left = Integer.parseInt(expression.substring(0, j));
                if (!expression.substring(i + 1).equals("")) right = Integer.parseInt(expression.substring(i + 1));

                int res = left * right * addition;
                if (res < mini) {
                    StringBuilder sb = new StringBuilder(expression);
                    mini = res;
                    sb.insert(j, '(');
                    sb.insert(i + 2, ')');
                    ans = sb.toString();
                }
            }
        }
        return ans;
    }

    //Author: Anand
    public String removeDigit(String number, char digit) {
        String maxi = "";

        for (int i = 0; i < number.length(); i++) {
            if (number.charAt(i) == digit) {
                String newNum = number.substring(0, i) + number.substring(i + 1);
                if (maxi.equals("")) {
                    maxi = newNum;
                    continue;
                }
                for (int j = 0; j < newNum.length(); j++) {
                    if (Integer.parseInt(String.valueOf(newNum.charAt(j))) > Integer.parseInt(String.valueOf(maxi.charAt(j)))) {
                        maxi = newNum;
                    } else if (Integer.parseInt(String.valueOf(newNum.charAt(j))) < Integer.parseInt(String.valueOf(maxi.charAt(j)))) {
                        break;
                    }
                }
            }
        }
        return maxi;
    }
}
