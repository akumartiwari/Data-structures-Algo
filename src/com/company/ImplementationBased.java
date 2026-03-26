package com.company;

import java.util.HashMap;
import java.util.Map;

public class ImplementationBased {

    // ══════════════════════════════════════════════════════════════════
    // #1  Minimum Consecutive Cards to Pick Up  (LeetCode 2260) — ★★★ Very Frequently Asked
    // ══════════════════════════════════════════════════════════════════
    /*
     * PROBLEM:
     *   Given an integer array cards, find the minimum number of consecutive
     *   cards you need to pick up to have a pair of matching cards.
     *   Return -1 if impossible.
     *
     *   Example:
     *     Input:  cards=[3,4,2,3,4,7]   Output: 4  (pick [3,4,2,3] → pair of 3s)
     *
     * SOLUTION — HashMap (last seen index):
     *   Track the last index each card was seen. If card is seen again,
     *   update ans = min(ans, i - lastSeen + 1).
     *   Time: O(n)  Space: O(n)
     */
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


    // ══════════════════════════════════════════════════════════════════
    // #2  Minimize Result by Adding Parentheses to Expression  (LeetCode 2232) — ★★ Frequently Asked
    // ══════════════════════════════════════════════════════════════════
    /*
     * PROBLEM:
     *   Given a string expression = "a+b", you can insert one pair of parentheses
     *   anywhere. Return the expression string that minimises the value of the result.
     *
     *   Example:
     *     Input:  expression="247+38"   Output: "2(47+38)"  → 2*(47+38)=170
     *
     * SOLUTION — Brute force all parenthesis placements:
     *   Try every split point left of '+' (j) and right of '+' (i).
     *   Compute left * (e1 + e2) * right; track minimum.
     *   Time: O(n²)  Space: O(n)
     */
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

    // ══════════════════════════════════════════════════════════════════
    // #3  Remove Digit From Number to Maximize Result  (LeetCode 2259) — ★★ Frequently Asked
    // ══════════════════════════════════════════════════════════════════
    /*
     * PROBLEM:
     *   Given a string number and a character digit, remove ONE occurrence of
     *   digit from number such that the resulting number is maximized.
     *
     *   Example:
     *     Input:  number="123", digit='1'   Output: "23"
     *     Input:  number="1231", digit='1'  Output: "231"
     *
     * SOLUTION — Try each occurrence and compare:
     *   For each occurrence of digit, remove it and compare lexicographically
     *   with the current maximum. Update max if larger.
     *   Time: O(n²)  Space: O(n)
     */
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
