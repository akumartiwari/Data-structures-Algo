package AdvancedRevision;

import java.util.HashMap;
import java.util.Map;


public class DigitDP {


    /*
     * -----------------------------------------------------------------------
     * PROBLEM (LeetCode 2801 - Hard): Count Stepping Numbers in Range
     * -----------------------------------------------------------------------
     * A "stepping number" is an integer where every pair of adjacent digits
     * differs by exactly 1 (absolute difference = 1).
     *   e.g.  1, 2, 3, ..., 9, 10, 12, 21, 23, 32, 34, 43, 45, ...
     *   NOT:  11 (|1-1|=0), 20 (|2-0|=2), 100 (|0-0|=0)
     *
     * Given two strings low and high (representing integers in the range),
     * return the COUNT of stepping numbers in [low, high], modulo 1e9+7.
     *
     * Example:
     *   Input : low = "1", high = "11"
     *   Output: 10
     *   Stepping numbers in [1,11]: 1,2,3,4,5,6,7,8,9,10  → total = 10
     *   (11 is excluded: |1-1| = 0, not a stepping number)
     *
     * -----------------------------------------------------------------------
     * WHAT IS DIGIT DP?
     * -----------------------------------------------------------------------
     * Digit DP is a technique to COUNT numbers in a range [low, high] that
     * satisfy some digit-level property, without enumerating each number.
     *
     * Core idea:
     *   - Build the answer number DIGIT BY DIGIT (from most-significant to least).
     *   - At each position, decide what digit to place (0–9).
     *   - Enforce an UPPER-BOUND "tight" constraint: if all previous digits match
     *     the upper limit, the current digit cannot exceed the limit's digit.
     *   - Track any extra STATE needed for the property (here: last digit placed).
     *   - MEMOIZE on (position, last_digit, is_tight, has_started) to avoid
     *     recomputing identical subproblems exponentially.
     *
     * Typical template:
     *   f(high)        = count of valid numbers in [0, high]
     *   f(low − 1)     = count of valid numbers in [0, low−1]
     *   answer         = f(high) − f(low−1)
     *
     * -----------------------------------------------------------------------
     * ALGORITHM (this implementation — top-down Digit DP with StringBuilder)
     * -----------------------------------------------------------------------
     * State:
     *   sb  — the digits chosen so far (current partial number being built)
     *   ind — the current digit-position index in `high` being considered
     *   dp  — memoization map keyed on (sb_string + "-" + ind)
     *
     * At each recursive call:
     *   BASE CASE:  ind < 0
     *     → If sb is non-empty (we placed at least one digit), this is a valid
     *       stepping number → return 1.
     *     → If sb is empty (no digit placed), return 0.
     *
     *   RECURSIVE CASE:
     *     For each digit position i from ind down to 0 (outer loop controls
     *     which position's upper-bound digit we're examining):
     *
     *       curr = high.charAt(i)   ← upper-bound digit at position i
     *
     *       For each candidate digit j from 0 to curr-1 (inner loop — all
     *       digits strictly below the bound, so "not tight" for this position):
     *
     *         TAKE j (place this digit):
     *           • If sb is empty (first digit): any j is accepted (start of number).
     *           • If sb is non-empty: only accept j if |sb.last - j| == 1
     *                                 (stepping property).
     *           → Append j, recurse for (ind-1), backtrack.
     *
     *         SKIP (don't place any digit at this position):
     *           → Recurse for (ind-1) without appending, allowing shorter numbers
     *             to also be counted.
     *
     *   MEMOIZE the result before returning.
     *
     * Step-by-step (low="1", high="11", starts with ind=1):
     *
     *   Call: helper(sb="", ind=1)
     *     Outer i=1, curr=high[1]='1' (ASCII 49)
     *     Inner j=0..48 (digits below '1'):
     *       j=0: sb empty → append '0', recurse ind=0 → ... (leading zero path)
     *       [no j satisfies 0 < '1' in meaningful digit range for 2-digit numbers]
     *
     *     Outer i=0, curr=high[0]='1' (ASCII 49)
     *     Inner j=0..48:
     *       j=48 ('0'): sb empty → append, recurse ind=-1
     *         ind<0, sb="0" (length>0) → return 1   [counts "0" but filtered by low]
     *       ...
     *       Single-digit stepping numbers 1-9 are built when sb="" → j placed as
     *       first digit, recurse to ind=-1, sb.length()>0 → return 1 each.
     *
     *   All valid stepping numbers ≤ "11" (i.e., 1–10) get counted = 10 ✓
     *
     * TC = O(D² × 10 × D)  where D = number of digits in high
     *      (D positions × 10 digits × D outer loop × memoized states)
     *      Effectively O(D³ × 10) — manageable since D ≤ ~20 for large numbers
     * SC = O(D × 10)        — memoization map entries bounded by distinct states
     * -----------------------------------------------------------------------
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
