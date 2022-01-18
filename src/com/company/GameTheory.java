package com.company;

import java.util.*;

public class GameTheory {

    // Author : Anand
    // TC = O(n)
    public List<String> generatePossibleNextMoves(String currentState) {
        List<String> ans = new ArrayList<>();
        return whoWins(currentState, ans);
    }

    private List<String> whoWins(String currentState, List<String> ans) {

        char[] charArray = currentState.toCharArray();
        for (int i = 0; i < currentState.length() - 1; i++) {
            if (charArray[i] == '+' && charArray[i + 1] == '+') {
                // we make a change
                charArray[i] = '-';
                charArray[i + 1] = '-';
                String newS = new String(charArray);
                if (!ans.contains(newS)) ans.add(newS);

                // Backtrack ie. check after skipping this step and make other moves
                charArray[i] = '+';
                charArray[i + 1] = '+';
            }
        }

        // In case if A is not able to move return false
        return ans;
    }

    //    Input: currentState = "++++"
    //    Output: true
    //    Explanation: The starting player can guarantee a win by flipping the middle "++" to become "+--+".
    // Iterate through string and check if opponent can win the after the current change
    // TC = O(n) where n = length of string
    // because for every index we need to check if opponent wins the game or not
    //Author : Anand
    // Lets' add memoization to improve TC
    public boolean canWin(String currentState) {
        Map<String, Boolean> dp = new HashMap<>();
        return whoWins(currentState, dp);
    }

    private Boolean whoWins(String currentState, Map<String, Boolean> dp) {

        // If already computed then use it
        if (dp.containsKey(currentState)) return dp.get(currentState);

        char[] charArray = currentState.toCharArray();
        for (int i = 0; i < currentState.length() - 1; i++) {
            if (charArray[i] == '+' && charArray[i + 1] == '+') {
                // we make a change
                charArray[i] = '-';
                charArray[i + 1] = '-';
                String newS = new String(charArray);
                // If now B can't win and hence A wins so return true
                if (!whoWins(newS, dp)) {
                    dp.put(currentState, true);
                    return true;
                }

                // Backtrack ie. check after skipping this step and make other moves
                charArray[i] = '+';
                charArray[i + 1] = '+';
            }
        }

        dp.put(currentState, false);
        // In case if A is not able to move return false
        return false;
    }
       /*
    We need ot track for all possible answers
    Solution:-
      1<=choosen<=desiredTotal
     choosen = {num,x, }
           num E [1, maxChoosableNumber]
           1<=num<=desiredTotal
    if A choose a number num then  we want to check if B can win with remaining total
    ie. desiredTotal-num

         Map<Integer, Boolean> mp --> Tell us after choosing x whether current player wins or not
         For every choice store the result in map
         x E [1, maxChoosableNumber]
         Subproblem:-
            if B chooses x (!choosen.contains(x)) desiredTotal-=x

         if (map.values.contains(true)) --> That was the most optimal step for curr player
         return true (curr)
         Recurse above step
         return the state ie, (a wins return true else return false)
        TC=O(maxChoosableInteger!) --> For each  number lyiung between 1 to maxChoosableInteger we can 2 choice(t/nt)
        and for each next iteration the number of operations reduced by 1
        Hence TC = O(n*(n-1)....3.2.1) ~ O(n!)
     */


    int[] dp = new int[1 << 21];

    public boolean canIWin(int maxChoosableInteger, int desiredTotal) {
        if (desiredTotal == 0) return true;
        int tot = (1 + maxChoosableInteger) * maxChoosableInteger / 2;
        if (tot < desiredTotal) return false;
        Arrays.fill(dp, -1);
        return dfs(0, maxChoosableInteger, desiredTotal, 0);
    }

    /****
     * DFS to traverse for all possibilities
     * @param sum
     * @param max
     * @param target
     * @param msk
     * @return
     */
    public boolean dfs(int sum, int max, int target, int msk) {
        // cache
        if (dp[msk] != -1) {
            return dp[msk] == 1;
        }
        // Already exhausted ie. previous player won and hence current lose
        if (sum >= target) {
            dp[msk] = 0;
            return false;
        }

        for (int i = 1; i <= max; i++) {
            // If number is taken
            if ((msk & (1 << i)) != 0) continue;
            //Take number and check if A wins with choice
            if (!dfs(sum + i, max, target, msk | (1 << i))) {
                dp[msk] = 1;
                return true;
            }
        }
        // backtrack
        dp[msk] = 0;
        return false;
    }
}