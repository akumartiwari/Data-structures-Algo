package com.company;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.HashMap;import java.util.*;

public class Main {
    /*
     * PROBLEM: Entry point (Main)
     * Fast I/O template reading test cases from stdin and dispatching to logic methods.
     *
     * ALGORITHM: Fast I/O (BufferedReader + StringTokenizer)
     * TC: O(t * n log n) | SC: O(n)
     */
    public static void main(String[] args) {

        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        PrintWriter out = new PrintWriter(System.out, true);
        {
            try {
                int t = Integer.parseInt(br.readLine());
                while (t-- > 0) {
                    String firstLine = br.readLine();
                    int n = Integer.parseInt(firstLine.split(" ")[0]);
                    int k = Integer.parseInt(firstLine.split(" ")[1]);



                    /*If you want to read numbers on the same line use StringTokenizer*/
                    StringTokenizer st = new StringTokenizer(br.readLine());
                    Integer[] arr = new Integer[n];

                    for (int i = 0; i < n; ++i) {
                        arr[i] = Integer.parseInt(st.nextToken());
                    }


                    // function to execute logic
                    minimumPossibleScore(arr, 0, k);
                }


            } catch (IOException e) {
                e.printStackTrace();
            }
        }

    }

    /*
     * PROBLEM: Minimum Possible Score (Helper)
     * Compute minimum possible score by repeatedly taking floor division of the smallest/largest pair.
     *
     * ALGORITHM: Greedy + Sort
     * TC: O(n log n) | SC: O(n)
     */
    private static void minimumPossibleScore(Integer[] arr, int score, int k) {
        // Math.floor(ai/aj) to your score, where ⌊xy⌋ is the maximum integer not exceeding xy.
        Arrays.sort(arr);
        List<Integer> arrlist = new ArrayList<Integer>();

        Collections.addAll(arrlist, arr);
        while (arrlist.size() > 0 && k-- > 0) {
            int num = arrlist.get(0);
            int den = arrlist.get(arrlist.size() - 1);
            score += Math.floorDiv(num, den);
            arrlist.remove(0);
            arrlist.remove(arrlist.size() - 1);
        }
        // Add remaining elements of array as it is:-
        for (Integer elem : arrlist) {
            score += elem;
        }

        System.out.println(score);
    }
}
