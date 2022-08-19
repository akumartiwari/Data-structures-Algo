package CFProblems;

import java.io.*;
import java.util.Arrays;
import java.util.StringTokenizer;

public class BoringNumbers {
    //-----------PrintWriter for faster output---------------------------------
    public static PrintWriter out;

    //-----------MyScanner class for faster input----------
    public static class MyScanner {
        BufferedReader br;
        StringTokenizer st;

        public MyScanner() {
            br = new BufferedReader(new InputStreamReader(System.in));
        }

        String next() {
            while (st == null || !st.hasMoreElements()) {
                try {
                    st = new StringTokenizer(br.readLine());
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
            return st.nextToken();
        }

        int nextInt() {
            return Integer.parseInt(next());
        }

        long nextLong() {
            return Long.parseLong(next());
        }

        double nextDouble() {
            return Double.parseDouble(next());
        }

        String nextLine() {
            String str = "";
            try {
                str = br.readLine();
            } catch (IOException e) {
                e.printStackTrace();
            }
            return str;
        }
    }

    public static void main(String[] args) {
        MyScanner sc = new MyScanner();
        out = new PrintWriter(new BufferedOutputStream(System.out));

        int test = sc.nextInt();
        int cnt = 1;
        while (test-- > 0) {
            String elements = sc.nextLine();
            String[] arr = elements.split(" ");
            String L = arr[0];
            String R = arr[1];


            int[][] dp = new int[18][2]; // index, greater
            for (int[] d : dp) Arrays.fill(d, -1);

            int ansL = 1;
            for (int d = 1; d < L.length(); ++d) ansL *= 5;

            ansL = ansL == 1 ? countBoring(L, 0, 1, dp) : ansL + countBoring(L, 0, 1, dp);

            boolean lastTaken = true;
            for (int i = 0; i < L.length(); i++) {
                int place = i + 1;
                if (Integer.parseInt(String.valueOf(L.charAt(i))) % 2 == 0 && place % 2 == 0) continue;
                if (Integer.parseInt(String.valueOf(L.charAt(i))) % 2 != 0 && place % 2 != 0) continue;
                lastTaken = false;
                break;
            }

            if (lastTaken) --ansL;

            for (int[] d : dp) Arrays.fill(d, -1);

            int ansR = 1;
            for (int d = 1; d < R.length(); ++d) ansR *= 5;

            ansR = ansR == 1 ? countBoring(R, 0, 1, dp) : ansR + countBoring(R, 0, 1, dp);

            System.out.println("Case #" + cnt++ + ": " + (ansR - ansL));
        }
        out.close();
    }

    private static int countBoring(String num, int ind, int greater, int[][] dp) {

        // base case
        if (ind == String.valueOf(num).length()) return 1;

        if (ind > String.valueOf(num).length()) return 0;

        if (dp[ind][greater] != -1) return dp[ind][greater];

        int ans = 0;
        for (int d = 0; d < 9; ++d) {

            int place = ind + 1;
            if ((place % 2 == 0 && d % 2 == 0) || (place % 2 != 0 && d % 2 != 0)) {
                if (d < String.valueOf(num).charAt(ind) - '0') ans += countBoring(num, ind + 1, 0, dp);

                else if (d == String.valueOf(num).charAt(ind) - '0') ans += countBoring(num, ind + 1, greater, dp);

                else if (d > String.valueOf(num).charAt(ind) - '0' && greater == 0)
                    ans += countBoring(num, ind + 1, greater, dp);
            }
        }

        return dp[ind][greater] = ans;
    }
}
