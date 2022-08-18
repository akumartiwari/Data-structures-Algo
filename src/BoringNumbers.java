package CFProblems;

import java.io.*;
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
        BoringNumbers.MyScanner sc = new BoringNumbers.MyScanner();
        out = new PrintWriter(new BufferedOutputStream(System.out));

        int test = sc.nextInt();
        int cnt = 1;
        while (test-- > 0) {
            int n = sc.nextInt();
            int[] arr = new int[n];
            for (int i = 0; i < n; i++) {
                arr[i] = sc.nextInt();
            }

            int L = arr[0];
            int R = arr[1];

            int nL = L;
            int ld = 0;
            while (nL > 0) {
                ld++;
                nL /= 10;
            }

            int nR = R;
            int rd = 0;
            while (nR > 0) {
                rd++;
                nR /= 10;
            }

            int ansL = 0;
            for (int d = 1; d < ld; ++d) ansL *= 5;
            ansL += countBoring(L, 1,1);

            int ansR = 0;
            for (int d = 1; d < ld; ++d) ansR *= 5;
            ansR += countBoring(R, 1,1);

            System.out.println("Case #" + cnt++ + ": " + (ansR-ansL));
        }
        out.close();
    }

    private static int countBoring(int L, int ind, int greater) {

        // base case
        if (ind == String.valueOf(L).length()) return 1;

        int ans = 0;
        for (int d=0;d<9;++d){

            if ((ind % 2 == 0 && d % 2 == 0) || (ind % 2 != 0 && d % 2 != 0)) {

                if (d < String.valueOf(L).charAt(ind))  ans += countBoring(L, ind+1, 0);

                else if (d == String.valueOf(L).charAt(ind)) ans += countBoring(L, ind+1, greater);

                else if (d > String.valueOf(L).charAt(ind) && greater == 0) ans += countBoring(L, ind+1, greater);
            }
        }

        return ans;
    }
}
