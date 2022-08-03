package CFProblems;

import java.io.*;
import java.util.Arrays;
import java.util.StringTokenizer;

public class FibonacciAdditions {
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
        FibonacciAdditions.MyScanner sc = new FibonacciAdditions.MyScanner();
        out = new PrintWriter(new BufferedOutputStream(System.out));

        int n = sc.nextInt();
        int q = sc.nextInt();
        int MOD = sc.nextInt();
        int[] A = new int[n];
        int[] B = new int[n];

        for (int i = 0; i < n; i++) {
            A[i] = sc.nextInt();
        }

        for (int i = 0; i < n; i++) {
            B[i] = sc.nextInt();
        }

        int[] fib = new int[n];
        fib[0] = 1;
        fib[1] = 1;
        for (int i = 2; i < n; i++) {
            fib[i] = fib[i - 1] + fib[i - 2];
        }

        while (q-- > 0) {
            char c = sc.next().toCharArray()[0];
            int l = sc.nextInt();
            int r = sc.nextInt();

            boolean b = false;
            if (c == 'A') {
                // perform operation on A
                for (int i = l - 1; i < r; i++) {
                    A[i] = (A[i] + fib[i] + MOD) % MOD;
                    if (A[i] != B[i]) {
                        b = true;
                    }
                }
            } else {
                // perform on B
                for (int i = l - 1; i < r; i++) {
                    B[i] = (B[i] + fib[i] + MOD) % MOD;
                    if (B[i] != A[i]) {
                        b = true;
                    }
                }
            }

            if (b) System.out.println("NO");
            else System.out.println("YES");

            System.out.println(Arrays.toString(A));
            System.out.println(Arrays.toString(B));
        }
        out.close();
    }
}
