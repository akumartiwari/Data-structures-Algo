package CFProblems;

import java.io.*;
import java.util.*;

public class MakeItIncreasing {
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
        while (test-- > 0) {
            int n = sc.nextInt();
            long[] arr = new long[n];
            for (int i = 0; i < n; i++) {
                arr[i] = sc.nextInt();
            }

            if (!isValid(arr)) {
                continue;
            }

            long op = 0;
            boolean infinite = false;
            while (!increasing(arr)) {
                boolean flag = false;
                for (int i = 0; i < n - 1; i++) {
                    long pow = 0;
                    if (arr[i] >= arr[i + 1] && arr[i + 1] != 0) {
                        pow = (arr[i] / arr[i + 1] / 2) + 1;
                        op += pow;
                    }
                    if (pow >= 1) flag = true;
                    arr[i] /= Math.pow(2, pow);
                }

                if (!flag) {
                    System.out.println(-1);
                    infinite = true;
                    break;
                }
            }

            if (!infinite) System.out.println(op);
        }
        out.close();
    }

    // check if it's possible to make it increasing
    private static boolean isValid(long[] arr) {
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] < i) {
                System.out.println(-1);
                return false;
            }
        }

        return true;
    }

    private static boolean increasing(long[] arr) {
        for (int i = 0; i < arr.length - 1; i++)
            if (arr[i] >= arr[i + 1]) return false;
        return true;
    }
}
