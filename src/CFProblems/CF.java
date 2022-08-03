import java.io.*;
import java.util.StringTokenizer;

public class CF {
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
            int x = sc.nextInt();
            int[] arr = new int[n];
            for (int i = 0; i < n; i++) {
                arr[i] = sc.nextInt();
            }

            solve(arr, x);
        }
        out.close();
    }

    private static void solve(int[] arr, int x) {


    }

    private static void antiFib(int n, int[] str) {
        for (int i = n; i >= 1; i--) {
            StringBuilder ans = new StringBuilder();
            StringBuilder left = new StringBuilder();
            StringBuilder right = new StringBuilder();

            // Let's fix i and generate subsequences
            for (int j = i - 2; j >= 0; j--) {
                left.append(str[j]);
                left.append(" ");
            }

            for (int j = str.length - 1; j >= i; j--) {
                right.append(str[j]);
                right.append(" ");
            }

            ans.append(i).append(" ");
            ans.append(right);
            ans.append(left);

            System.out.println();
            // print the string
            for (char c : ans.toString().toCharArray()) {
                System.out.print(c);
            }
        }

    }
}
