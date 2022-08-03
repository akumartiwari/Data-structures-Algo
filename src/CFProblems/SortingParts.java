import java.io.*;
import java.util.StringTokenizer;

public class SortingParts {
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
        SortingParts.MyScanner sc = new SortingParts.MyScanner();
        out = new PrintWriter(new BufferedOutputStream(System.out));

        int test = sc.nextInt();
        while (test-- > 0) {
            int n = sc.nextInt();
            int[] arr = new int[n];
            for (int i = 0; i < n; i++) {
                arr[i] = sc.nextInt();
            }

            int i;
            for (i = 1; i < n; i++) {
                if (arr[i - 1] > arr[i]) {
                    System.out.println("YES");
                    break;
                }
            }
            if (i == n) System.out.println("NO");
        }
        out.close();
    }
}
