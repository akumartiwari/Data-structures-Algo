import java.io.*;
import java.util.ArrayList;
import java.util.List;
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

    static int cnt = 0;

    // TODO
    public static void main(String[] args) {
        MyScanner sc = new MyScanner();
        out = new PrintWriter(new BufferedOutputStream(System.out));

        int test = sc.nextInt();
        while (test-- > 0) {
            int n = sc.nextInt();
            List<Integer> substr = new ArrayList<>();
            for (int i = 1; i <= n; i++) {
                substr.add(i);
            }
            solve(n, substr, new ArrayList<>());
        }
        out.close();
    }

    private static void solve(int n, List<Integer> substr, List<Integer> list) {
        if (substr.size() == 0) {
            boolean isValid = true;
            for (int i = 1; i < list.size() - 1; i++) {
                if (list.get(i - 1) + list.get(i) == list.get(i + 1)) {
                    isValid = false;
                    break;
                }
            }

            if (isValid && cnt < n) {
                System.out.println();
                list.forEach(x -> System.out.print(x + " "));
                cnt++;
            }
            return;
        }

        for (int i = 0; i < n; i++) {
            substr.remove(i);
            list.add(i+1);
            solve(n, substr, list);
            substr.add(i+1);
        }
    }
}
