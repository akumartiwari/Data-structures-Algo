import java.io.*;
import java.util.HashSet;
import java.util.Set;
import java.util.StringTokenizer;

public class ReverseConcat {
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
        ReverseConcat.MyScanner sc = new ReverseConcat.MyScanner();
        out = new PrintWriter(new BufferedOutputStream(System.out));

        int test = sc.nextInt();
        while (test-- > 0) {
            int n = sc.nextInt();
            int k = sc.nextInt();
            String s = sc.nextLine();
            Set<String> set = new HashSet<>();
            StringBuilder sb = new StringBuilder(s);
            set.add(sb.toString());
            StringBuilder sbe = new StringBuilder(sb);
            StringBuilder reve = new StringBuilder(sbe);
            reve.reverse();
            if (k == 0) System.out.println(1);
            else {
                if (sbe.toString().equals(reve.toString())) {
                    System.out.println(1);
                } else {
                    System.out.println(2);
                }
            }
        }
        out.close();
    }
}
