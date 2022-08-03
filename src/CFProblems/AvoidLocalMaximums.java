import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.StringTokenizer;

public class AvoidLocalMaximums {
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
            int[] arr = new int[n];
            for (int i = 0; i < n; i++) {
                arr[i] = sc.nextInt();
            }

            solve(arr);
        }
        out.close();
    }

    private static void solve(int[] arr) {

        // Store local maximas index
        List<Integer> v = new ArrayList<>();
        for (int i = 1; i < arr.length - 1; i++) {
            // local maxima
            if (arr[i - 1] < arr[i] && arr[i + 1] < arr[i]) {
                v.add(i);
            }
        }


        List<Integer> w = new ArrayList<>();
        // Iterate all local maximas and check if there is a local minima b/w them
        for (int i = 0; i < v.size(); i++) {
            if (i + 1 < v.size() && v.get(i + 1) - v.get(i) == 2) {
                w.add(v.get(i) + 1);
                i++;
            } else {
                w.add(v.get(i));
            }
        }

        System.out.print(w.size());
        for (int i : w) {
            int best = Math.max(arr[i - 1], arr[i + 1]);
            arr[i] = best;
        }
        System.out.println();
        Arrays.stream(arr).forEach(x -> System.out.print(x + " "));
        System.out.println();
    }
}
