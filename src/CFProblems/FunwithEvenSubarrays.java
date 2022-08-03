package CFProblems;

import java.io.*;
import java.util.StringTokenizer;

public class FunwithEvenSubarrays {
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
    //--------------------------------------------------------

    // TC = O(exponential)
    // This can be solved via logic greedily
    public static void main(String[] args) {
        minMax.MyScanner sc = new minMax.MyScanner();
        out = new PrintWriter(new BufferedOutputStream(System.out));

        /*
        //code below
        int test = sc.nextInt();
        while (test-- > 0) {
            int n = sc.nextInt();
            int[] a = Arrays.stream(sc.nextLine().split(" ")).mapToInt(Integer::parseInt).toArray();
            int[] b = Arrays.stream(sc.nextLine().split(" ")).mapToInt(Integer::parseInt).toArray();
            TreeMap<Integer, Integer> tm1 = new TreeMap<>(Collections.reverseOrder());
            TreeMap<Integer, Integer> tm2 = new TreeMap<>(Collections.reverseOrder());

            for (int e : a) tm1.put(e, tm1.getOrDefault(e, 0) + 1);
            for (int e : b) tm2.put(e, tm2.getOrDefault(e, 0) + 1);

            Map<String, Integer> dp = new HashMap<>();
            System.out.println(max_min(n, a, b, 0, tm1, tm2, dp));
        }

        out.close();

         */
        // TC = O(n)
        int test = sc.nextInt();
        while (test-- > 0) {
            int n = sc.nextInt();

            int[] a = new int[n];
            int[] b = new int[n];
            for (int i = 0; i < n; i++) a[i] = sc.nextInt();
            for (int i = 0; i < n; i++) b[i] = sc.nextInt();


        }
        out.close();
    }


}
