import java.io.*;
import java.util.Arrays;
import java.util.Map;
import java.util.StringTokenizer;
import java.util.TreeMap;

/*
Input:-
3
6
1 2 6 5 1 2
3 4 3 2 2 5
3
3 3 3
3 3 3
2
1 2
2 1
Output:-
18
9
2

 */
public class minMax {
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

            // Assume global max lies in b then minimise a after operation such that ax > bx E 0<=x<n
            for (int i = 0; i < n; i++) {
                // swap a[i], b[i]
                if (a[i] > b[i]) {
                    int temp = a[i];
                    a[i] = b[i];
                    b[i] = temp;
                }
            }

            System.out.println(calculate_max(a) * calculate_max(b));
        }
        out.close();
    }

    private static int calculate_max(int[] a) {
        int res = 0;
        for (int e : a) res = Math.max(res, e);
        return res;
    }

    private static int max_min(int n, int[] a, int[] b, int ind, TreeMap<Integer, Integer> tm1, TreeMap<Integer, Integer> tm2, Map<String, Integer> dp) {

        // base case
        if (ind >= n) return tm1.firstKey() * tm2.firstKey();

        int prevmaxa = tm1.firstKey();
        int prevmaxb = tm2.firstKey();

        String prevkey = prevmaxa + "-" + prevmaxb + "-" + ind;
        if (dp.containsKey(prevkey)) dp.get(prevkey);


        //  cases
        //  If nr. is swapped then only calclate maxa, maxb
//        case 1:- swapped

        tm2.put(b[ind], tm2.getOrDefault(b[ind], 0) - 1);
        if (tm2.get(b[ind]) <= 0) tm2.remove(b[ind]);

        tm1.put(a[ind], tm1.getOrDefault(a[ind], 0) - 1);
        if (tm1.get(a[ind]) <= 0) tm1.remove(a[ind]);

        tm1.put(b[ind], tm1.getOrDefault(b[ind], 0) + 1);
        tm2.put(a[ind], tm2.getOrDefault(a[ind], 0) + 1);

        String key = tm1.firstKey() + "-" + tm2.firstKey() + "-" + ind;
        if (dp.containsKey(key)) dp.get(key);

        int left = max_min(n, a, b, ind + 1, tm1, tm2, dp);
        // backtrack
        tm2.put(b[ind], tm2.getOrDefault(b[ind], 0) + 1);
        tm1.put(a[ind], tm1.getOrDefault(a[ind], 0) + 1);


        tm2.put(a[ind], tm2.getOrDefault(a[ind], 0) - 1);
        if (tm2.get(a[ind]) <= 0) tm2.remove(a[ind]);

        tm1.put(b[ind], tm1.getOrDefault(b[ind], 0) - 1);
        if (tm1.get(b[ind]) <= 0) tm1.remove(b[ind]);


        // case 2 :  no swap
        int right = max_min(n, a, b, ind + 1, tm1, tm2, dp);
        String nk = prevmaxa + "-" + prevmaxb + "-" + ind;
        int val = Math.min(left, right);
        dp.put(nk, val);
        return val;
    }

}
