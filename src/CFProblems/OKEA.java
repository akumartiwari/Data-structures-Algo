import java.io.*;
import java.util.*;

public class OKEA {
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
        OKEA.MyScanner sc = new OKEA.MyScanner();
        out = new PrintWriter(new BufferedOutputStream(System.out));

        int test = sc.nextInt();
        while (test-- > 0) {
            int n = sc.nextInt();
            int k = sc.nextInt();
            int odd = 0, even = 0;
            if (n * k % 2 == 0) {
                odd = (n * k) / 2;
                even = (n * k) / 2;
            } else {
                odd = 1 + ((n * k) / 2);
                even = (n * k) / 2;
            }

            // k is odd
            if (k % 2 != 0) {
                if (k != 1) {
                    // Pick either all odd or all even
                    int re = even % k;
                    int ro = odd % k;

                    if (re != ro) System.out.println("NO");
                    else {
                        System.out.println("YES");

                        Set<Integer> e = new HashSet<>();
                        Set<Integer> o = new HashSet<>();
                        for (int i = 1; i <= n * k; i++) {
                            if (i % 2 == 0) e.add(i);
                            else o.add(i);
                        }
                        int ie = 0, io = 0;

                        // printed all even elememts
                        for (int s : e) {
                            if (ie < k) System.out.print(s + " ");
                            else {
                                System.out.println();
                                System.out.print(s + " ");
                                ie = 1;
                            }
                        }


                        // printed all odd elememts
                        for (int s : o) {
                            if (ie < k) System.out.print(s + " ");
                            else {
                                System.out.println();
                                System.out.print(s + " ");
                                ie = 1;
                            }
                        }
                    }
                } else {
                    System.out.println("YES");
                    for (int i = 1; i <= n * k; i++) {
                        System.out.println(i);
                    }
                }
            } else {
                System.out.println("YES");
                Map<Integer, Boolean> map = new HashMap<>();
                for (int i = 1; i <= n * k; i++) {
                    map.put(i, false);
                }

                int ind = 0;
                int cnt = 0;
                for (int i = 1; i <= n * k; i++) {
                    if (cnt < n) {
                        if (ind <= k) {
                            System.out.print(i + " " + (i + 2));
                            System.out.println();
                            ind += 2;
                            cnt++;
                        } else {
                            ind = 2;
                        }
                    } else break;
                }
            }

        }
        out.close();
    }
}
