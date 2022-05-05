package CF_Templates;

import java.io.*;
import java.util.*;

public class FoodforAnimals {
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
            int[] arr = new int[5];
            for (int i = 0; i < 5; i++) {
                arr[i] = sc.nextInt();
            }
            int dogs = arr[3];
            int cats = arr[4];

            dogs -= arr[0];
            cats -= arr[1];

            dogs = Math.max(dogs, 0);
            cats = Math.max(cats, 0);

            if ((dogs <= 0 && cats <= 0) || dogs + cats <= arr[2])
                System.out.println("YES");
            else System.out.println("NO");
        }
        out.close();
    }
}
