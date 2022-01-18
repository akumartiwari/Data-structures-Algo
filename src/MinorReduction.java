import java.io.*;

public class MinorReduction {
    public static void main(String[] args) throws IOException {

        //	 System.setIn(new FileInputStream("Case.txt"));
        BufferedReader ob = new BufferedReader(new InputStreamReader(System.in));
        BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(System.out));

        int t = Integer.parseInt(ob.readLine());


        while (t-- > 0) {

	 	            	/*
	 	            	StringTokenizer st = new StringTokenizer(ob.readLine());
	 	            	int n = Integer.parseInt(st.nextToken());
	 	            	int n = Integer.parseInt(ob.readLine());
	 	            	int []ar = new int[n];

	 	            	 */

            //  int n = Integer.parseInt(ob.readLine());

            String s = ob.readLine();
            int f = -1;
            int sec = -1;
            int ind = -1;
            int ind2 = -1;
            for (int i = s.length() - 1; i >= 1; i--) {
                int a = s.charAt(i) - '0';
                int b = s.charAt(i - 1) - '0';
                if (a + b >= 10) {
                    int v = a + b;
                    f = 1;
                    sec = v % 10;
                    ind = i - 1;
                    ind2 = i;
                    break;
                }
            }

            int a = 2;
            char v = (char) (a + '0');

            if (f != -1) {
                StringBuilder sb = new StringBuilder();
                for (int i = 0; i < ind; i++) {
                    sb.append(s.charAt(i));
                }
                sb.append('1');
                char v1 = (char) (sec + '0');
                sb.append(v1);
                for (int i = ind2 + 1; i < s.length(); i++) {
                    sb.append(s.charAt(i));
                }
                String ans = sb.toString();
                System.out.println(ans);
            } else {
                int x = s.charAt(0) - '0';
                int y = s.charAt(1) - '0';
                char h = (char) (x + y + '0');
                StringBuilder sb = new StringBuilder();
                sb.append(h);
                for (int i = 2; i < s.length(); i++) {
                    sb.append(s.charAt(i));
                }

                String ans = sb.toString();
                System.out.println(ans);
            }
        }
    }
}
