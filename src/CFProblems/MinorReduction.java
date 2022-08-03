package CFProblems;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.Scanner;


public class MinorReduction {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int t = sc.nextInt();
        //	 System.setIn(new FileInputStream("Case.txt"));
        BufferedReader ob = new BufferedReader(new InputStreamReader(System.in));
        BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(System.out));

        while (t-- > 0) {
            String number = sc.next();
            reductions(number);
        }
    }

    private static void reductions(String str) {
        int n = str.length();
        int l = -1, r = -1, f = -1, s = -1;
        for (int i = n - 1; i >= 1; i--) {
            int nn = Integer.parseInt(String.valueOf(str.charAt(i))) + Integer.parseInt(String.valueOf(str.charAt(i - 1)));
            if (String.valueOf(nn).length() == 2) {
                f = 1;
                s = nn % 10;
                l = i - 1;
                r = i;
                break;
            }
        }
        StringBuilder sb = new StringBuilder();
        if (f != -1) {
            for (int i = 0; i < l; i++) {
                sb.append(str.charAt(i));
            }
            sb.append('1');
            sb.append((char) (s + '0'));

            for (int i = r + 1; i < n; i++) {
                sb.append(str.charAt(i));
            }
        } else {

            int x = str.charAt(0) - '0';
            int y = str.charAt(1) - '0';
            char h = (char) (x + y + '0');
            sb.append(h);
            for (int i = 2; i < n; i++) sb.append(str.charAt(i) - '0');
        }

        String ans = sb.toString();
        System.out.println(ans);
    }
}

