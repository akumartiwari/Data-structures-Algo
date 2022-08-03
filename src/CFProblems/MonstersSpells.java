import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

// TODO : Solve this
public class MonstersSpells {
    int MOD = 1_00000_0000 + 7;

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int t = sc.nextInt();
        //	 System.setIn(new FileInputStream("Case.txt"));
        BufferedReader ob = new BufferedReader(new InputStreamReader(System.in));
        BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(System.out));

        while (t-- > 0) {
            int n = sc.nextInt();
            Integer[] k = Arrays.stream(sc.next().split(" ")).map(Integer::parseInt).toArray(Integer[]::new);
            Integer[] h = Arrays.stream(sc.next().split(" ")).map(Integer::parseInt).toArray(Integer[]::new);
            reductions(n, k, k);
        }
    }

    static class Pair {
        int first;
        int second;

        Pair(int f, int s) {
            this.first = f;
            this.second = s;
        }
    }

    // TC = O(n)
    private static void reductions(int n, Integer[] k, Integer[] h) {
        // Map to store spell and damage
        List<Pair> pairs = new ArrayList<>();

        for (int i = 0; i < n; i++) {
            int nk = k[i];
            int nh = h[i];
            if (i == 0 && nh > 1) {
                int second = pairs.size() > 0 ? (pairs.get(pairs.size() - 1)).second : 1;
                second += nh - 1;
                // calculate sum of integers till (nh-1)
                second += (int) Math.abs(nh - 1) * (nh - 2) / 2;

                for (int j = 0; true; j++) {
                    second += j + 1;
                    pairs.add(new Pair(j, nh)); // put {spell, damage}
                }
            } else if (nh > 1) {
                Pair last = pairs.size() > 0 ? (pairs.get(pairs.size() - 1)) : new Pair(1, 1);
                last.second++;

                if (last.first == nk) {
                    pairs.add(new Pair(nk, last.second));
                }
            }
        }
        System.out.println(pairs.size() > 0 ? (pairs.get(pairs.size() - 1)).second : 1);
    }
}
