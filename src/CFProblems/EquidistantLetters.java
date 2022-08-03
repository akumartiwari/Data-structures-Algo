package CFProblems;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Scanner;
import java.util.stream.Collectors;

public class EquidistantLetters {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int t = sc.nextInt();
        while (t-- > 0) {
            String str = sc.next();
            rearrange(str);
        }
    }

    private static void rearrange(String str) {
        // create a Map of letters and count
        Map<Character, Integer> freq = new HashMap<>();
        boolean isDuplicate = false;
        for (Character c : str.toCharArray()) {
            int newF = freq.getOrDefault(c, 0) + 1;
            if (newF >= 2) isDuplicate = true;
            freq.put(c, newF);
        }
        if (!isDuplicate) {
            System.out.println(str);
        } else {
            StringBuilder sb = new StringBuilder("");
            HashMap<Character, Integer> nm = sortByValue(freq);
            for (Map.Entry entry : nm.entrySet()) {
                int times = (int) entry.getValue();
                while (times-- > 0) sb.append(entry.getKey());
            }

            System.out.println(sb.toString());
        }
    }

    // function to sort hashmap by values
    public static HashMap<Character, Integer> sortByValue(Map<Character, Integer> hm) {
        HashMap<Character, Integer> temp
                = hm.entrySet()
                .stream()
                .sorted((i1, i2) -> i2.getValue().compareTo(i1.getValue()))
                .collect(Collectors.toMap(
                        Map.Entry::getKey,
                        Map.Entry::getValue,
                        (e1, e2) -> e1, LinkedHashMap::new));

        return temp;
    }
}
