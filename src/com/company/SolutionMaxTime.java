package com.company;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

class SolutionMaxTime {

    // Function to return the updated frequency map
    // for the array passed as argument
    static HashMap<Integer, Integer> getFrequencyMap(int arr[]) {
        HashMap<Integer, Integer> hashMap = new HashMap<>();
        for (int i = 0; i < arr.length; i++) {
            if (hashMap.containsKey(arr[i])) {
                hashMap.put(arr[i], hashMap.get(arr[i]) + 1);
            } else {
                hashMap.put(arr[i], 1);
            }
        }
        return hashMap;
    }

    // Function that returns true if the passed digit is present
    // in the map after decrementing it's frequency by 1
    static boolean hasDigit(HashMap<Integer, Integer> hashMap, int digit) {

        // If map contains the digit
        if (hashMap.containsKey(digit) && hashMap.get(digit) > 0) {

            // Decrement the frequency of the digit by 1
            hashMap.put(digit, hashMap.get(digit) - 1);

            // True here indicates that the digit was found in the map
            return true;
        }

        // Digit not found
        return false;
    }

    // Function to return the maximum possible time in 24-Hours format
    public String largestTimeFromDigits(int[] arr) {
        boolean iszero = false;
        for (int a : arr) {
            if (a == 0) iszero = true;
        }

        HashMap<Integer, Integer> hashMap = getFrequencyMap(arr);
        int i;
        boolean flag;
        String time = "";

        flag = false;

        // First digit of hours can be from the range [0, 2]
        for (i = 2; i >= 0; i--) {
            if (hasDigit(hashMap, i)) {
                flag = true;
                time += i;
                break;
            }
        }

        // If no valid digit found
        if (!flag) {
            return "";
        }

        flag = false;

        // If first digit of hours was chosen as 2 then
        // the second digit of hours can be
        // from the range [0, 3]
        if (time.charAt(0) == '2') {
            for (i = 3; i >= 0; i--) {
                if (hasDigit(hashMap, i)) {
                    flag = true;
                    time += i;
                    break;
                }
            }
        }

        // Else it can be from the range [0, 9]
        else {
            for (i = 9; i >= 0; i--) {
                if (hasDigit(hashMap, i)) {
                    flag = true;
                    time += i;
                    if (hasDigit(hashMap, 0)) time = "0" + time;
                    break;
                }
            }
        }
        if (!flag) {
            return "";
        }

        // Hours and minutes separator
        time += ":";

        flag = false;

        // First digit of minutes can be from the range [0, 5]
        for (i = 5; i >= 0; i--) {
            if (hasDigit(hashMap, i)) {
                flag = true;
                time += i;
                break;
            }
        }
        if (!flag) {
            return "";
        }

        flag = false;

        // Second digit of minutes can be from the range [0, 9]
        for (i = 9; i >= 0; i--) {
            if (hasDigit(hashMap, i)) {
                flag = true;
                time += i;
                break;
            }
        }
        if (!flag) {
            return "";
        }

        // Return the maximum possible time
        return time;
    }

    public static String getLargestTime(int[] input) {
        String largestTime = "00:00";
        String str = input[0] + "" + input[1] + "" + input[2] + "" + input[3];
        List<String> times = new ArrayList<>();
        permutation(str, times);
        Collections.sort(times, Collections.reverseOrder());
        for (String t : times) {
            int hours = Integer.parseInt(t) / 100;
            int minutes = Integer.parseInt(t) % 100;
            if (hours < 24 && minutes < 60) {
                if (hours < 10 && minutes < 10) {
                    largestTime = "0" + hours + ":0" + minutes;
                } else if (hours < 10) {
                    largestTime = "0" + hours + ":" + minutes;
                } else if (minutes < 10) {
                    largestTime = hours + ":0" + minutes;
                } else {
                    largestTime = hours + ":" + minutes;
                }
            }
        }
        return largestTime;
    }

    public static void permutation(String str, List<String> list) {
        permutation("", str, list);
    }

    private static void permutation(String prefix, String str, List<String> list) {
        int n = str.length();
        if (n == 0) list.add(prefix);
        else {
            for (int i = 0; i < n; i++)
                permutation(prefix + str.charAt(i), str.substring(0, i) + str.substring(i + 1, n), list);
        }
    }


}
