package com.company;

import java.util.*;
import java.util.stream.Collectors;

public class Prime {

    // Write a function to return a prime or not
    // 4 , 7, 11
    public boolean isPrime(int n) {
        for (int i = 3; i < Math.sqrt(n); i += 2) {
            if (n % i == 0) return false;
        }
        return true;
    }

    public int nthPrimeNumber(int n) {
        int counter = 0;
        int number = 2;
        while (true) {
            if (isPrime(number)) {
                counter++;
            }
            number++;
            if (counter == n) break;
        }
        return number;
    }

    public int distinctPrimeFactors(int[] nums) {
        int prod = 1;
        Set<Integer> ans = new HashSet<>();
        for (int num : nums) {
            List<Integer> factors = new ArrayList<>();
            generatePrimeFactors(num, factors);

            // System.out.println(Arrays.toString(factors.toArray()));
            ans.addAll(factors);
        }

        return ans.size();
    }

    // Using SieveOfEratosthenes
    // to find smallest prime
    // factor of all the numbers.
    // For example, if N is 10,
    // s[2] = s[4] = s[6] = s[10] = 2
    // s[3] = s[9] = 3
    // s[5] = 5
    // s[7] = 7
    private void sieveOfEratosthenes(int num, int[] s) {
        // Create a boolean array
        // "prime[0..n]"  and initialize
        // all entries in it as false.
        boolean[] prime = new boolean[num + 1];

        // Initializing smallest
        // factor equal to 2
        // for all the even numbers
        for (int i = 2; i <= num; i += 2)
            s[i] = 2;

        // For odd numbers less
        // then equal to n
        for (int i = 3; i <= num; i += 2) {
            if (!prime[i]) {
                // s(i) for a prime is
                // the number itself
                s[i] = i;

                // For all multiples of
                // current prime number
                for (int j = i; j * i <= num; j += 2) {
                    if (!prime[i * j]) {
                        prime[i * j] = true;

                        // i is the smallest prime
                        // factor for number "i*j".
                        s[i * j] = i;
                    }
                }
            }
        }
    }

    class Solution {
        public List<List<Integer>> findPrimePairs(int num) {
            // smallest prime factor of i.
            int[] s = new int[num + 1];

            // Filling values in s[] using sieve
            sieveOfEratosthenes(num, s);

            Set<Integer> primeFac = new HashSet<>();
            for (int sn : s) primeFac.add(sn);
            List<Integer> primes = new ArrayList<>();
            primes.addAll(primeFac);
            Collections.sort(primes);
            System.out.println(primes.size());
            System.out.println(Arrays.toString(primes.toArray()));

            List<List<Integer>> ans = new ArrayList<>();
            for (int i = 0; i < primes.size(); i++) {
                for (int j = i; j < primes.size(); j++) {
                    if (i + j == num) ans.add(new ArrayList<>(Arrays.asList(i, j)));
                }
            }

            return ans;
        }

        private void sieveOfEratosthenes(int num, int[] s) {
            // Create a boolean array
            // "prime[0..n]"  and initialize
            // all entries in it as false.
            boolean[] prime = new boolean[num + 1];

            // Initializing smallest
            // factor equal to 2
            // for all the even numbers
            for (int i = 2; i <= num; i += 2)
                s[i] = 2;

            // For odd numbers less
            // then equal to n
            for (int i = 3; i <= num; i += 2) {
                if (!prime[i]) {
                    // s(i) for a prime is
                    // the number itself
                    s[i] = i;

                    // For all multiples of
                    // current prime number
                    for (int j = i; j * i <= num; j += 2) {
                        System.out.println(i * j);
                        if (!prime[(int) i * j]) {
                            prime[(int) i * j] = true;

                            // i is the smallest prime
                            // factor for number "i*j".
                            s[(int) i * j] = i;
                        }
                    }
                }
            }
        }
    }


    // Function to generate prime
    // factors and its power
    private void generatePrimeFactors(int num, List<Integer> factors) {
        // s[i] is going to store
        // smallest prime factor of i.
        int[] s = new int[num + 1];

        // Filling values in s[] using sieve
        sieveOfEratosthenes(num, s);

        // System.out.println("Factor Power");

        int curr = s[num]; // Current prime factor of N
        int cnt = 1; // Power of current prime factor

        // Printing prime factors
        // and their powers
        while (num > 1) {
            num /= s[num];

            // N is now N/s[N]. If new N
            // also has smallest prime
            // factor as curr, increment power
            if (curr == s[num]) {
                cnt++;
                continue;
            }

            // System.out.println("Factor=" + curr);
            factors.add(curr); // Add factor
            // System.out.println(curr + "\t" + cnt);

            // Update current prime factor
            // as s[N] and initializing
            // count as 1.
            curr = s[num];
            cnt = 1;
        }
    }

    public List<List<Integer>> findPrimePairs(int num) {
        // smallest prime factor of i.
        int[] s = new int[num + 1];

        // Filling values in s[] using sieve
        sieveOfEratosthenes((long) num, s);

        Set<Integer> primeFac = new HashSet<>();
        for (int sn : s) primeFac.add(sn);
        List<Integer> primes = new ArrayList<>();
        primes.addAll(primeFac);
        Collections.sort(primes);
        List<List<Integer>> ans = new ArrayList<>();

        int l = 1, r = primes.size() - 1;

        while (l <= r) {
            int sum = primes.get(l) + primes.get(r);
            if (primes.get(l) + primes.get(l) == num) {
                ans.add(new ArrayList<>(Arrays.asList(primes.get(l), primes.get(l))));
                l++;
                continue;
            }
            if (sum == num) {
                ans.add(new ArrayList<>(Arrays.asList(primes.get(l), primes.get(r))));
                l++;
            } else if (sum < num) l++;
            else r--;
        }

        return ans;
    }


    /*
    Input: left = 10, right = 19
    Output: [11,13]
    Explanation: The prime numbers between 10 and 19 are 11, 13, 17, and 19.
    The closest gap between any pair is 2, which can be achieved by [11,13] or [17,19].
    Since 11 is smaller than 17, we return the first pair.

    TC = O(NlogN)
    S = O(N)
     */

    private void sieveOfEratosthenes(long num, int[] s) {
        // Create a boolean array
        // "prime[0..n]"  and initialize
        // all entries in it as false.
        boolean[] prime = new boolean[(int) (num + 1L)];

        // Initializing smallest
        // factor equal to 2
        // for all the even numbers
        for (int i = 2; i <= num; i += 2)
            s[i] = 2;

        // For odd numbers less
        // then equal to n
        for (int i = 3; i <= num; i += 2) {
            if (!prime[i]) {
                // s(i) for a prime is
                // the number itself
                s[i] = i;

                // For all multiples of
                // current prime number
                for (int j = i; (long) j * i <= num; j += 2) {
                    if (!prime[i * j]) {
                        prime[i * j] = true;

                        // i is the smallest prime
                        // factor for number "i*j".
                        s[i * j] = i;
                    }
                }
            }
        }
    }

    public int[] closestPrimes(int left, int right) {
        int[] ans = new int[2];
        Arrays.fill(ans, -1);

        int[] s = new int[(int) ((long) right + 1L)];
        sieveOfEratosthenes((long) right, s);
        List<Integer> pn = Arrays.stream(s).boxed().distinct().sorted().collect(Collectors.toList());

        int last = -1, d = Integer.MAX_VALUE;
        for (int e : pn) {
            if (e < left) continue;
            if (last != -1 && e - last < d) {
                ans[0] = last;
                ans[1] = e;
                d = e - last;
            }
            last = e;
        }

        return ans;
    }

}
