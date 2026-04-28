package com.company;

import java.util.HashMap;import java.util.*;
import java.util.stream.Collectors;

public class Prime {

    /*
     * PROBLEM: Count Primes (LeetCode 204) (Helper)
     * Return true if n is a prime number, false otherwise.
     *
     * ALGORITHM: Trial division up to sqrt(n), checking odd divisors only
     * TC: O(√N) | SC: O(1)
     */
    public boolean isPrime(int n) {
        for (int i = 3; i < Math.sqrt(n); i += 2) {
            if (n % i == 0) return false;
        }
        return true;
    }

    /*
     * PROBLEM: Nth Prime Number (Helper)
     * Return the n-th prime number using sequential primality checks.
     *
     * ALGORITHM: Sequential isPrime calls until the n-th prime is found
     * TC: O(N * √P) where P is the n-th prime | SC: O(1)
     */
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

    /*
     * PROBLEM: Distinct Prime Factors of Product of Array (LeetCode 2521)
     * Return the count of distinct prime factors across all numbers in the array.
     *
     * ALGORITHM: Sieve-based prime factorization per element, union into a HashSet
     * TC: O(N * max(nums)) | SC: O(max(nums))
     */
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

    /*
     * PROBLEM: Count Primes (LeetCode 204) (Helper)
     * Fill array s where s[i] = smallest prime factor of i, using the Sieve of Eratosthenes.
     *
     * ALGORITHM: Sieve of Eratosthenes (smallest prime factor variant)
     * TC: O(N log log N) | SC: O(N)
     */
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
        /*
         * PROBLEM: Prime Pairs With Target Sum (LeetCode 2761)
         * Find all pairs of prime numbers (x, y) where x ≤ y and x + y == num.
         *
         * ALGORITHM: Sieve of Eratosthenes to enumerate primes, then check pairs
         * TC: O(N log log N) | SC: O(N)
         */
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

        /*
         * PROBLEM: Count Primes (LeetCode 204) (Helper)
         * Fill array s where s[i] = smallest prime factor of i (inner-class version).
         *
         * ALGORITHM: Sieve of Eratosthenes (smallest prime factor variant)
         * TC: O(N log log N) | SC: O(N)
         */
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


    /*
     * PROBLEM: Prime Factorization (Helper)
     * Factorize num using precomputed smallest-prime-factor array and collect distinct prime factors.
     *
     * ALGORITHM: Repeated division by smallest prime factor (sieve-based)
     * TC: O(log N) | SC: O(log N)
     */
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

    /*
     * PROBLEM: Prime Pairs With Target Sum (LeetCode 2761)
     * Find all pairs of prime numbers (x, y) where x ≤ y and x + y == num.
     *
     * ALGORITHM: Sieve of Eratosthenes + Two Pointers on sorted prime list
     * TC: O(N log log N + P) where P = number of primes ≤ N | SC: O(N)
     */
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
     * PROBLEM: Count Primes (LeetCode 204) (Helper)
     * Fill array s with smallest prime factors for integers up to num (long version for larger ranges).
     *
     * ALGORITHM: Sieve of Eratosthenes (smallest prime factor variant, long range)
     * TC: O(N log log N) | SC: O(N)
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

    /*
     * PROBLEM: Closest Prime Numbers in Range (LeetCode 2523)
     * Find the pair of prime numbers in [left, right] with the smallest gap; return [-1,-1] if fewer than 2 primes.
     *
     * ALGORITHM: Sieve of Eratosthenes to get primes in range, then scan for minimum gap
     * TC: O(N log log N) | SC: O(N)
     */
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
