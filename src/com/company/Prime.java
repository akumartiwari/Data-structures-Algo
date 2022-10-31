package com.company;

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

}
