+++
author = "Vishnu PR"
categories = ["programming", "algorithms"]
date = "2017-04-17T14:46:03-07:00"
description = ""
featured = ""
featuredalt = ""
featuredpath = ""
linktitle = "Dynamic Programming"
title = "A Primer on Dynamic Programming"
+++

## Overview 

Not many who have left school use concepts such as **Dynamic Programming** on a day to day basis, only to come to them when preparing for an interview. However, it helps to understand these concepts, not because one may use them everyday, but they help look at old problems in new ways, hence helping find better solutions to even mundane problems one may face at work. This post should provide an introduction to Dynamic Programming and after reading this, it should be possible to attempt the problems provided at the end with ease.

Dynamic programming (or dynamic optimization) is a method for solving problems which can split into multiple simpler, smaller and overlapping (or repeating) sub-problems. Since the sub-problems are overlapping (meaning same sub-problems get repeated several times), it will be possible to save the solutions to them in a memory-based data structure. This helps reduce computation time at the expence of (hopefully moderate) storage space.  Each of the subproblem solutions is indexed in some way, typically based on the values of its input parameters, so as to facilitate its lookup. The technique of storing solutions to subproblems instead of recomputing them is called **Memoization**.

The two main requirements for a problem to be solved this way are **Optimal Substructure** and **Overlapping sub-problems**.

## Optimal Substructure

This simply means that the optimal solution to the actual problem is a combination of the optimal solutions to its sub-problems. This will become clearer as we look at Examples, but a quick look at Dijkstra's algorithm for single-source shortest path computation will help us understand this. In that, we repeatedly compute the shortest so-far from a source, by storing the vertices in a priority queue, weighted by their distance from the source. So the shortest distance from a node `s` to a node `n`  will be 
```
d(s,n) = d(s,m) + e(m,n) 
```
where `e(m,n)` is the length of the edge from m to n. In this case, the optimal substructure is that the shortest distance from s to n is the shortest distance from s to m + the edge from m to n. **Recursion** plays a big role in visualizing these problems and is usually the first approach to solving such problems. 

## Overlapping sub-problems

This requirement needs the problem to be split into a small space of sub-problems which would be solved over and over again in the process of getting to the solution of the problem - ie., we should be solving the same problems over and over again instead of solving newer and newer sub-problems. An example of **non-overlapping** subproblems is the merge-sort (and quicksort). Merge sort falls into the **Divide and Conquer** category, since we solve distinct subproblems and repeatedly merge the results. 

Example: To calculate Fibonacci numbers, `fib(4) = fib(3) + fib(2)` and `fib(3) = fib(2) + fib(1)` and so on. For each higher number in the series, if we solve this recursively, we will end up calculating the same lower-order problems over and over again. Such is a requirement for the problem to fall into this category.

## Memoization

If the above two requirements are met, then we can save or cache the results to be looked up later when we are solving the same sub-problems again and again. This will result in a huge drop in computation times at the expence of a small increase in memory space consumption. For example, in the fibonacci series problem, when calculating fib(4), we need fib(3) and fib(2). But, `fib(3) = fib(2) + fib(1)`. So fib(2) could be calculated twice here. If we cache fib(2) the first time it is calculated and then re-use it, this optimizes the computation. This is memoization.

## Solving the problem

There are two approaches to solving such problems - the *top-down* approach and the *bottom-up* approach.

### Top-down approach

This directly results from the recursion & memoization principles. We start from the problem and recursively solve the sub-problems while caching the results. Example: for fibonacci series, to get `fib(50)` - the 50th number in the series, we will calculate fib(49) and fib(48) and for `fib(49)`, we'll calculate fib(48) and fib(47) and so on recursively. If we think about this, then we can see that the first actual computation per se will be to compute the edge cases fib(0) and fib(1) and then fib(2) and so on until we get to fib(49) and fib(48). We recursively compute the smaller problems optimally and cache the results. This directly leads to the next approach.

### Bottom-up approach

In this approach, we start from a base case and iteratively build up to the solution we are looking for. Instead of recursively solving the problem, we formulate the sub problems, and solve them first before using them to build a solution for the problem itself. It's usually a table that stores the data needed with the indices somehow referring to the parameters of the problem itself. Taking the same example of Fibonacci numbers, if we need fib(50), we start at fib(0) and fib(1) as our base cases, and iteratively build the solutions to bigger problems from them to get to `fib(50)`.

## Examples

The key to solving these problems is identifying the recurrence correctly and what the table actually represents. Once that is done, formulating the table is trivial. 

### Fibonacci numbers

A good example is calculating Fibonacci numbers. `fib(n) = fib(n-1) + fib(n-2)`. A recursive approach to solving this would be:

```
func fibonacci(num int) int {
    if num == 0 || num == 1 {
        return 1
    }

    return fibonacci(num - 1) + fibonacci(num - 2)
}
```

We can see the calls to sub-problems will quickly explode computing the same problems over and over again, especially the smaller sub-problems - fib(2), fib(1), fib(0).

To avoid this, we could cache the solution.

``` 
func fibonacci(num int, cache map[int]int) int {
    if num == 0 || num == 1 {
        // base case
        return num
    }

    result := 0
    if result, ok := cache[num]; ok {
        return result
    }

    result = fibonacci(num - 1, cache) + fibonacci(num - 2, cache)
    cache[num] = result

    return result
}
```

We now store the solution and then use it if needed without having to recompute the sub-problems again. This leads to out next optimization where we start from the bottom-up.

```
func fibonacci(num int) int {
    // base cases
    num1 := 0
    num2 := 1

    fib := 0

    for i := 2; i <= num; i++ {
        fib = num1 + num2
        num1 = num2
        num2 = fib
    }

    return fib
}
```

### Coin change problems

There are two coin change problems. The two seem similar, but are subtly different.

#### Given an amount and a list of coin values (each with infinite coins), find out in how many ways we can make change for the amount.

When we have n coins, we need to count the number of ways we can make change with all combination of coins. Say we have `[1,2,5]`, we need to find out in how many ways we make change with `[1], [2], [5] [1,2], [1,5], [2,5], [1,2,5]`. But counting this way is not efficient. So we look at this problem another way. Each time we either use a coin or we don't and then we count the number of ways and then we sum them up. ie., 

```
NumberOfWays(Amount, coins[1...n]) = NumberOfWays(Amount - coin[n], coins[1...n]] + NumberOfWays(Amount, coins[1...n-1])
```

This way, we get all combination of coins. Not intuitive right away, but makes sense if we think about it. This leads to the following solution.

```
func change(amount int, coins []int) int {
    if amount < 0 {
        return 0
    }

    if amount == 0 {
        return 1
    }

    if len(coins) == 0 {
        return 0
    }

    length := len(coins)
    return change(amount, coins[:length-1]) + change(amount - coins[length-1], coins)
}
```

The top-down approach will be trivial. We simply need to cache the solution for `[amount][len(coins))]`. The bottom up approach is more interesting.

```
func change(amount int, coins []int) int {
    cache := make([][]int, amount+1)
    
    for i := 0; i <= amount; i++ {
        cache[i] = make([]int, len(coins) + 1)
    }

    for i := 0; i <= len(coins); i++ {
	    cache[0][i] = 1
    }

    for i := 1; i <= len(coins); i++ {
        for j := 1; j <= amount; j++ {
            if j - coins[i-1] >= 0 {
                //cached optimal sub-problem solution
                cache[j][i] = cache[j - coins[i-1]][i]
            }
            
            cache[j][i] = cache[j][i] + cache[j][i-1]
        }
    }
    return cache[amount][len(coins)]
}
```

#### Given an amount and a list of coin values (each with infinite coins), find out the minimum number of coins with which we can make change for the amount.

This problem can't be solved by looking at it like we did for the previous problem. We don't need all combinations of change-making. We only need one - the minimum #coins that can make the change. The subproblem here is that at each point, we use the best coin and then try and find the minimum number of coins required to change the remaining amount with ALL coins. This means we can't take the approach as in the previous variety, where we can use any coin.

If V is the amount to be changed, and there are n coins, the recursion here can be represented by

```
If V == 0, 0 coins are needed.

If V > 0 {
    minCoins(coins[0...n-1], V) = min {1 + minCoins(coins[0...n-1], V - coins[i])}
                                where V > coins[i] && i varies from 0...m-1.
}
```

The simple recursion approach will work by just calling the above. The top-down approach will work by caching the results. Caching will work, since the optimal sub-problem will always have the most optimal solution. In the earlier version, we cached against both the amount and the number of coins used. There we needed that since we were looking at all combinations. Now, we only calculate the minimum #coins, so we need to cache against only that.

Here is the bottom-up approach. Here we start by finding the minimum #coins needed to change 1, 2, ... so on until we hit V. At each step, we calculate the minimum for V by using `Solution(V-coin[i])`. This is very similar to how we arrive at the solution for Dijkstra's single source shortest path. There, to get to a node M, we find the shortest path to some N and then add the edge from N to M. This is very similar.

```
func coinChange(coins []int, amount int) int {
    if amount <= 0 {
        return 0
    }

    seen := make([]int, amount + 1)
    for i := 1; i <= amount; i++ { 
        min := 4111111111
        for _, coin := range coins {
            res := -1
            if i >= coin  {
                res = seen[i-coin] + 1
                if res < min {
                    min = res
                }
            }
        }   
        seen[i] = min
    }
    if seen[amount] == 4111111111 {
        return -1
    }

    return seen[amount]
}
``` 

### Knapsack problem

Given weights `W[0...n-1]` and the value of the weights `V[0...n-1]`, put the weights in a bag/knapsack of capacity W, so that the value is maximized. 

In this problem, we have one item of each weight and hence the subproblems will be when we either use an item or we don't. If `m[v,w]` is the maximum value attained with items [1...v] for a capacity w, then 

```
m[0,w] = 0 //no value for empty Value array
m[v,0] = 0 //no value for capacity=0
m[v,w] = max(m[v-1,w], m[v-1, w-W[v]] + V[v]) if w > V[v]
       = m[v-1,w] if w < V[v]

```
The basic recurrence will keep evaluating the sub-problems and hence we will cache the results. The bottom-up approach immediately follows from this.

```
func fillKnapsack(weights, values []int, capacity int) (maxValue int) {
    W := len(weights)
    V := len(Values)

    if W == 0 || V == 0 {
        return 0
    }
    cache := make([][]int, V + 1)
    for i, _ := range cache {
        cache[i] = make([]int, capacity + 1)
    }

    for i := 0; i <= V; i++ {
        for j := 0; j <= capacity; j++ {
            if i == 0 || j == 0 {
                continue
            }

            if W[i-1] <= capacity {
                cache[i][j] = cache[i-1][j]
            } else {
                x := cache[i-1][j]
                y := cache[i-1][j-W[i-1]] + V[i-1]

                if x > y {
                    cache[i][j] = x
                } else {
                    cache[i][j] = y
                }
            }
        }
    }
}
```

### Edit Distance

Given two character sequences, we need to find the edit-distance between them. Edit distance is used to measure the difference between the two sequences - ie., the number of operations required to transform one sequence into the other.

Now the operations(addition of a character, deletion of a character or the replacement of a character) can be weighed. For the sake of simplicity, our example here doesn't do that and assumes a cost of 1 for every operation.

**Levenshtein distance** is a popular measure of edit distance and this is a simplified form of it. The general recursive implementation will use the following logic. If D(i,j) is used to indicate the edit distance between two sequences I[1...i] and J[1...j], then,

```
//Edit distance for sequence and empty sequence is the length of the sequence itself.
D(i,0) = i  for 1 <= i <= len(I)
D(0,j) = j  for i <= j <= len(J)

if I[i] == J[j], D(i,j) = D(i-1,j-1)
else D(i,j) = min{D(i,j-1) + 1,   //insertion
                  D(i-1,j) + 1,   //deletion
                  D(i-1, j-1) + 1}//substitution
```

Implementing this using recursion will mean re-computing the same subproblems repeatedly. The optimization would be the following. This can be optimized further by saving only two rows at any time, but that is not done here for simplicity.

```
func min(A ...int) int {
    min := 4111111111
    for _, a := range A {
        if min > a {
            min = a
        }
    }
    return min
}

func minDistance(word1 string, word2 string) int {
    I := len(word1)
    J := len(word2)
    
    //Dij represents the edit distance between word1[0...i-1] and word2[0...j-1].
    //Do D0j = j, Di0 = i, since the edit distance between word[0...n-1] and an empty string is n+1
    D := make([][]int, I+1)
    for i, _ := range D {
        D[i] = make([]int, J+1)
    }
    for i := range D {
        D[i][0] = i
    }
    
    for j := range D[0] {
        D[0][j] = j
    }

    for i := 1; i <= I; i++ {
        for j := 1; j <= J; j++ {
            if word1[i-1] == word2[j-1] {
                D[i][j] = D[i-1][j-1]
            } else {
                D[i][j] = min(D[i][j-1] + 1, D[i-1][j] + 1, 1 + D[i-1][j-1])
            }
        }
    }
    return D[I][J]
}
```
