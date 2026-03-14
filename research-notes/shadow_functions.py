#!/usr/bin/env python3
import collections
import math
import functools
import itertools


def int_sqrt(n):
    """Return the integer square root of n."""
    k = n
    while k*k > n:
        k = (k + n//k)//2
    return k

def primes_below(limit):
    """Generate all primes below `limit` and yield them one-by-one.

    The Sieve of Eratosthenes is used here so this function will be
    reasonably fast but consume a lot of memory.
    """
    sieve = bytearray([True]) * len(range(1, limit, 2))
    sieve[1//2] = False
    for n in range(3, int_sqrt(limit-1)+1, 2):
        if sieve[n//2]:
            sieve[n*n//2 :: n] = bytes(len(range(n*n//2, len(sieve), n)))
    yield 2
    yield from itertools.compress(range(1, limit, 2), sieve)

# Generate some prime numbers on initialisation for use with the class
# `Factorisation`.
PRIME_LIMIT = 10**8
PRIMES = list(primes_below(PRIME_LIMIT))
PRIME_SET = set(PRIMES)

class Factorisation:
    """An integer which knows its prime factors.

    Integers must be at least 2 and less than PRIME_LIMIT**2 in absolute
    value.  Attempting to create an instance with an integer outside of
    this range will raise `ValueError()`.

    When printed, an instance of `Factorisation` will display its
    prime factors with powers.  For example:

    >>> print(Factorisation(20))
    2^2*5

    The attribute `factors` is a `collections.defaultdict` matching
    the prime factors to their corresponding powers.  This can be used,
    for example, to count the total number of prime factors.

    >>> fact = Factorisation(360)
    >>> sum(Factorisation(360).factors.values())
    6
    """

    def __init__(self, n):
        self.factors = collections.defaultdict(int)
        if abs(n) < 2:
            raise ValueError('abs(n) must be at least 2')
        if abs(n) >= PRIME_LIMIT**2:
            raise ValueError('n too large')
        self.sign = 1 if n > 0 else -1
        n *= self.sign
        for p in PRIMES:
            if n in PRIME_SET:
                self.factors[n] += 1
                return
            if p * p > n:
                break
            while not n % p:
                n //= p
                self.factors[p] += 1
        if n > 1:
            self.factors[n] += 1

    def __str__(self):
        sign = '' if self.sign == 1 else '-'
        return sign + '*'.join(f'{p}^{a}' if a > 1 else f'{p}' for (p, a) in self.factors.items())
    
    def __repr__(self):
        sign = '' if self.sign == 1 else '-'
        return sign + '*'.join(f'{p}^{a}' if a > 1 else f'{p}' for (p, a) in self.factors.items())        


def shadow(a, b, n):
    """The `(a, b)` shadow function.

    Return the number that follows `n` in the `(a, b)` shadow sequence along
    with its corresponding blueprint value.
    """
    n, R = divmod(-n, a)
    n *= -1
    if not R:
        n *= b
    return (n, R)

# Test cases for the function `shadow`.
assert shadow(2, 3, 5) == (3, 1)
assert shadow(2, 3, 14) == (21, 0)
assert shadow(5, 12, 43) == (9, 2)
assert shadow(6, 7, 42) == (49, 0)
assert shadow(3, 14, -12) == (-56, 0)
assert shadow(4, 6, -26) == (-6, 2)


def print_chain(a, b, n, m, blueprint=False, modulus=None, binary=False,
            factorisation=False, full_chain=True, summary=False):
    """Print the details of a chain with one row per iteration.

    :param int m:
        The length of the chain.
    :param bool blueprint:
        Whether or not to print blueprint values.
    :param int modulus:
        If not None, print the residue of each value with respect to
        this modulus.
    :param bool binary:
        If `True`, print values in binary.  If `False`, print values in
        decimal.
    :param bool factorisation:
        Whether or not to report the factorisation of each value.
    :param bool summary:
        Whether or not to print a summary at the end.

    Note: The capability of the class `Factorisation` is limited by the
    size of the sieve generated when this module is loaded.  Values
    which cannot be factorised will have their factorisations reported
    as `'???'`.
    """
    g = functools.partial(shadow, a, b)
    def factors(n):
        if not factorisation:
            return ''
        try:
            return str(Factorisation(n))
        except ValueError:
            return '???'
    template = f'i={{i:{len(str(m))}}}'
    if blueprint:
        template += f', R={{R:{len(str(a-1))}}}'
    if modulus is not None:
        template += f', r={{r:{len(str(modulus-1))}}}'
    template += ', n={n:b}' if binary else ', n={n}'
    if factorisation:
        template += ' ({factors})'

    chain = [n]
    R_indices = collections.defaultdict(set)
    r_indices = collections.defaultdict(set)
    for i in range(m):
        r = n % modulus
        r_indices[r].add(i)
        next_n, R = g(n)
        R_indices[R].add(i)
        if full_chain:
            print(template.format(i=i, R=R, r=r, n=n, factors=factors(n)))
        n = next_n
        chain.append(n)
    r = n % modulus
    r_indices[r].add(m)
    if full_chain:
        print(template.format(i=m, R='', r=r, n=n, factors=factors(n)))

    if summary:
        if blueprint and m:
            print('Blueprint:')
            for R in range(a):
                ratio = len(R_indices[R]) / m
                print(f'    {{R:{len(str(a-1))}}}: {{ratio:9.4%}}'
                      .format(R=R, ratio=ratio))
        if modulus is not None:
            print('Modulus:')
            for r in range(modulus):
                ratio = len(r_indices[r]) / (m + 1)
                print(f'    {{r:{len(str(modulus-1))}}}: {{ratio:9.4%}}'
                      .format(r=r, ratio=ratio))

class AttractorCache:
    """An environment for finding attractors for a given `a` and `b`."""

    def __init__(self, a, b, limit):
        self.g = functools.partial(shadow, a, b)
        self.limit = limit
        self.cache = {}

    def attractor(self, n):
        """Return the attractor `n` eventually reaches.

        Start from `n` and repeatedly apply the `(a, b)` shadow function
        `self.g`.  The value will eventually either settle into a cycle
        or reach `self.limit`.

        A cycle is an attractor, represented by the tuple `(s, m)` where
        `s` is the member of the cycle with minimum magnitude and `m` is
        the length of the cycle.  If the value eventually settles into a
        cycle then this attractor is returned along with the number of
        steps taken to get from `n` to a member of the cycle.

        If a value reaches or exceeds `self.limit` in absolute value at
        any point then we stop iterating and return the special
        attractor `(math.inf, math.inf)` along with the number of steps
        taken to reach this state from `n`.

        Every value encountered when iterating `n` will be recorded and
        their attractors will be calculated and cached before returning
        the result for `n` itself.  This can significantly speed up
        repeated calls at the expense of memory.
        """
        if abs(n) >= self.limit:
            return ((math.inf, math.inf), 0)
        if n in self.cache:
            return self.cache[n]
        chain = []
        chain_set = set()
        while True:
            chain.append(n)
            chain_set.add(n)
            n, _ = self.g(n)
            if abs(n) >= self.limit:
                for i, n in enumerate(reversed(chain), start=1):
                    self.cache[n] = ((math.inf, math.inf), i)
                return self.cache[chain[0]]
            if n in self.cache:
                end, num_steps = self.cache[n]
                for i, n in enumerate(reversed(chain), start=1):
                    self.cache[n] = end, num_steps + i
                return self.cache[chain[0]]
            if n in chain_set:
                loop_start = chain.index(n)
                loop = chain[loop_start:]
                loop_min = min((n for n in loop), key=abs)
                end = loop_min, len(loop)
                for n in loop:
                    self.cache[n] = (end, 0)
                for i, n in enumerate(reversed(chain[:loop_start]), start=1):
                    self.cache[n] = (end, i)
                return self.cache[chain[0]]

def clock_iter(a, b, q):
    """Return a map of each residue of `q` to its image under `g`.

    For each `r < q` we ask what `g(nq + r)` might be congruent to
    modulo `q`.
    """
    g = functools.partial(shadow, a, b)
    result = collections.defaultdict(set)
    for s in range(a * q // math.gcd(a, q)):
        n, __ = g(s)
        result[s % q].add(n % q)
    return result


if __name__ == '__main__':
    # Example 1 - Find attractors for a range of values.
    a = 2
    b = 5
    limit = 10**200
    search_space = range(10**3)
    print(f'Let `g` be the `({a}, {b})` shadow function.')
    print(f'Iterating all values in {search_space} with limit {limit:.2e}...')

    cache = AttractorCache(a, b, limit)
    attractor_origins = collections.defaultdict(set)
    for n in search_space:
        att, __ = cache.attractor(n)
        attractor_origins[att].add(n)
    if search_space:
        for att, origins in sorted(attractor_origins.items()):
            ratio = len(origins) / len(search_space)
            print(f'{ratio:9.4%} -- {att}')
        candidate = min(attractor_origins[(math.inf, math.inf)], key=abs)
        print('Smallest, seemingly divergent value:', candidate)
    print()

    # Example 2 - Print a single chain with all available details.
    n = candidate
    m = 20
    q = 3
    print(f'Now iterate `g` from {n} through {m} steps.')
    print(f'In what follows, `r` is the residue of `n` modulo {q}.')
    print_chain(a, b, n, m, blueprint=True, modulus=q, binary=False,
                factorisation=True, full_chain=True, summary=True)
    print()

    # Example 3 - Listing all the values that reach a particular attractor.
    attractor = (52, 7)
    print(f"Finally, list all the values in {search_space} that reach the"
          f" attractor {attractor} against the number of steps taken.")
    n_str_max = max(len(str(min(search_space))), len(str(max(search_space))))
    for n in attractor_origins[attractor]:
        __, num_steps = cache.attractor(n)
        print(f'    {{:{n_str_max}}}: {{}}'.format(n, num_steps))
