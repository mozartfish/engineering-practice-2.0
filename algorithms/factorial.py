def iterativeFactorial(n):
    result = 1
    for i in range(1, n + 1):
        result *= i


def recursiveFactorial(n):
    if n == 0:
        return 1
    return n * recursiveFactorial(n - 1)


def isPalindrome(s):
    if len(s) <= 1:
        return True

    firstChar = s[0]
    lastChar = s[-1]
    if firstChar != lastChar:
        return False

    s = s[1:-1]

    return isPalindrome(s)
