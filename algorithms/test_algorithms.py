def binarySearch(array, targetValue):
    minIndex = 0
    maxIndex = len(array) - 1

    while minIndex <= maxIndex:
        midIndex = minIndex + (maxIndex - minIndex) // 2
        if array[midIndex] > targetValue:
            maxIndex = midIndex - 1
        elif array[midIndex] < targetValue:
            minIndex = midIndex + 1
        else:
            return midIndex

    return -1


def selectionSort(array):
    for i in range(0, len(array)):
        minIndex = i
        for j in range(i + 1, len(array)):
            if array[j] < array[minIndex]:
                minIndex = j

        temp = array[i]
        array[i] = array[minIndex]
        array[minIndex] = temp

    return array


def iterativeFactorial(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result


def recursiveFactorial(n):
    if n == 0:
        return 1
    return n * recursiveFactorial(n - 1)


def insertionSort(array):
    for i in range(1, len(array)):
        key = array[i]
        j = i - 1

        while j >= 0 and array[j] > key:
            array[j + 1] = array[j]
            j -= 1

        array[j + 1] = key

    return array

def isPalindrome(s):
    if len(s) <= 1:
        return True

    firstChar = s[0]
    lastChar = s[-1]
    if firstChar != lastChar:
        return False

    s = s[1:-1]

    return isPalindrome(s)


def test_binarySearch():
    primes = [
        2,
        3,
        5,
        7,
        11,
        13,
        17,
        19,
        23,
        29,
        31,
        37,
        41,
        43,
        47,
        53,
        59,
        61,
        67,
        71,
        73,
        79,
        83,
        89,
        97,
    ]
    assert binarySearch(primes, 73) == 20
    assert binarySearch(primes, 2) == 0
    assert binarySearch(primes, 100) == -1


def test_selectionSort():
    array = [22, 11, 99, 88, 9, 7, 42]
    assert selectionSort(array) == [7, 9, 11, 22, 42, 88, 99]


def test_insertionSort():
    array = [22, 11, 99, 88, 9, 7, 42]
    assert insertionSort(array) == [7, 9, 11, 22, 42, 88, 99]

def test_iterativeFactorial():
    assert iterativeFactorial(0) == 1 
    assert iterativeFactorial(3) == 6 
    assert iterativeFactorial(4) == 24 
    assert iterativeFactorial(5) == 120

def test_recursiveFactorial():
    assert recursiveFactorial(0) == 1 
    assert recursiveFactorial(3) == 6 
    assert recursiveFactorial(4) == 24 
    assert recursiveFactorial(5) == 120 

def test_isPalindrome():
    assert isPalindrome("racecar") == True 
    assert isPalindrome("motor") == False 
    assert isPalindrome("rotor") == True 
    assert isPalindrome("toys") == False 



if __name__ == "__main__":
    print("hello world")
    test_binarySearch()
    test_selectionSort()
    test_insertionSort()
    test_iterativeFactorial() 
    test_recursiveFactorial()
    test_isPalindrome()
