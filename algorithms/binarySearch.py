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
