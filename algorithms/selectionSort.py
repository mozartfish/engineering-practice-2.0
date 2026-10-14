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
