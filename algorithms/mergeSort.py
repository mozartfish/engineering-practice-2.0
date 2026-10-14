def merge(array, start, mid, end):
    lowHalf = []
    highHalf = []

    for i in range(start, mid + 1):
        lowHalf.append(array[i])
    for j in range(mid + 1, end + 1):
        highHalf.append(array[j])

    i = 0
    j = 0
    k = start
    while i < len(lowHalf) and j < len(highHalf):
        if lowHalf[i] < highHalf[j]:
            array[k] = lowHalf[i]
            i += 1
        else:
            array[k] = highHalf[j]
            j += 1

        k += 1

    while i < len(lowHalf):
        array[k] = lowHalf[i]
        i += 1
        k += 1
    while j < len(highHalf):
        array[k] = highHalf[j]
        j += 1
        k += 1

    return array


def mergeSort(array, start, end):
    if end <= start:
        return array

    mid = start + (end - start) // 2
    mergeSort(array, start, mid)
    mergeSort(array, mid + 1, end)
    return merge(array, start, mid, end)
