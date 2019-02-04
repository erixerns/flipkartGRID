def areaOfBox(l):
    return abs((l[2] - l[0]) * (l[3] - l[1]))


# list1 x1,y1,x2,y2
# list2 xmin,ymin,xmax,ymax


def getArea(list1, list2):
    # topleft
    f1 = list1[0] > list2[0] and list1[1] > list2[1] and list1[0] < list2[2] and list1[1] < list2[3]
    # topright
    f2 = list1[2] > list2[0] and list1[1] > list2[1] and list1[2] < list2[2] and list1[1] < list2[3]
    # bottomleft
    f3 = list1[0] > list2[0] and list1[3] > list2[1] and list1[0] < list2[2] and list1[3] < list2[3]
    # bottomright
    f4 = list1[2] > list2[0] and list1[3] > list2[1] and list1[2] < list2[2] and list1[3] < list2[3]

    if ((f1 or f2 or f3 or f4) == 0):
        return 0

    elif ((f1 and f2 and f3 and f4) == 1):
        return areaOfBox(list1)
    else:
        x1 = max(list1[0], list2[0])
        x2 = min(list1[2], list2[2])
        y1 = max(list1[1], list2[1])
        y2 = min(list1[3], list2[3])
        return areaOfBox([x1, y1, x2, y2])


# Example

area = getArea([0, 0, 10, 10], [5, 5, 20, 20])
print(area)
