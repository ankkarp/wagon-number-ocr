

def is_valid(result):
    if not result:
        return 0

    cont_sum = 0
    control_num = -1

    if len(result) == 8:
        control_num = int(result[-1:])
        cont_sum = 0
        for i in range(7):
            num = int(result[i]) * (2, 1)[i % 2 == 1]
            if num >= 10:
                cont_sum += sum(list(map(int, set(str(num)))))
            else:
                cont_sum += num
    return int((cont_sum % 10 == 0 and control_num == 0) or (10 - cont_sum % 10) == control_num)

