def min_edit_distance(str1, str2, insert_cost=1, delete_cost=1, replace_cost=1):
    m = len(str1)
    n = len(str2)

    # 创建一个 (m+1) x (n+1) 的矩阵
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # 初始化边界条件
    for i in range(m + 1):
        dp[i][0] = i * delete_cost
    for j in range(n + 1):
        dp[0][j] = j * insert_cost

    # 填充动态规划表
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                insert = dp[i][j - 1] + insert_cost
                delete = dp[i - 1][j] + delete_cost
                replace = dp[i - 1][j - 1] + replace_cost
                dp[i][j] = min(insert, delete, replace)

    # 回溯路径
    i, j = m, n
    path = []
    while i > 0 and j > 0:
        if str1[i - 1] == str2[j - 1]:
            path.append((str1[i - 1], 'match'))
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i - 1][j - 1] + replace_cost:
            path.append((str1[i - 1], 'replace', str2[j - 1]))
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i - 1][j] + delete_cost:
            path.append((str1[i - 1], 'delete'))
            i -= 1
        else:
            path.append((str2[j - 1], 'insert'))
            j -= 1

    while i > 0:
        path.append((str1[i - 1], 'delete'))
        i -= 1

    while j > 0:
        path.append((str2[j - 1], 'insert'))
        j -= 1

    path.reverse()
    return dp[m][n], path


# 主循环
while True:
    str1 = input("请输入字符串1：")
    str2 = input("请输入字符串2：")

    # 计算最小编辑距离和路径
    distance, path = min_edit_distance(str1, str2)
    print("字符串1：", str1)
    print("字符串2：", str2)
    print("最小编辑距离:", distance)
    print("最小编辑路径:")
    for step in path:
        print(step)

    # 询问用户是否继续
    continue_choice = input("continue(1) or end(0) ?")
    if continue_choice.lower() != '1':
        break