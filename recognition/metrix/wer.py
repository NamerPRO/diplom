def wer(recognized_text, reference_transcript):
    """
    Метод вычисляет значение метрики Word Error Rate (WER).
    WER - это метрика, используемая для оценки точности систем
    автоматического распознования речи. Она показывает, насколько
    сильно расшифрованный текст отличается от эталонного (правильного)
    текста.

    Формально вычисляется как:
    WER = (S + D + I) / N, где
        S - количество замененных слов.
        D - количество пропущенных слов.
        I - количество лишних вставленных слов.
        N - общее количество слов в эталонной транскрипции

    Аргументы:
        recognized_transcript: Текст, распознанный системой автоматического
            распознования речи.
        reference_transcript: Эталонный текст.

    Возвращаемое значение:
        Посчитанное значение метрики Word Error Rate.
    """
    reftext = reference_transcript.split(" ")
    rectext = recognized_text.split(" ")
    l1, l2 = len(reftext) + 1, len(rectext) + 1
    dp = [[0] * l2 for _ in range(l1)]
    for i in range(l1):
        dp[i][0] = i
    for j in range(l2):
        dp[0][j] = j
    for i in range(1, l1):
        for j in range(1, l2):
            if reftext[i - 1] == rectext[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i][j - 1] + 1, dp[i - 1][j] + 1, dp[i - 1][j - 1] + 1)

    return dp[-1][-1] / len(reftext)
