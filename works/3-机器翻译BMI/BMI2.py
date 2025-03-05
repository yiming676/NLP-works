import os
import re
import numpy as np

# Function to initialize translation probabilities and alignment probabilities
def initialize_probabilities(english_sentences, chinese_sentences):
    t = {}
    q = {}

    # Initialize t with all possible word pairs, including 'NULL'
    for en_sentence, cn_sentence in zip(english_sentences, chinese_sentences):
        l_e = len(en_sentence)
        l_f = len(cn_sentence)

        for cn_word in cn_sentence:
            t[('NULL', cn_word)] = 1 / (l_f + 1)
            for en_word in en_sentence:
                t[(en_word, cn_word)] = 1 / (l_f + 1)

        for j in range(1, l_f + 1):
            for i in range(0, l_e + 1):
                q[(i, j, l_e, l_f)] = 1 / (l_e + 1)

    return t, q

# IBM Model 2 implementation
def ibm_model_2(english_sentences, chinese_sentences, num_iterations=10):
    t, q = initialize_probabilities(english_sentences, chinese_sentences)
    smoothing_factor = 1e-6  # Smoothing to avoid zero probabilities

    for _ in range(num_iterations):
        count_e_f = {}
        total_f = {}
        count_align = {}
        total_align = {}

        for en_sentence, cn_sentence in zip(english_sentences, chinese_sentences):
            l_e = len(en_sentence)
            l_f = len(cn_sentence)

            # E-step: Compute normalization
            s_total = {}
            for j, cn_word in enumerate(cn_sentence, 1):
                s_total[cn_word] = 0
                for i, en_word in enumerate(['NULL'] + en_sentence):
                    s_total[cn_word] += t.get((en_word, cn_word), smoothing_factor) * q.get((i, j, l_e, l_f), smoothing_factor)

            # Collect counts
            for j, cn_word in enumerate(cn_sentence, 1):
                for i, en_word in enumerate(['NULL'] + en_sentence):
                    delta = t.get((en_word, cn_word), smoothing_factor) * q.get((i, j, l_e, l_f), smoothing_factor) / s_total[cn_word]
                    count_e_f[(en_word, cn_word)] = count_e_f.get((en_word, cn_word), 0) + delta
                    total_f[cn_word] = total_f.get(cn_word, 0) + delta
                    count_align[(i, j, l_e, l_f)] = count_align.get((i, j, l_e, l_f), 0) + delta
                    total_align[(j, l_e, l_f)] = total_align.get((j, l_e, l_f), 0) + delta

        # M-step: Update probabilities with smoothing
        for (en_word, cn_word), count in count_e_f.items():
            if en_word != 'NULL':  # Skip updating probabilities for 'NULL' explicitly
                t[(en_word, cn_word)] = (count + smoothing_factor) / (total_f[cn_word] + smoothing_factor * len(t))

        for (i, j, l_e, l_f), count in count_align.items():
            q[(i, j, l_e, l_f)] = (count + smoothing_factor) / (total_align[(j, l_e, l_f)] + smoothing_factor * len(q))

    return t

# Preprocess sentences: clean and remove punctuation
def preprocess_sentence(sentence):
    sentence = re.sub(r'[^\w\s]', '', sentence)  # Remove punctuation
    sentence = sentence.lower()  # Convert to lowercase
    return sentence.split()

# Load and preprocess sentences from text files
def load_sentences(file_path):
    file_path = os.path.join(os.path.expanduser("~"), "Desktop", "IBM-DATA", file_path)
    with open(file_path, 'r', encoding='utf-8') as file:
        sentences = [preprocess_sentence(line.strip()) for line in file]
    return sentences

# Main function to run the translation model
def main():
    chinese_sentences = load_sentences('cn.txt')
    english_sentences = load_sentences('en.txt')

    translation_probabilities = ibm_model_2(english_sentences, chinese_sentences)

    # Print sample translation probabilities (filtering out NULL values and low probabilities)
    threshold = 0.1  # Only show probabilities above this threshold
    filtered_translations = [
        (en_word, cn_word, prob)
        for (en_word, cn_word), prob in translation_probabilities.items()
        if en_word != 'NULL' and prob > threshold
    ]

    # Sort by probabilities in descending order and print top results
    for en_word, cn_word, prob in sorted(filtered_translations, key=lambda x: -x[2])[:20]:
        print(f"P({en_word}|{cn_word}) = {prob:.4f}")

if __name__ == '__main__':
    main()





