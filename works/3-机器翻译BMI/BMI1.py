import os
import re
import numpy as np

# Function to initialize translation probabilities uniformly
def initialize_probabilities(english_sentences, chinese_sentences):
    t = {}
    for en_sentence, cn_sentence in zip(english_sentences, chinese_sentences):
        for en_word in en_sentence:
            for cn_word in cn_sentence:
                t[(en_word, cn_word)] = 1 / len(cn_sentence)
    return t

# IBM Model 1 implementation with smoothing
def ibm_model_1(english_sentences, chinese_sentences, num_iterations=500):
    t = initialize_probabilities(english_sentences, chinese_sentences)
    smoothing_factor = 1e-6  # Add smoothing to avoid extreme probabilities

    for _ in range(num_iterations):
        count_e_f = {}
        total_f = {}

        for en_sentence, cn_sentence in zip(english_sentences, chinese_sentences):
            # E-step: Compute normalization
            s_total = {en_word: sum(t[(en_word, cn_word)] for cn_word in cn_sentence) for en_word in en_sentence}

            # Collect counts
            for en_word in en_sentence:
                for cn_word in cn_sentence:
                    count = t[(en_word, cn_word)] / s_total[en_word]
                    count_e_f[(en_word, cn_word)] = count_e_f.get((en_word, cn_word), 0) + count
                    total_f[cn_word] = total_f.get(cn_word, 0) + count

        # M-step: Update probabilities with smoothing
        for (en_word, cn_word), count in count_e_f.items():
            t[(en_word, cn_word)] = (count + smoothing_factor) / (total_f[cn_word] + smoothing_factor * len(t))

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

    translation_probabilities = ibm_model_1(english_sentences, chinese_sentences)

    # Print sample translation probabilities
    for (en_word, cn_word), prob in sorted(translation_probabilities.items(), key=lambda x: -x[1])[:20]:
        print(f"P({en_word}|{cn_word}) = {prob:.4f}")

if __name__ == '__main__':
    main()


