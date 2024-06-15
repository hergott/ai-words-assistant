import nltk
from wordfreq import top_n_list
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
import csv

# Ensure the required resources are downloaded
nltk.download('words')
nltk.download('wordnet')
nltk.download('stopwords')

def get_most_common_words(n=1500, remove_stopwords=False):
    # Get the top n most common English words using wordfreq
    common_words = top_n_list('en', n)
    # Filter out words containing apostrophes or escape characters
    filtered_words = [word for word in common_words if "'" not in word and "\\" not in word]
    
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in filtered_words if word.lower() not in stop_words]
    
    return filtered_words

def get_word_synonyms(word):
    synonyms = set()
    for synset in wn.synsets(word):
        for lemma in synset.lemmas():
            synonyms.add(lemma.name())
    # Remove the original word from the synonyms
    synonyms.discard(word)
    return synonyms

def save_words_to_csv(words, filename="common_words.csv"):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_NONE, escapechar='\\')
        for word in words:
            writer.writerow([word])

def save_words_with_synonyms_to_csv(words, filename="words_synonyms.csv"):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_NONE, escapechar='\\')
        for word in words:
            synonyms = get_word_synonyms(word)
            if synonyms:
                synonyms_str = ', '.join(synonyms)
                writer.writerow([f"{word}: {synonyms_str}"])
            else:
                writer.writerow([word])

if __name__ == "__main__":
    remove_stopwords_option = True  # Set this to False if you don't want to remove stopwords
    most_common_words = get_most_common_words(remove_stopwords=remove_stopwords_option)
    save_words_to_csv(most_common_words)
    save_words_with_synonyms_to_csv(most_common_words)
    print(f"Saved {len(most_common_words)} common words to 'common_words.csv'")
    print(f"Saved words with synonyms to 'words_synonyms.csv'")
