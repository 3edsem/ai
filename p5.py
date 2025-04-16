!pip install gensim transformers --quiet

import gensim.downloader as api
from transformers import pipeline
import nltk
import string
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('punkt_tab')


print("🔹 Loading pre-trained word vectors...")
word_vectors = api.load("glove-wiki-gigaword-100") 


def replace_keyword_in_prompt(prompt, keyword, word_vectors, topn=1):
    words = word_tokenize(prompt)
    enriched_words = []

    for word in words:
        cleaned_word = word.lower().strip(string.punctuation)
        if cleaned_word == keyword.lower():
            try:
                similar_words = word_vectors.most_similar(cleaned_word, topn=topn)
                if similar_words:
                    replacement_word = similar_words[0][0]
                    print(f"🔁 Replacing '{word}' → '{replacement_word}'")
                    enriched_words.append(replacement_word)
                    continue
            except KeyError:
                print(f"⚠️ '{keyword}' not found in the vocabulary. Using original word.")
        enriched_words.append(word)

    enriched_prompt = " ".join(enriched_words)
    print(f"\n🔹 Enriched Prompt: {enriched_prompt}")
    return enriched_prompt


print("\n🧠 Loading GPT-2 model...")
generator = pipeline("text-generation", model="gpt2")


def generate_response(prompt, max_length=100):
    try:
        response = generator(prompt, max_length=max_length, num_return_sequences=1)
        return response[0]['generated_text']
    except Exception as e:
        print(f"Error generating response: {e}")
        return None


original_prompt = "Who is king."
print(f"\n🔹 Original Prompt: {original_prompt}")
key_term = "king"


enriched_prompt = replace_keyword_in_prompt(original_prompt, key_term, word_vectors)

print("\n💬 Generating response for original prompt...")
original_response = generate_response(original_prompt)
print("\nOriginal Response:\n", original_response)

print("\n💬 Generating response for enriched prompt...")
enriched_response = generate_response(enriched_prompt)
print("\nEnriched Response:\n", enriched_response)


print("\n📊 Comparison:")
print("Original Length:", len(original_response))
print("Enriched Length:", len(enriched_response))
print("Original Sentences:", original_response.count("."))
print("Enriched Sentences:", enriched_response.count("."))


'''
=====================
Output: 
[nltk_data] Downloading package punkt to /root/nltk_data... [nltk_data] Package punkt is already up-to-date!
Loading pre-trained word vectors...
Loading GPT-2 model... 
Device set to use cpu 
Truncation was not explicitly activated but `max_length` is provided a specific value, please use `truncation=True` to explicitly truncate examples to max length. Defaulting to 'longest_first' truncation strategy. If you encode pairs of sequences (GLUE-style) with the tokenizer you can select this strategy more precisely by providing a specific strategy to `truncation`. 
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation. 
🔹 Original Prompt: Who is king. Replacing 'king' → 'prince'
🔹 Enriched Prompt: Who is prince . 
Generating response for the original prompt...
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation 
Original Prompt Response: 
Who is king. Is any one of them a son of God? He is the Lord of every kingdom. (3)--But, in the case of the Son of Man there was seen: how much is it with us to know that he is the Son of God? Now I am in an immolated body so that you could look with a clear eye at the Scriptures. And I was told by a man whom I know not whom you will come out, that the Lord had his own daughter 
Generating response for the enriched prompt... 
Enriched Prompt Response: 
Who is prince ...?" 
And this prince is one of the lords of the earth and of the kings of the world? The God of his Kingdom. 
He was an uncle who was named by God and was taken away by men for the adultery of some of them who had gone before him. 
And what is a prince? 
One who is Prince of heaven and of the earth, like him who comes to me with water in one hand and his Lord with 
Comparison of Responses: 
Original Prompt Response Length: 380 Enriched Prompt Response Length: 382 
Original Prompt Response Detail: 3 Enriched Prompt Response Detail: 5 
'''
