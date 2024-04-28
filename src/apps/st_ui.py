import streamlit as st
import pandas as pd
import nltk
nltk.download('punkt')

from nltk.tokenize import sent_tokenize

# from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration

# tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
# retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", dataset=None, index_name='exact', use_dummy_dataset=True)
# model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)


@st.cache_data(func=None, hash_funcs=None)
def load_data(file):
    """
    Load data from a specified CSV file
    """
    data = pd.read_csv(file)
    data = data.applymap(str)  # apply str() to each cell
    data["Title len"] = data["Title"].apply(lambda x: len(x))

    return data

def validate_data(data):
    """
    Check if any title exceeds 200 characters
    """
    if any(data['Title len'] > 200):
        st.error("Some article titles exceed 200 characters.")
    else:
        st.success("All article titles are within 200 characters.")

def extract_sentences(text, sentence, num_before, num_after):
    """
    Extract sentences from text.
    :param text: Text to extract from.
    :param sentence: The sentence to match.
    :param num_before: Number of sentences before matched sentence.
    :param num_after: Number of sentences after matched sentence.
    :return: extracted sentences.
    """
    sentences = sent_tokenize(text)
    for i, sent in enumerate(sentences):
        if sentence in sent:
            start = max(0, i - num_before)
            end = min(len(sentences), i + num_after + 1)
            return sentences[start:end]
    return []


from typing import List, Union
import re


def get_sentences_around(text: str, match_phrase: str, num_before: int, num_after: int) -> List[str]:
    """
    Tokenize text, find match_phrase, and extract sentences around it.

    This function treats both single and multi-sentence matches
    and extracts num_before sentences before and num_after sentences after each match.
    If ranges to extract overlap, they get merged into a single range.

    Args:
    text (str): text to search in.
    match_phrase (str): phrase to look for.
    num_before (int): number of sentences before each match_phrase occurrence.
    num_after (int): number of sentences after each match_phrase occurrence.

    Returns:
    List[str]: list of text fragments containing match_phrase and surrounding sentences.
    """
    # Tokenize text.
    sentences = nltk.sent_tokenize(text)
    
    match_indices = []
    for i, sentence in enumerate(sentences):
        if re.search(match_phrase, sentence, flags=re.IGNORECASE):
            match_indices.append(i)
    
    match_ranges = [(max(0, i - num_before), min(len(sentences), i + num_after + 1)) for i in match_indices]
    
    # Merge overlapping ranges.
    merged_match_ranges = []
    for start, end in sorted(match_ranges):
        if merged_match_ranges and start <= merged_match_ranges[-1][1]:
            # If the current range overlaps with the last range in the list, merge them.
            merged_match_ranges[-1][1] = max(merged_match_ranges[-1][1], end)
        else:
            # Otherwise, add the current range as a new range.
            merged_match_ranges.append([start, end])

    # Get matched fragments.
    fragments = []
    for start, end in merged_match_ranges:
        fragment = ' '.join(sentences[start:end])
        fragments.append(fragment)

    return fragments


def main():
    """
    The main function to be called when the script is run
    """
    st.title('RAG-based Search \n(in Medium text set)')
    
    DATA_FILE = "data/medium.csv"

    # Load dataset
    data = load_data(DATA_FILE)

    # Display data validation and longest title
    st.markdown("**Data Validation and Longest Title Length**")
    validate_data(data)
    longest_title_length = max(data["Title len"])
    st.write(f"Longest title length: {longest_title_length}")

    # let the user choose number of sentences before and after
    st.markdown("**Number of Sentences Before and After**")
    num_sent_before = st.number_input('Number of sentences before', value=3, min_value=0)
    num_sent_after = st.number_input('Number of sentences after', value=4, min_value=0)

    # Search options
    st.markdown("**Search Options**")
    search_title = st.checkbox('Search in title')
    search_text = st.checkbox('Search in text')
    use_regex = st.checkbox('Use regular expressions')

    # Prepare search
    if search_text or search_title:
        text_to_search = st.text_input('Search:')
        search_button = st.button('Authenticate and Search')

        # Do search
        if search_button:
            results = []

            # convert pandas series to list, so we can iterate over rows
            titles = data['Title'].tolist() if search_title else []
            texts = data['Text'].tolist() if search_text else []

            for text in titles + texts:
                # Find match_phrase in the text
                match_fragments = get_sentences_around(text, text_to_search, num_sent_before, num_sent_after)

                for fragment in match_fragments:
                    if use_regex:
                        matches = re.findall(text_to_search, fragment, flags=re.IGNORECASE)
                    else:
                        matches = [match for match in re.findall(r'\b\w+\b', fragment) if match.lower() == text_to_search.lower()]
                    
                    # Make matched words bold using "**" in markdown
                    for match in matches:
                        fragment = fragment.replace(match, f"**{match}**")

                    results.append(fragment)

            if results:
                st.markdown("**Search Results**")
                for result in results:
                    st.markdown(result, unsafe_allow_html=True)
                    st.markdown("-"*10)
            else:
                st.write("No results found.")

    else:
        st.error("Please select at least one option to search in.")



if __name__ == "__main__":
    main()