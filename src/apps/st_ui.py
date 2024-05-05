import streamlit as st
import pandas as pd
from pathlib import Path
from typing import List, Union, Tuple
import re
import os

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

st.title('RAG-based Fragment Search \n(in Medium text set)') 

import nltk
nltk.download('punkt')

from nltk.tokenize import sent_tokenize


@st.cache_data(func=None, hash_funcs=None)
def load_data(file: Union[str, Path]) -> pd.DataFrame:
    """
    Load data from a specified CSV file
    """

    st.markdown('Loading data...') 
    data = pd.read_csv(file)
    data = data.applymap(str)  # apply str() to each cell
    st.markdown('Data loaded') 

    # return pd.read_csv(file).applymap(str)
    return data


@st.cache_data(func=None, hash_funcs=None)
def validate_titles(data: pd.DataFrame):
    """
    Check if any title exceeds 200 characters
    """

    title_lengths = data["Title"].apply(lambda x: len(x))

    longest_title_length = max(title_lengths)
    st.write(f"Longest title length: {longest_title_length}")

    if any(title_lengths > 200):
        st.error("Some article titles exceed 200 characters.")
    else:
        st.success("All article titles are within 200 characters.")

    return data


@st.cache_data(func=None, hash_funcs=None)
def validate_newlines(data: pd.DataFrame) -> pd.DataFrame:
    # Check the newlines structure it they come in pairs

    def check_newline(txt: str):
        if '\n' in txt:
            return txt.index('\n')
        else:
            return None
        
    def check_newlines(paras: List) -> List[Tuple[int, int]]:
        """Check for extra newlines after split to patagraphs
        
        Return the list of tuples (paragraph number, position of newline in paragraph)"""
        return [(i, check_newline(p)) for i, p in enumerate(paras) if check_newline(p)]

    # First, create a copy of the DataFrame
    data2 = data.copy()

    data2["ParaChecks"] = data2["Paragraphs"].apply(check_newlines)
    no_extra_newlines = (data2["ParaChecks"].value_counts().count() == 1) and (data2["ParaChecks"].value_counts().index[0] == [])
    if not no_extra_newlines:
        for [para_no, para_pos] in data2["ParaChecks"]:
            st.error(f"Paragraph {para_no}: Newline in position {para_pos}")
    data2.drop("ParaChecks", axis='columns', inplace=True)

    return data2


@st.cache_data(func=None, hash_funcs=None)
def split_paragraphs(data: pd.DataFrame, do_validate_newlines: bool = False) -> pd.DataFrame:
    # Check the newlines structure it they come in pairs

    # First, create a copy of the DataFrame
    data2 = data.copy()

    data2["Paragraphs"] = data2["Text"].apply(lambda x: [(i, t) for i, t in enumerate(x.split("\n\n"))])
    data2.drop("Text", axis='columns', inplace=True)

    if do_validate_newlines:
        data2 = validate_newlines(data2)

    return data2


@st.cache_data(func=None, hash_funcs=None)
def by_paragraphs(data: pd.DataFrame) -> pd.DataFrame:

    # First, create a copy of the DataFrame
    data2 = data.copy()

    noisy = True

    if noisy:
        my_bar = st.progress(0, text="Paragraph data preparation")
    if noisy:
        my_bar.progress(0, text="Exploding paragraphs...")
    data_by_para = data2.explode("Paragraphs").reset_index(names="paper_no")
    if noisy:
        my_bar.progress(25, text="Splitting into number and text...")
    data_by_para[["ParaNo","ParaText"]] = data_by_para.apply(lambda row: [row["Paragraphs"][0], row["Paragraphs"][1]], axis='columns', result_type='expand')
    if noisy:
        my_bar.progress(40, text="Dropping unnecessary tuples...")
    data_by_para.drop("Paragraphs", axis='columns', inplace=True)
    if noisy:
        my_bar.progress(80, text="Resetting index...")
    data_by_para.reset_index(drop=False, names="ParagraphIndex", inplace=True)  # Add index as column 'Index'
    if noisy:
        my_bar.progress(100, text="Done prepping paragrph data")

    return data_by_para


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


# @st.cache_data(func=None, hash_funcs=None)
def create_or_load_embeddings(data, model):
    # Define the path of your embeddings file
    embeddings_file_path = "embeddings.npy"

    # Initialization
    if 'embeddings' in st.session_state:
        embeddings = st.session_state['embeddings']
    else:
        # Check if the embeddings file exists
        if os.path.isfile(embeddings_file_path):
            # Load embeddings from disk
            st.write(f"Loading embeddings from disk...")
            embeddings = np.load(embeddings_file_path)
        else:
            # Generate embeddings and save them to disk
            st.write(f"Generating embeddings for all paragraphs...")
            embeddings = model.encode(
                data["ParaText"].tolist(), 
                convert_to_tensor=True, 
                show_progress_bar=True
            ).numpy()
            np.save(embeddings_file_path, embeddings)
        st.session_state['embeddings'] = embeddings

    return embeddings

@st.cache_resource(func=None, hash_funcs=None)
def prepare_rag(data: pd.DataFrame):
    # Model for generating embeddings
    # https://www.sbert.net/docs/pretrained_models.html
    # model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
    # model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
    # model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

    if 'model' not in st.session_state:
        st.session_state['model'] = SentenceTransformer('all-MiniLM-L6-v2')
    

    # Flatten the list of documents in your dataframe
    data_with_paragraphs = by_paragraphs(data)

    # paragraphs = data_with_paragraphs["ParaText"].tolist()


    # Generate embeddings for all paragraphs
    if 'embeddings' not in st.session_state:
        st.session_state['embeddings'] = create_or_load_embeddings(
            data_with_paragraphs, st.session_state['model']
        )
    # st.write(f"Generating embeddings for all paragraphs...")
    # embeddings = model.encode(
    #     data_with_paragraphs["ParaText"].tolist(), 
    #     convert_to_tensor=True, 
    #     show_progress_bar=True
    # )

    # Convert embeddings to numpy array for FAISS
    # xb = embeddings.numpy()
    xb = st.session_state['embeddings']  # .numpy()

    # Dimension of embeddings
    d = xb.shape[1]

    # Build the FAISS index  # https://github.com/facebookresearch/faiss/wiki/Getting-started
    st.write(f"Building the FAISS index...")
    index = faiss.IndexFlatL2(d)
    index.add(xb)

    return {
        "model": st.session_state['model'],
        "index": index,
        # "paragraphs": paragraphs,
        "embeddings": st.session_state['embeddings'],
        "data_with_paragraphs": data_with_paragraphs,
    }


def literal_search(data):
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
            search_button = st.button('Search')

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


def ask_question_and_search_old(embed_model, faiss_index, df, paragraphs, question):
    # Generate embedding for the question
    question_embedding = embed_model.encode([question])
    
    # Search the FAISS index
    D, I = faiss_index.search(np.ascontiguousarray(question_embedding), k=3)
    
    # Get the actual embedding and text
    closest_embeddings = [df['Embeddings'][i] for i in I[0]]
    closest_paragraphs = [paragraphs[i] for i in I[0]]

    for p in closest_paragraphs:
        st.text(p)

    return closest_embeddings, closest_paragraphs


# def ask_question_and_search(embed_model, faiss_index, data, question):
def ask_question_and_search(embed_model, faiss_index, para_data, question):
    # Generate embedding for the question
    question_embedding = embed_model.encode([question])
    
    # Search the FAISS index
    D, I = faiss_index.search(np.ascontiguousarray(question_embedding), k=3)

    # Get the actual paragraph index, title and text
    closest_paragraphs = para_data.loc[I[0]]

    for i, row in closest_paragraphs.iterrows():
        st.markdown(f":red[Title:] {row['Title']}")
        st.text(f"Proximity score: {D[0][I[0].tolist().index(row.name)]}")  # ParagraphIndex
        st.write(f"Paragraph: :blue[{row['ParaText']}]")

    if False:
    # Create a new DataFrame where each row is a paragraph, along with its index and title
        paragraph_data = pd.DataFrame({
            'ParagraphIndex': list(range(len(paragraphs))),
            'ParagraphText': paragraphs,
            'Title': data_with_paragraphs['Title'],  # assuming data_with_paragraphs has a 'Title' column
        })

        return {
            "index": index,
            "paragraph_data": paragraph_data,
            "embeddings": embeddings,
        }

def main():
    """
    The main function to be called when the script is run
    """
    # st.title('RAG-based Fragment Search \n(in Medium text set)') 
    
    DATA_FILE = "data/medium.csv"

    # Load dataset
    data = load_data(DATA_FILE)

    # Display data validation and longest title
    st.markdown("**Data Prep and Validation**")
    st.markdown("...in progress...")

    # validate_titles(data)
    data_paras_split = split_paragraphs(data, do_validate_newlines = True)

    # literal_search(data)

    embed_data_dict = prepare_rag(data=data_paras_split)
    # return {
    #     "model": model,
    #     "index": index,
    #     # "paragraphs": paragraphs,
    #     "embeddings": embeddings,
    #     "data_with_paragraphs": data_with_paragraphs,
    # }

    st.markdown("The end of data prep. Ready to answer question on the dataset...")

    # Unpack the embedding model, faiss index, and dataset from embed_data_dict 
    model = embed_data_dict['model']
    index = embed_data_dict['index']
    # paragraphs = embed_data_dict['paragraphs']
    embeddings = embed_data_dict['embeddings']
    data_with_paragraphs = embed_data_dict['data_with_paragraphs']

    # The loop where the user inputs their question
    # while True:
    question_input = st.text_input("Please enter your question:")
    if st.button("Search"):
        # Ask a question
        question = question_input

        # Generate embedding
        ask_question_and_search(
            embed_model=model, 
            faiss_index=index, 
            para_data=data_with_paragraphs, 
            question=question,
        )

        # # Display the results
        # for para in closest_paragraphs:
        #     st.text(para)



    print("the end")


if __name__ == "__main__":
    main()