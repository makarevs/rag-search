import streamlit as st
import pandas as pd
from pathlib import Path
from typing import List, Union, Tuple
import re
import os

import numpy as np
from scipy import spatial
import faiss
from sentence_transformers import SentenceTransformer

st.title('RAG-based Fragment Search \n(in Medium text set)') 

import nltk
nltk.download('punkt')

from nltk.tokenize import sent_tokenize

noisy = True
cum_percent = 0
inc_percent = 5
my_bar = None

if noisy:
    my_bar = st.progress(0, text="Paragraph data preparation")


def show_progress(message: str, percent: float = None):
    global cum_percent, inc_percent, my_bar
    if noisy:
        if percent:
            show_percent = percent
        else:
            cum_percent = min(100, cum_percent + inc_percent)
            show_percent = cum_percent
        my_bar.progress(show_percent, text=message)


# @st.cache_data(func=None, hash_funcs=None)
def load_data(file: Union[str, Path]) -> pd.DataFrame:
    """
    Load data from a specified CSV file
    """

    show_progress('Loading data...') 
    data = pd.read_csv(file)
    data = data.applymap(str)  # apply str() to each cell
    show_progress('Data loaded') 

    # return pd.read_csv(file).applymap(str)
    return data


@st.cache_data(func=None, hash_funcs=None)
def validate_titles(data: pd.DataFrame):
    """
    Check if any title exceeds 200 characters
    """

    title_lengths = data["Title"].apply(lambda x: len(x))

    longest_title_length = max(title_lengths)
    # st.write(f"Longest title length: {longest_title_length}")

    if any(title_lengths > 200):
        st.error("Some article titles exceed 200 characters.")
    # else:
    #     st.success("All article titles are within 200 characters.")

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


# @st.cache_data(func=None, hash_funcs=None)
def by_paragraphs(data: pd.DataFrame) -> pd.DataFrame:

    # First, create a copy of the DataFrame
    data2 = data.copy()

    show_progress("Exploding paragraphs...")
    data_by_para = data2.explode("Paragraphs").reset_index(names="paper_no")
    show_progress("Splitting into number and text...")
    data_by_para[["ParaNo","ParaText"]] = data_by_para.apply(lambda row: [row["Paragraphs"][0], row["Paragraphs"][1]], axis='columns', result_type='expand')
    show_progress("Dropping unnecessary tuples...")
    data_by_para.drop("Paragraphs", axis='columns', inplace=True)
    show_progress("Resetting index...")
    data_by_para.reset_index(drop=False, names="ParagraphIndex", inplace=True)  # Add index as column 'Index'
    show_progress("Done prepping paragrph data")

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


# @st.cache_resource(func=None, hash_funcs=None)  # had to comment after added progress
def prepare_rag(data: pd.DataFrame):
    # Model for generating embeddings
    # https://www.sbert.net/docs/pretrained_models.html
    model_name = 'all-MiniLM-L6-v2'
    # model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
    # model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
    # model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

    if 'model' not in st.session_state:
        show_progress(f"Loading model '{model_name}'...")
        st.session_state['model'] = SentenceTransformer(model_name)
        show_progress(f"Finished loading the model '{model_name}'")

    # Flatten the list of documents in your dataframe
    if 'data_with_paragraphs' not in st.session_state:
        st.session_state['data_with_paragraphs'] = by_paragraphs(data)

    # Generate embeddings for all paragraphs
    if 'embeddings' not in st.session_state:
        st.session_state['embeddings'] = create_or_load_embeddings(
            st.session_state['data_with_paragraphs'], 
            st.session_state['model'],
        )

    # Convert embeddings to numpy array for FAISS
    # xb = embeddings.numpy()
    xb = st.session_state['embeddings']  # .numpy()

    # Dimension of embeddings
    d = xb.shape[1]

    # Build the FAISS index  # https://github.com/facebookresearch/faiss/wiki/Getting-started
    if 'faiss_index' not in st.session_state:
        show_progress(f"Building the FAISS index...")
        st.session_state['faiss_index'] = faiss.IndexFlatL2(d)
        st.session_state['faiss_index'].add(xb)
    show_progress("Finished indexing")

    return {
        "model": st.session_state['model'],
        "index": st.session_state['faiss_index'],
        # "paragraphs": paragraphs,
        "embeddings": st.session_state['embeddings'],
        "data_with_paragraphs": st.session_state['data_with_paragraphs'],
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


def expand_around_relevant_paragraphs(question_embedding, best_abs_index, para_data, embeddings, pre_para_num, post_para_num):
    """Decide which surrounding paragraphs to inludein answer
    
    Draft implementation:
        pre_para_num, post_para_num - fixed number to include before and after, checked for article boundaries
    Refined inplementation:
        pre_para_num, post_para_num - max number to include before and after, checked for article boundaries, and truncated by goal function
    """

    # Get pararaph data for next best index
    best_data = para_data.loc[best_abs_index]
    # Get data for the paper where best index was found
    paper_data = para_data[para_data['paper_no'] == best_data['paper_no']]
    # Get index bounds for the paper
    lowest_abs_index = paper_data['ParagraphIndex'].index[0]  # or paper_data['ParagraphIndex'].tolist()[0]
    highest_abs_index = paper_data['ParagraphIndex'].index[-1]  # or paper_data['ParagraphIndex'].tolist()[-1]
    selected_abs_indices = range(
        max(lowest_abs_index, best_abs_index-pre_para_num), 
        min(highest_abs_index, best_abs_index+post_para_num)+1
    )
    expanded_para_data = para_data.loc[selected_abs_indices]
    expanded_embeddings = embeddings[selected_abs_indices]

    # expanded_indices = [
    #     i 
    #     for index in paper_data['ParaNo']
    #     for i in range(max(0, index-pre_para_num), min(len(data['paragraphs'])-1, index+post_para_num))
    # ]
    # expanded_paragraphs = [data['paragraphs'][i] for i in expanded_indices]
    # expanded_embeddings = [data['embedding'][i] for i in expanded_indices]

    # reference_embedding = np.array([...])
    # embeddings = np.array([[...], [...], ...])  # array of multiple embeddings
    question_embedding_1d = question_embedding.squeeze() # or question_embedding.reshape(-1)
    # print(f"question_embedding.shape={question_embedding_1d.shape}")  # If using plain Python
    # for e in expanded_embeddings:
    #     print(f"e.shape={e.shape}")

    expanded_distances = [spatial.distance.cosine(question_embedding_1d, e) for e in expanded_embeddings]

    return expanded_distances, expanded_para_data


# def ask_question_and_search(embed_model, faiss_index, data, question):
def ask_question_and_search(embed_model, faiss_index, para_data, embeddings, question):
    # Generate embedding for the question
    question_embedding = embed_model.encode([question])
    
    # Search the FAISS index
    D, I = faiss_index.search(np.ascontiguousarray(question_embedding), k=3)

    just_best = False

    if just_best:
        # Get the actual paragraph index, title and text
        closest_paragraphs = para_data.loc[I[0]]

        for i, row in closest_paragraphs.iterrows():
            st.markdown(f"### :red[Title:] {row['Title']}")
            st.text(f"Proximity score: {D[0][I[0].tolist().index(row.name)]}")  # ParagraphIndex
            st.write(f"Paragraph:\n :blue[{row['ParaText']}]")
    else:
        # Get the surrounding paragraphs with metadata about distances and relative indices in paper
        for best_abs_index in I[0].tolist():
            st.markdown(f"-----------------------------")
            st.markdown(f"### :red[Title:] {para_data.loc[best_abs_index]['Title']}")
            expanded_distances, expanded_para_data = expand_around_relevant_paragraphs(
                question_embedding=question_embedding, 
                best_abs_index=best_abs_index,
                para_data=para_data, 
                embeddings=embeddings, 
                pre_para_num=1, 
                post_para_num=2
            )
            # for para_distance, para_data in zip(expanded_distances, expanded_para_data):
            for i, para_data_row in enumerate(expanded_para_data.itertuples()):
                para_distance = expanded_distances[i]
                para_data_row = pd.Series(data=para_data_row[1:], index=expanded_para_data.columns)
                st.write(f":green[{round(para_distance,3)}] | :grey[{para_data_row['ParaNo']}] | :blue[{para_data_row['ParaText']}]")


def main():
    """
    The main function to be called when the script is run
    """
    # st.title('RAG-based Fragment Search \n(in Medium text set)') 
    
    DATA_FILE = "data/medium.csv"

    # Load dataset
    if 'papers_data' not in st.session_state:
        st.session_state['papers_data'] = load_data(DATA_FILE)

    # Display data validation and longest title
    show_progress("**Data Prep and Validation** in progress...")

    # validate_titles(data)
    data_paras_split = split_paragraphs(st.session_state['papers_data'], do_validate_newlines = True)

    # literal_search(data)

    embed_data_dict = prepare_rag(data=data_paras_split)
    # return {
    #     "model": model,
    #     "index": index,
    #     # "paragraphs": paragraphs,
    #     "embeddings": embeddings,
    #     "data_with_paragraphs": data_with_paragraphs,
    # }

    show_progress("The end of data prep. Ready to answer question on the dataset...")

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
            embeddings=embeddings,
            question=question,
        )

        # # Display the results
        # for para in closest_paragraphs:
        #     st.text(para)



    print("the end")


if __name__ == "__main__":
    main()