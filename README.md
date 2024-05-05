# rag-search
### Semantic fragment search based on the RAG

## Granularity
The paragraph level was chosen for embedding creation, since we expect some ;level of topical consistency to be contained within paragraph of plain text usually.

## Creation of paragraph embeddings
Hugging Face Transformers were chosed for simplicity of operation and availablity of models (see 
[Pretrained Models](https://www.sbert.net/docs/pretrained_models.html)
)

Specifically, all-MiniLM-L6-v2 is fast and still offers good quality.

## Embedding database
In-memory implementation was deemed sufficient for the size of data (13K texts).

[FAISS](https://github.com/facebookresearch/faiss/wiki/Getting-started) was used for indexing of embeddings.
