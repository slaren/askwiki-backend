import logging
import os
from typing import List

import openai
from bs4 import BeautifulSoup
from sentence_transformers.util import semantic_search

import model
from util import perf_logger
from wiki import get_wiki_article_text

logger = logging.getLogger(__name__)

# TODO: move to configuration file
openai_api_enabled = False
max_prompt_length = 1000
max_answer_length = 500
prompt_template = (
    "Answer the following questions truthfully, using only the information in the article\n"
    "===\n"
    "Article: Cats are mammals. They have fur and whiskers. They are very cute.\n"
    "===\n"
    "Q: What are cats?\n"
    "A: Cats are mammals. They have fur and whiskers and they are also very cute.\n"
    "---\n"
    "Q: Do cats have fur?\n"
    "A: Yes, cats have fur and also whiskers.\n"
    "---\n"
    "Q: What is a dog?\n"
    "A: The article does not contain information regarding this question.\n"
    "===\n"
    "Article: {}\n"
    "===\n"
    "Q: {}\n"
    "A: "
)


def get_article_chunks(text: str) -> List[str]:
    soup = BeautifulSoup(text, "html.parser")

    # Find all math tags and replace with their alttext
    for math_tag in soup.find_all("math"):
        alttext = math_tag.get("alttext")

        # Check if the math tag is inside a p tag
        in_paragraph = False
        parent = math_tag.parent
        while parent:
            if parent.name == "p":
                in_paragraph = True
                break
            parent = parent.parent

        if in_paragraph:
            # If the math tag is inside a p tag, just replace it with the alttext
            math_tag.replace_with(alttext)
        else:
            # If the math tag isn't inside a p tag, create a new paragraph and insert the alttext as its text
            new_paragraph = soup.new_tag("p")
            new_paragraph.string = alttext
            math_tag.replace_with(new_paragraph)

    # Find all paragraphs and headings
    paragraphs = soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6"])
    chunks = [p.get_text().strip() for p in paragraphs]

    # These sections are usually not relevant
    ignored_sections = ["See also", "References", "External links"]
    chunks = [c for c in chunks if c not in ignored_sections]
    return chunks


class ContextBuilder:
    """Builds a context for a question by adding chunks from an article."""

    def __init__(
        self, chunks: List[str], chunk_lengths: List[int], max_context_length: int
    ):
        self._chunks = chunks
        self._chunk_lengths = chunk_lengths
        self._max_context_length = max_context_length
        self._cur_context_length = 0
        self._added_chunks = []

    def add_chunk(self, chunk_id: int) -> bool:
        if chunk_id in self._added_chunks:
            return True

        if chunk_id < 0 or chunk_id >= len(self._chunks):
            return True

        chunk_length = self._chunk_lengths[chunk_id]
        if self._cur_context_length + chunk_length > self._max_context_length:
            return False

        self._cur_context_length += chunk_length
        self._added_chunks.append(chunk_id)
        return True

    def add_adj_chunks(self, chunk_id: int, num_adj_chunks: int) -> bool:
        if not self.add_chunk(chunk_id):
            return False
        for i in range(1, num_adj_chunks + 1):
            if not self.add_chunk(chunk_id - i):
                return False
            if not self.add_chunk(chunk_id + i):
                return False
        return True

    def build_context(self) -> str:
        self._added_chunks.sort()
        logger.debug("Selected chunks: %s", self._added_chunks)
        context = "\n".join([self._chunks[i] for i in self._added_chunks])
        return context


def get_answer(article, query):
    # Compute query embedding
    with perf_logger("Computing query embedding", logger):
        query_embedding = model.encode([query])

    # Retrieve article text
    text = get_wiki_article_text(article)

    # Split into paragraphs
    chunks = get_article_chunks(text)

    # Compute chunk embeddings
    with perf_logger("Computing chunk embeddings", logger):
        chunk_embeddings = model.encode(chunks, batch_size=64)

    # Find most relevant chunks
    hits = semantic_search(query_embedding, chunk_embeddings, top_k=5)[0]

    # Compute the length of the prompt without the context to be able to compute the maximum context length
    chunk_lengths = model.tokenize(chunks, return_length=True).length
    prompt_query_length = model.tokenize(
        prompt_template.format("", query), return_length=True
    ).length[0]

    # Build the context concatenate relevant chunks and adjacent chunks, within the token limit
    cbuilder = ContextBuilder(
        chunks, chunk_lengths, max_prompt_length - prompt_query_length
    )
    for hit in hits:
        if not cbuilder.add_adj_chunks(hit["corpus_id"], 4):
            break
    context = cbuilder.build_context()

    # Build prompt
    prompt = prompt_template.format(context, query)
    prompt_length = model.tokenize(prompt, return_length=True).length[0]

    logger.debug("Prompt: %s", prompt)
    logger.info("Computed prompt length: %d tokens", prompt_length)

    if openai_api_enabled:
        # Query the language model
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        openai.api_key = OPENAI_API_KEY
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0,
            max_tokens=max_answer_length,
            stop=["\n---\n"],
            stream=stream,
        )

        # Iterate through the stream of events
        answer = ""
        for event in response:
            prompt_tokens = event["usage"]["prompt_tokens"]
            completion_tokens = event["usage"]["completion_tokens"]
            total_tokens = event["usage"]["total_tokens"]
            logger.info(
                "OpenAI completion usage: %d prompt tokens, %d completion tokens, %d total tokens",
                prompt_tokens,
                completion_tokens,
                total_tokens,
            )

            event_text = event["choices"][0]["text"]  # extract the text
            answer += event_text  # append the text
            yield event_text
    else:
        ## debug
        import time

        time.sleep(2)
        yield "API "
        time.sleep(1)
        yield "access "
        time.sleep(1)
        yield "disabled"
        time.sleep(1)
        yield f" ({article},\n{query})\n"
