from __future__ import annotations

from textwrap import dedent
from typing import Iterable, List

from AiModel.models import LLM
from models.schemas import SearchResult


class SearchSummarizer:
    """Summarize retrieval results and support token streaming responses."""

    def __init__(self, llm: LLM, max_context_chars: int = 12000) -> None:
        self.llm = llm
        self.max_context_chars = max_context_chars

    def _build_prompt(self, query: str, results: List[SearchResult]) -> str:
        context_sections: list[str] = []
        accumulated = 0

        for idx, result in enumerate(results, start=1):
            text = (result.metadata.text or "").strip()
            if not text:
                continue

            snippet = text[:1000]
            accumulated += len(snippet)
            location = f"chunk {result.metadata.chunk_index}"
            if result.metadata.start_time is not None and result.metadata.end_time is not None:
                location = f"timestamp {result.metadata.start_time:.2f}-{result.metadata.end_time:.2f}s"

            score_display = result.score if result.score is not None else 0.0
            context_sections.append(
                "\n".join(
                    [
                        f"Source {idx}: {result.metadata.file_path} ({location})",
                        f"Relevance score: {score_display:.4f}",
                        f"Excerpt: {snippet}",
                    ]
                )
            )

            if accumulated >= self.max_context_chars:
                break

        context_block = "\n\n".join(context_sections)
        if not context_block:
            return dedent(
                f"""
                The user asked: {query}
                No supporting documents were found. Respond courteously that no answer is available.
                """
            ).strip()

        return dedent(
            f"""
            You are a helpful assistant that writes concise, human-friendly summaries.
            Use the supplied context snippets to answer the question.

            Question: {query}
            Context:
            {context_block}

            Craft a short answer (6-10 sentences maximum) that directly addresses the question.
            Add bullet points for key takeaways if useful. Avoid fabricating details not present in the context.
            """
        ).strip()

    def summarize(self, query: str, results: List[SearchResult]) -> str:
        prompt = self._build_prompt(query, results)
        response = self.llm.generate(prompt)
        return response.get("text", "") if isinstance(response, dict) else str(response)

    def summarize_stream(self, query: str, results: List[SearchResult]) -> Iterable[str]:
        prompt = self._build_prompt(query, results)
        return self.llm.generate_stream(prompt)

    def converse(self, message: str) -> str:
        prompt = dedent(
            f"""
            You are a friendly, concise assistant. Respond helpfully to the user input.

            User: {message}
            Assistant:
            """
        ).strip()
        response = self.llm.generate(prompt)
        return response.get("text", "") if isinstance(response, dict) else str(response)

    def converse_stream(self, message: str) -> Iterable[str]:
        prompt = dedent(
            f"""
            You are a friendly, concise assistant. Respond helpfully to the user input.

            User: {message}
            Assistant:
            """
        ).strip()
        return self.llm.generate_stream(prompt)
