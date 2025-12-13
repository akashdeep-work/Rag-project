from __future__ import annotations
from typing import List
from textwrap import dedent

from models.schemas import SearchResult
from AiModel.models import LLM


class SearchSummarizer:
    """Summarize retrieval results using a local Mistral LLM."""

    def __init__(self, llm: LLM, max_context_chars: int = 12000) -> None:
        self.llm = llm
        self.max_context_chars = max_context_chars

    def _build_prompt(self, query: str, results: List[SearchResult]) -> str:
        context_sections = []
        accumulated = 0

        for idx, result in enumerate(results, start=1):
            text = (result.metadata.text or "").strip()
            if not text:
                continue

            snippet = text[:1000]
            accumulated += len(snippet)
            context_sections.append(
                f"{idx}. Source: {result.metadata.file_path}\n" f"Snippet: {snippet}\n"
            )

            if accumulated >= self.max_context_chars:
                break

        context_block = "\n".join(context_sections)
        if not context_block or context_sections.count == 0:
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

            Craft a short answer (3-6 sentences) that directly addresses the question.
            Add bullet points for key takeaways if useful. Avoid fabricating details not present in the context.
            """
        ).strip()

    def summarize(self, query: str, results: List[SearchResult]) -> str:
        prompt = self._build_prompt(query, results)
        print(f'prompt is {prompt}')
        response = self.llm.generate(prompt)
        return response.get("text", "") if isinstance(response, dict) else str(response)
