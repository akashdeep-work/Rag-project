from __future__ import annotations
from fastapi import Request

from indexer import Indexer
from services.summarizer import SearchSummarizer


def get_indexer(request: Request) -> Indexer:
    return request.app.state.indexer


def get_summarizer(request: Request) -> SearchSummarizer:
    return request.app.state.summarizer
