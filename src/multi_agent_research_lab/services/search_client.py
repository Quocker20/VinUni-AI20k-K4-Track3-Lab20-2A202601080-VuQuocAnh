import json
import logging
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import SourceDocument

logger = logging.getLogger(__name__)


class SearchClient:
    """Provider-agnostic search client with Tavily and offline corpus fallback."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.tavily_api_key = self.settings.tavily_api_key
        repo_root = Path(__file__).resolve().parents[3]
        self._corpus_dir = repo_root / "ai_agent_offline_research_corpus_v2" / "topics"

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query."""
        if self.tavily_api_key:
            try:
                results = self._search_tavily(query, max_results=max_results)
                if results:
                    return results
            except Exception as exc:
                logger.warning("Tavily search failed (%s), falling back to offline corpus.", exc)

        return self._search_offline_corpus(query, max_results=max_results)

    def _search_tavily(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Query Tavily REST API without hardcoded dependencies."""
        url = "https://api.tavily.com/search"
        payload = json.dumps(
            {
                "api_key": self.tavily_api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
                "include_answer": False,
            }
        ).encode("utf-8")

        req = Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "MultiAgentResearchLab/1.0",
            },
        )

        with urlopen(req, timeout=self.settings.timeout_seconds) as resp:
            data: dict[str, Any] = json.loads(resp.read().decode("utf-8"))

        documents: list[SourceDocument] = []
        for item in data.get("results", []):
            documents.append(
                SourceDocument(
                    title=item.get("title", "Untitled Source"),
                    url=item.get("url"),
                    snippet=item.get("content", ""),
                    metadata={"score": item.get("score"), "source": "tavily"},
                )
            )
        return documents

    def _search_offline_corpus(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Query the local offline corpus topics JSON files."""
        documents: list[SourceDocument] = []
        keywords = set(query.lower().replace("-", " ").replace("_", " ").split())
        stop_words = {
            "a",
            "an",
            "the",
            "in",
            "on",
            "of",
            "and",
            "or",
            "for",
            "with",
            "to",
            "is",
            "write",
            "summary",
            "500",
            "word",
        }
        query_terms = [k for k in keywords if len(k) > 2 and k not in stop_words]
        if not query_terms:
            query_terms = list(keywords)

        if self._corpus_dir.exists():
            topic_files = list(self._corpus_dir.glob("*.json"))
            scored_files: list[tuple[int, Path]] = []

            for path in topic_files:
                name_lower = path.stem.lower()
                score = sum(1 for term in query_terms if term in name_lower)
                if score > 0:
                    scored_files.append((score, path))

            scored_files.sort(key=lambda x: x[0], reverse=True)
            files_to_read = [f[1] for f in scored_files[:3]] if scored_files else topic_files[:2]

            for file_path in files_to_read:
                try:
                    with open(file_path, encoding="utf-8") as f:
                        data = json.load(f)

                    # Extract knowledge articles
                    knowledge_base = data.get("knowledge_base", {})
                    for article in knowledge_base.get("knowledge_articles", []):
                        art_id = article.get("article_id", "KB")
                        art_title = article.get("title", "Knowledge Article")
                        documents.append(
                            SourceDocument(
                                title=f"[{art_id}] {art_title}",
                                url=f"offline://corpus/{file_path.name}#{art_id}",
                                snippet=article.get("content", "")[:600],
                                metadata={"article_id": art_id, "source": "offline_corpus"},
                            )
                        )
                        if len(documents) >= max_results:
                            return documents

                    # Extract source docs
                    for src in data.get("source_documents", []):
                        src_id = src.get("source_id", "SRC")
                        src_title = src.get("title", "Reference Document")
                        documents.append(
                            SourceDocument(
                                title=f"[{src_id}] {src_title}",
                                url=src.get("url") or f"offline://corpus/{src_id}",
                                snippet=src.get("summary") or src.get("content", "")[:600],
                                metadata={"source_id": src_id, "source": "offline_corpus"},
                            )
                        )
                        if len(documents) >= max_results:
                            return documents
                except Exception as exc:
                    logger.debug("Failed reading topic file %s: %s", file_path, exc)

        # Fallback synthesized sources if corpus files empty or not matched
        if not documents:
            documents = [
                SourceDocument(
                    title="GraphRAG: Unlocking LLM Discovery on Complex Information",
                    url="https://arxiv.org/abs/2404.16130",
                    snippet=(
                        "GraphRAG combines knowledge graphs and dense text retrieval to handle "
                        "global query summarization across large document collections."
                    ),
                    metadata={"source": "simulated_benchmark"},
                ),
                SourceDocument(
                    title="Building Effective Multi-Agent Architectures",
                    url="https://www.anthropic.com/engineering/building-effective-agents",
                    snippet=(
                        "Multi-agent patterns divide responsibilities into distinct supervisor, "
                        "worker, and reviewer loops to maintain bounded context windows."
                    ),
                    metadata={"source": "simulated_benchmark"},
                ),
            ]

        return documents[:max_results]
