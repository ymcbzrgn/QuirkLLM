"""
Hybrid Search Pipeline for QuirkLLM RAG System (Phase 3.6)

Implements semantic + keyword search with ranking and context enrichment.
"""

from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
import re

from quirkllm.rag.lancedb_store import LanceDBStore, SearchResult
from quirkllm.rag.embeddings import EmbeddingGenerator
from quirkllm.core.profile_manager import ProfileConfig


@dataclass
class ContextualResult:
    """Search result with enriched context"""
    content: str
    file_path: str
    language: str
    start_line: int
    end_line: int
    score: float
    parent_context: Optional[str] = None  # Parent function/class
    imports: Optional[List[str]] = None  # File imports
    metadata: Optional[Dict] = None


class QueryProcessor:
    """Process and decompose user queries"""
    
    @staticmethod
    def extract_keywords(query: str) -> List[str]:
        """
        Extract code-relevant keywords from query.
        
        Args:
            query: User query
            
        Returns:
            List of keywords (function names, class names, etc.)
        """
        # Remove common words
        stop_words = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'can', 'may', 'might', 'must', 'to', 'from',
            'in', 'on', 'at', 'for', 'with', 'by', 'about', 'as', 'into',
            'of', 'that', 'this', 'it', 'and', 'or', 'but', 'not'
        }
        
        # Tokenize (split on spaces and special chars, keep alphanumeric)
        tokens = re.findall(r'\b\w+\b', query.lower())
        
        # Filter stop words and very short tokens
        keywords = [t for t in tokens if t not in stop_words and len(t) > 2]
        
        return keywords
    
    @staticmethod
    def decompose_query(query: str) -> List[str]:
        """
        Decompose complex queries into sub-queries.
        
        Args:
            query: Complex query (e.g., "function to parse JSON and validate schema")
            
        Returns:
            List of sub-queries
        """
        # Split on 'and', 'or', 'also', 'then'
        connectors = r'\b(and|or|also|then|plus)\b'
        sub_queries = re.split(connectors, query, flags=re.IGNORECASE)
        
        # Filter empty and connector tokens
        sub_queries = [
            q.strip() for q in sub_queries 
            if q.strip() and not re.match(connectors, q.strip(), re.IGNORECASE)
        ]
        
        # If no split, return original query
        return sub_queries if len(sub_queries) > 1 else [query]


class RankFusion:
    """Rank fusion algorithms for combining search results"""
    
    @staticmethod
    def reciprocal_rank_fusion(
        results_lists: List[List[SearchResult]],
        k: int = 60
    ) -> List[Tuple[SearchResult, float]]:
        """
        Combine multiple ranked lists using Reciprocal Rank Fusion.
        
        Args:
            results_lists: Multiple ranked lists of search results
            k: Constant for RRF formula (default 60)
            
        Returns:
            Unified ranked list with RRF scores
        """
        # Compute RRF score for each unique result
        rrf_scores: Dict[str, Tuple[SearchResult, float]] = {}
        
        for results in results_lists:
            for rank, result in enumerate(results, start=1):
                # Use file_path + content as unique key
                key = f"{result.file_path}:{result.start_line}"
                
                # RRF formula: 1 / (k + rank)
                score = 1.0 / (k + rank)
                
                if key in rrf_scores:
                    # Accumulate scores from different lists
                    existing_result, existing_score = rrf_scores[key]
                    rrf_scores[key] = (existing_result, existing_score + score)
                else:
                    rrf_scores[key] = (result, score)
        
        # Sort by RRF score descending
        ranked = sorted(rrf_scores.values(), key=lambda x: x[1], reverse=True)
        
        return ranked


class CodeRetriever:
    """Hybrid search retriever with semantic + keyword search"""
    
    def __init__(
        self,
        db: LanceDBStore,
        embedder: EmbeddingGenerator,
        profile: ProfileConfig
    ):
        """
        Initialize retriever.
        
        Args:
            db: LanceDB vector store
            embedder: Embedding generator
            profile: RAM profile config
        """
        self.db = db
        self.embedder = embedder
        self.profile = profile
        self.query_processor = QueryProcessor()
        self.rank_fusion = RankFusion()
    
    def _semantic_search(
        self,
        query: str,
        k: int = 20,
        filters: Optional[Dict] = None
    ) -> List[SearchResult]:
        """Semantic search using embeddings."""
        query_embedding = self.embedder.embed_query(query)
        results = self.db.search(query_embedding, k=k, filters=filters)
        return results
    
    def _keyword_search(
        self,
        query: str,
        k: int = 10,
        filters: Optional[Dict] = None
    ) -> List[SearchResult]:
        """
        Keyword-based search using BM25-like scoring.
        
        Note: Simplified implementation. Uses keyword matching + TF-IDF like scoring.
        """
        keywords = self.query_processor.extract_keywords(query)
        
        if not keywords:
            return []
        
        # Get all chunks (filtered)
        # Note: In production, this should use an inverted index
        # For now, we'll do a semantic search and rerank by keyword matching
        all_results = self.db.search(
            self.embedder.embed_query(query),
            k=k * 3,  # Get more candidates
            filters=filters
        )
        
        # Score by keyword matches
        scored_results = []
        for result in all_results:
            content_lower = result.content.lower()
            
            # Count keyword occurrences
            matches = sum(
                content_lower.count(keyword)
                for keyword in keywords
            )
            
            if matches > 0:
                # Simple TF-IDF like score
                score = matches / len(keywords)
                scored_results.append((result, score))
        
        # Sort by keyword score
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        return [result for result, _ in scored_results[:k]]
    
    def retrieve(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict] = None,
        use_hybrid: bool = True
    ) -> List[SearchResult]:
        """
        Retrieve relevant code snippets.
        
        Args:
            query: User query (natural language or code)
            k: Number of results to return
            filters: Metadata filters (language, framework, file_path)
            use_hybrid: Use hybrid search (semantic + keyword)
            
        Returns:
            List of SearchResult sorted by relevance
        """
        if not query or not query.strip():
            return []
        
        if use_hybrid:
            # Hybrid search: semantic + keyword
            semantic_results = self._semantic_search(query, k=20, filters=filters)
            keyword_results = self._keyword_search(query, k=10, filters=filters)
            
            # Combine using Reciprocal Rank Fusion
            fused_results = self.rank_fusion.reciprocal_rank_fusion(
                [semantic_results, keyword_results]
            )
            
            # Update scores and return top k
            final_results = []
            for result, rrf_score in fused_results[:k]:
                # Create new result with RRF score
                result.score = rrf_score
                final_results.append(result)
            
            return final_results
        else:
            # Pure semantic search
            return self._semantic_search(query, k=k, filters=filters)
    
    def _find_parent_context(self, result: SearchResult, file_chunks: List) -> Optional[str]:
        """Find parent context for a result."""
        if result.chunk_index == 0:
            return None
        
        parent_chunks = [
            c for c in file_chunks
            if c.chunk_index == result.chunk_index - 1
        ]
        
        return parent_chunks[0].content if parent_chunks else None
    
    def _extract_imports(self, content: str) -> Optional[List[str]]:
        """Extract import statements from code."""
        import_lines = [
            line.strip()
            for line in content.split('\n')
            if line.strip().startswith(('import ', 'from '))
        ]
        
        return import_lines if import_lines else None
    
    def _enrich_result(self, result: SearchResult) -> ContextualResult:
        """Enrich a search result with context."""
        # Get all chunks from same file
        file_chunks = self.db.get_by_file(result.file_path)
        
        # Find parent context and extract imports
        parent_content = self._find_parent_context(result, file_chunks)
        imports = self._extract_imports(result.content)
        
        return ContextualResult(
            content=result.content,
            file_path=result.file_path,
            language=result.language,
            start_line=result.start_line,
            end_line=result.end_line,
            score=result.score,
            parent_context=parent_content,
            imports=imports,
            metadata=result.metadata
        )
    
    def retrieve_with_context(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict] = None
    ) -> List[ContextualResult]:
        """
        Retrieve code with enriched context (parent chunks, imports).
        
        Args:
            query: User query
            k: Number of results to return
            filters: Metadata filters
            
        Returns:
            List of ContextualResult with enriched context
        """
        # Get base results
        base_results = self.retrieve(query, k=k, filters=filters)
        
        # Enrich each result with context
        return [self._enrich_result(result) for result in base_results]
    
    def multi_query_retrieve(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict] = None
    ) -> List[SearchResult]:
        """
        Retrieve using query decomposition for complex queries.
        
        Args:
            query: Complex query
            k: Number of results per sub-query
            filters: Metadata filters
            
        Returns:
            Unified list of results from all sub-queries
        """
        # Decompose query
        sub_queries = self.query_processor.decompose_query(query)
        
        if len(sub_queries) == 1:
            # Simple query, use normal retrieve
            return self.retrieve(query, k=k, filters=filters)
        
        # Retrieve for each sub-query
        all_results = []
        for sub_query in sub_queries:
            results = self.retrieve(sub_query, k=k, filters=filters)
            all_results.append(results)
        
        # Combine using RRF
        fused_results = self.rank_fusion.reciprocal_rank_fusion(all_results)
        
        # Return top k unique results
        final_results = []
        seen = set()
        for result, rrf_score in fused_results:
            key = f"{result.file_path}:{result.start_line}"
            if key not in seen:
                result.score = rrf_score
                final_results.append(result)
                seen.add(key)
                
                if len(final_results) >= k:
                    break
        
        return final_results
