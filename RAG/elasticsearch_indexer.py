"""
ElasticSearch indexer for FDA drug label data

Search modes:
- token: Exact token/word matching (no fuzzy)
- prefix: Prefix matching (e.g., "asp" matches "aspirin")
- fuzzy: Fuzzy matching with AUTO fuzziness
"""

import json
import os
import re
import unicodedata
from typing import List, Dict, Any, Optional
from enum import Enum
from elasticsearch import Elasticsearch, helpers


class SearchMode(Enum):
    TOKEN = "token"  # Exact token matching (no fuzzy)
    PREFIX = "prefix"  # Prefix matching
    FUZZY = "fuzzy"  # Fuzzy matching with AUTO


class ElasticsearchDrugIndexer:
    def __init__(self, host="localhost", port=9200, index_name="fda_drugs", analyzer_name: str = "drug_name_analyzer"):
        """Initialize Elasticsearch indexer"""
        self.host = host
        self.port = port
        self.index_name = index_name
        self.es = None
        self.server_version = "unknown"
        self.analyzer_name = analyzer_name

    def connect(self):
        """Connect to Elasticsearch"""
        try:
            self.es = Elasticsearch([{"host": self.host, "port": self.port, "scheme": "http"}])
            if not self.es.ping():
                raise ConnectionError("Could not connect to Elasticsearch")
            # Best-effort server info for reproducibility
            try:
                info = self.es.info()
                self.server_version = str(info.get("version", {}).get("number", "unknown"))
            except Exception:
                self.server_version = "unknown"
        except Exception as e:
            raise ConnectionError(f"Error connecting to Elasticsearch: {e}")

    def index_exists(self) -> bool:
        """Check if the index exists"""
        if not self.es:
            return False
        return self.es.indices.exists(index=self.index_name)

    def get_doc_count(self) -> int:
        """Get total number of documents in the index"""
        if not self.es or not self.index_exists():
            return 0
        result = self.es.count(index=self.index_name)
        return result["count"]

    def validate_index(self) -> tuple:
        """
        Validate that the index exists and has documents

        Returns:
            tuple: (is_valid: bool, message: str, doc_count: int)
        """
        if not self.es:
            return (False, "Not connected to Elasticsearch", 0)

        if not self.index_exists():
            return (False, f"Index '{self.index_name}' does not exist", 0)

        doc_count = self.get_doc_count()
        if doc_count == 0:
            return (False, f"Index '{self.index_name}' is empty", 0)

        return (True, f"Index '{self.index_name}' is valid with {doc_count} documents", doc_count)

    def create_index(self):
        """Create Elasticsearch index with optimized mapping"""
        self.connect()

        analyzer_name = self.analyzer_name
        normalizer_name = "lowercase_ascii"

        # Delete existing index if it exists
        if self.es.indices.exists(index=self.index_name):
            self.es.indices.delete(index=self.index_name)
            print(f"Deleted existing index: {self.index_name}")

        # Create index with custom mapping
        mapping = {"settings": {"number_of_shards": 1, "number_of_replicas": 0, "analysis": {"normalizer": {normalizer_name: {"type": "custom", "filter": ["lowercase", "asciifolding"]}}, "analyzer": {analyzer_name: {"type": "custom", "tokenizer": "standard", "filter": ["lowercase", "asciifolding"]}}}}, "mappings": {"properties": {"doc_id": {"type": "keyword"}, "brand_names": {"type": "text", "analyzer": analyzer_name, "fields": {"keyword": {"type": "keyword", "normalizer": normalizer_name}}}, "generic_names": {"type": "text", "analyzer": analyzer_name, "fields": {"keyword": {"type": "keyword", "normalizer": normalizer_name}}}, "substance_names": {"type": "text", "analyzer": analyzer_name, "fields": {"keyword": {"type": "keyword", "normalizer": normalizer_name}}}, "manufacturer_names": {"type": "text"}, "product_type": {"type": "keyword"}, "route": {"type": "text"}, "purpose": {"type": "text"}, "indications_and_usage": {"type": "text"}, "warnings": {"type": "text"}, "adverse_reactions": {"type": "text"}, "drug_interactions": {"type": "text"}, "dosage_and_administration": {"type": "text"}, "active_ingredient": {"type": "text", "analyzer": analyzer_name}, "inactive_ingredient": {"type": "text"}, "full_data": {"type": "object", "enabled": False}}}}

        self.es.indices.create(index=self.index_name, body=mapping, request_timeout=60)
        print(f"Created index: {self.index_name}")

    def _flatten_list(self, data):
        """Flatten list or return string"""
        if isinstance(data, list):
            return " ".join(str(x) for x in data if x)
        return str(data) if data else ""

    def _get_openfda_field(self, entry, field):
        """Safely get field from openfda section"""
        if "openfda" in entry and field in entry["openfda"]:
            return self._flatten_list(entry["openfda"][field])
        return ""

    def _tokenize_for_keyword(self, text: str) -> List[str]:
        normalized = unicodedata.normalize("NFKD", text or "")
        without_marks = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
        lowered = without_marks.lower()
        tokens = re.findall(r"[^\W_]+", lowered, flags=re.UNICODE)
        if not tokens:
            return []
        return sorted(set(tokens))

    def index_data(self, data_dir="fda_data"):
        """Index all FDA JSON files using bulk API"""
        json_files = [f for f in os.listdir(data_dir) if f.endswith(".json")]

        if not json_files:
            print("No JSON files found")
            return

        print(f"Indexing {len(json_files)} files...")
        total_indexed = 0

        for json_file in json_files:
            file_path = os.path.join(data_dir, json_file)
            print(f"Processing {json_file}...")

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "results" not in data:
                continue

            # Prepare bulk actions
            actions = []
            for entry in data["results"]:
                brand_names = self._get_openfda_field(entry, "brand_name")
                generic_names = self._get_openfda_field(entry, "generic_name")
                substance_names = self._get_openfda_field(entry, "substance_name")
                active_ingredient = self._flatten_list(entry.get("active_ingredient", ""))
                doc = {"_index": self.index_name, "_source": {"doc_id": entry.get("id", ""), "brand_names": brand_names, "generic_names": generic_names, "substance_names": substance_names, "manufacturer_names": self._get_openfda_field(entry, "manufacturer_name"), "product_type": self._get_openfda_field(entry, "product_type"), "route": self._get_openfda_field(entry, "route"), "purpose": self._flatten_list(entry.get("purpose", "")), "indications_and_usage": self._flatten_list(entry.get("indications_and_usage", "")), "warnings": self._flatten_list(entry.get("warnings", "")), "adverse_reactions": self._flatten_list(entry.get("adverse_reactions", "")), "drug_interactions": self._flatten_list(entry.get("drug_interactions", "")), "dosage_and_administration": self._flatten_list(entry.get("dosage_and_administration", "")), "active_ingredient": active_ingredient, "inactive_ingredient": self._flatten_list(entry.get("inactive_ingredient", "")), "full_data": entry}}
                actions.append(doc)

                # Bulk index every 1000 records
                if len(actions) >= 1000:
                    success, _ = helpers.bulk(self.es, actions, raise_on_error=False, request_timeout=600)
                    total_indexed += success
                    print(f"  Indexed {total_indexed} records...")
                    actions = []

            # Index remaining records
            if actions:
                success, _ = helpers.bulk(self.es, actions, raise_on_error=False)
                total_indexed += success

        print(f"\nTotal records indexed: {total_indexed}")

        # Refresh index to make documents searchable
        self.es.indices.refresh(index=self.index_name)

    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for drugs using multi-match query"""
        search_body = {"query": {"multi_match": {"query": query, "fields": ["brand_names^3", "generic_names^3", "substance_names^2", "active_ingredient^2", "purpose", "indications_and_usage", "warnings"], "type": "best_fields", "fuzziness": "AUTO"}}, "size": limit}

        response = self.es.search(index=self.index_name, body=search_body)

        results = []
        for hit in response["hits"]["hits"]:
            results.append({"score": hit["_score"], "data": hit["_source"]["full_data"]})

        return results

    def _build_drug_name_query(self, drug_name: str, mode: SearchMode) -> Dict[str, Any]:
        """Build the Elasticsearch query DSL for drug-name search.

        NOTE (benchmark correctness): SearchMode.TOKEN uses analyzer-consistent match semantics
        (Option A) rather than keyword token lists, to better align with FTS5 unicode61 MATCH.
        """
        drug_name = (drug_name or "").lower().strip()
        search_fields = [
            "brand_names^4",
            "generic_names^4",
            "substance_names^3",
            "active_ingredient^3",
        ]

        if mode == SearchMode.TOKEN:
            # Option A: analyzer-consistent match semantics with operator=and.
            # This avoids semantic mismatch vs FTS5 TOKEN which uses MATCH/unicode61.
            return {
                "multi_match": {
                    "query": drug_name,
                    "fields": search_fields,
                    "type": "best_fields",
                    "operator": "and",
                }
            }
        if mode == SearchMode.PREFIX:
            return {
                "multi_match": {
                    "query": drug_name,
                    "fields": search_fields,
                    "type": "bool_prefix",
                    "operator": "or",
                }
            }
        if mode == SearchMode.FUZZY:
            return {
                "multi_match": {
                    "query": drug_name,
                    "fields": search_fields,
                    "type": "best_fields",
                    "fuzziness": "AUTO",
                }
            }
        return {
            "multi_match": {
                "query": drug_name,
                "fields": search_fields,
                "type": "best_fields",
            }
        }

    def search_drug_name(self, drug_name: str, limit: Optional[int] = 10, mode: SearchMode = SearchMode.TOKEN) -> List[Dict[str, Any]]:
        """
        Search specifically by drug name with configurable matching mode

        Args:
            drug_name: The search term
            limit: Maximum results to return (None for unlimited - uses 10000)
            mode: Search mode (token, prefix, or fuzzy)
        """
        drug_name = drug_name.lower().strip()

        # Use a high limit if None specified (ES requires a size).
        # NOTE: Elasticsearch does not allow unbounded "size"; 10000 is the max default.
        # If you need more results, you must use search_after or scroll API.
        effective_limit = limit if limit is not None else 10000
        if limit is None and effective_limit == 10000:
            # User requested unlimited results, but we're capping at 10000
            # This is a known limitation - could add logging/warning here if needed
            pass

        search_body = {
            "query": self._build_drug_name_query(drug_name, mode),
            "size": effective_limit,
        }

        response = self.es.search(index=self.index_name, body=search_body)

        results = []
        for hit in response["hits"]["hits"]:
            results.append(hit["_source"]["full_data"])

        return results

    def count_drug_name(self, drug_name: str, mode: SearchMode = SearchMode.TOKEN) -> int:
        """Return a full match count for the given drug-name query (for parity checks)."""
        if not self.es or not self.index_exists():
            return 0
        q = {"query": self._build_drug_name_query(drug_name, mode)}
        try:
            res = self.es.count(index=self.index_name, body=q)
            return int(res.get("count", 0))
        except Exception:
            return 0

    def get_server_version(self) -> str:
        """Return Elasticsearch server version if known."""
        return self.server_version or "unknown"

    def get_analyzer_name(self) -> str:
        """Return the analyzer name used for drug name fields."""
        return self.analyzer_name

    def close(self):
        """Close Elasticsearch connection"""
        if self.es:
            self.es.close()


def main():
    indexer = ElasticsearchDrugIndexer()
    indexer.create_index()
    indexer.index_data()
    indexer.close()


if __name__ == "__main__":
    main()
