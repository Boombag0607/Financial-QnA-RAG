import os
import re
import json
import requests
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from pathlib import Path
import time
from urllib.parse import urljoin
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Document:
    """Document chunk with metadata"""
    content: str
    company: str
    year: str
    page: Optional[int] = None
    section: Optional[str] = None
    chunk_id: Optional[str] = None

@dataclass
class QueryResult:
    """Result from a query with sources"""
    query: str
    answer: str
    reasoning: str
    sub_queries: List[str]
    sources: List[Dict[str, Any]]

class SECDataCollector:
    """Collects 10-K filings from SEC EDGAR database"""
    
    BASE_URL = "https://www.sec.gov/Archives/edgar/data/"
    HEADERS = {
        'User-Agent': 'Financial RAG System educational@example.com'
    }
    
    COMPANY_CIKS = {
        'GOOGL': '1652044',
        'MSFT': '789019', 
        'NVDA': '1045810'
    }
    
    def __init__(self, data_dir: str = "sec_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    def get_company_filings(self, cik: str, filing_type: str = "10-K") -> List[Dict]:
        """Get filing information for a company"""
        url = f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"
        
        try:
            response = requests.get(url, headers=self.HEADERS)
            response.raise_for_status()
            data = response.json()
            
            filings = []
            recent = data.get('filings', {}).get('recent', {})
            
            for i, form in enumerate(recent.get('form', [])):
                if form == filing_type:
                    filing_date = recent['filingDate'][i]
                    year = int(filing_date.split('-')[0])
                    
                    if year in [2022, 2023, 2024]:  # Only get required years
                        filings.append({
                            'accessionNumber': recent['accessionNumber'][i],
                            'filingDate': filing_date,
                            'year': year,
                            'primaryDocument': recent['primaryDocument'][i]
                        })
            
            return sorted(filings, key=lambda x: x['year'])
            
        except Exception as e:
            logger.error(f"Error fetching filings for CIK {cik}: {e}")
            return []
    
    def download_filing(self, cik: str, accession_number: str, primary_document: str, 
                       company: str, year: int) -> Optional[str]:
        """Download a specific filing"""
        # Clean accession number for URL
        clean_accession = accession_number.replace('-', '')
        url = f"{self.BASE_URL}{cik}/{clean_accession}/{primary_document}"
        
        filename = f"{company}_{year}_10K.html"
        filepath = self.data_dir / filename
        
        if filepath.exists():
            logger.info(f"File already exists: {filepath}")
            return str(filepath)
        
        try:
            response = requests.get(url, headers=self.HEADERS)
            response.raise_for_status()
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(response.text)
            
            logger.info(f"Downloaded: {filepath}")
            time.sleep(1)  # Be respectful to SEC servers
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error downloading {url}: {e}")
            return None
    
    def collect_all_filings(self) -> Dict[str, List[str]]:
        """Collect all required filings"""
        all_files = {}
        
        for company, cik in self.COMPANY_CIKS.items():
            logger.info(f"Collecting filings for {company}")
            filings = self.get_company_filings(cik)
            company_files = []
            
            for filing in filings:
                filepath = self.download_filing(
                    cik, 
                    filing['accessionNumber'],
                    filing['primaryDocument'],
                    company,
                    filing['year']
                )
                if filepath:
                    company_files.append(filepath)
            
            all_files[company] = company_files
            
        return all_files

class DocumentProcessor:
    """Processes SEC filings and creates document chunks"""
    
    def __init__(self):
        self.financial_sections = [
            "item 7", "management's discussion", "md&a", 
            "item 8", "financial statements", "consolidated statements",
            "revenue", "operating income", "net income", "gross profit",
            "operating margin", "data center", "cloud", "azure", "gcp", "aws"
        ]
    
    def extract_text_from_html(self, filepath: str) -> str:
        """Extract text from HTML filing"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Simple HTML cleaning - remove scripts, styles
            content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL | re.IGNORECASE)
            content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.DOTALL | re.IGNORECASE)
            
            # Remove HTML tags
            content = re.sub(r'<[^>]+>', ' ', content)
            
            # Clean up whitespace
            content = re.sub(r'\s+', ' ', content)
            content = re.sub(r'\n+', '\n', content)
            
            return content.strip()
            
        except Exception as e:
            logger.error(f"Error extracting text from {filepath}: {e}")
            return ""
    
    def is_financial_relevant(self, text: str) -> bool:
        """Check if text chunk contains financial information"""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.financial_sections)
    
    def create_chunks(self, text: str, company: str, year: str, 
                     chunk_size: int = 800, overlap: int = 100) -> List[Document]:
        """Create overlapping text chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            # Only keep chunks that seem financially relevant
            if len(chunk_text.strip()) > 100 and self.is_financial_relevant(chunk_text):
                doc = Document(
                    content=chunk_text,
                    company=company,
                    year=year,
                    chunk_id=f"{company}_{year}_{i//chunk_size}"
                )
                chunks.append(doc)
        
        return chunks
    
    def process_filing(self, filepath: str) -> List[Document]:
        """Process a single filing into document chunks"""
        # Extract company and year from filename
        filename = Path(filepath).stem  # e.g., "GOOGL_2023_10K"
        parts = filename.split('_')
        company = parts[0]
        year = parts[1]
        
        text = self.extract_text_from_html(filepath)
        if not text:
            return []
        
        chunks = self.create_chunks(text, company, year)
        logger.info(f"Created {len(chunks)} chunks for {company} {year}")
        
        return chunks

class VectorStore:
    """Simple FAISS-based vector store"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = None
        self.index = None
        self.dimension = None
    
    def add_documents(self, documents: List[Document]):
        """Add documents to the vector store"""
        logger.info(f"Adding {len(documents)} documents to vector store")
        
        self.documents.extend(documents)
        
        # Encode all documents
        texts = [doc.content for doc in self.documents]
        embeddings = self.encoder.encode(texts, show_progress_bar=True)
        
        # Initialize FAISS index
        if self.dimension is None:
            self.dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(self.dimension)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
        
        self.embeddings = embeddings
        logger.info(f"Vector store now contains {len(self.documents)} documents")
    
    def search(self, query: str, k: int = 5, company_filter: str = None, 
               year_filter: str = None) -> List[Tuple[Document, float]]:
        """Search for relevant documents"""
        if self.index is None:
            return []
        
        # Encode query
        query_embedding = self.encoder.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), 
                                          min(k * 3, len(self.documents)))
        
        # Filter and rank results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.documents):
                doc = self.documents[idx]
                
                # Apply filters
                if company_filter and doc.company != company_filter:
                    continue
                if year_filter and doc.year != year_filter:
                    continue
                
                results.append((doc, float(score)))
        
        return results[:k]

class FinancialAgent:
    """Agent for query decomposition and multi-step reasoning"""
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        
        self.companies = ['GOOGL', 'MSFT', 'NVDA']
        self.years = ['2022', '2023', '2024']
        
        # Patterns for query decomposition
        self.comparison_patterns = [
            r'compare.*across.*companies',
            r'which company.*highest|lowest',
            r'how.*grow.*from.*to',
            r'growth.*from.*to',
            r'between.*and.*'
        ]
        
        self.metrics = {
            'revenue': ['revenue', 'net revenues', 'total revenue'],
            'operating_margin': ['operating margin', 'operating income margin'],
            'gross_margin': ['gross margin', 'gross profit margin'], 
            'rd_spending': ['research and development', 'r&d', 'research & development'],
            'cloud_revenue': ['cloud', 'azure', 'google cloud', 'gcp', 'aws'],
            'data_center': ['data center', 'datacenter', 'compute'],
            'advertising': ['advertising', 'ads', 'search advertising']
        }
    
    def needs_decomposition(self, query: str) -> bool:
        """Determine if query needs decomposition"""
        query_lower = query.lower()
        
        # Check for comparison patterns
        for pattern in self.comparison_patterns:
            if re.search(pattern, query_lower):
                return True
        
        # Check for multiple companies mentioned
        mentioned_companies = sum(1 for company in self.companies 
                                if company.lower() in query_lower)
        if mentioned_companies > 1:
            return True
            
        # Check for year ranges
        if re.search(r'\d{4}.*to.*\d{4}', query_lower):
            return True
            
        return False
    
    def decompose_query(self, query: str) -> List[str]:
        """Break down complex queries into sub-queries"""
        query_lower = query.lower()
        sub_queries = []
        
        # Extract companies mentioned or use all
        mentioned_companies = [c for c in self.companies 
                             if c.lower() in query_lower]
        if not mentioned_companies:
            mentioned_companies = self.companies
        
        # Extract years mentioned or use recent
        mentioned_years = []
        for year in self.years:
            if year in query_lower:
                mentioned_years.append(year)
        
        # Handle year ranges
        year_range_match = re.search(r'(\d{4}).*to.*(\d{4})', query_lower)
        if year_range_match:
            start_year, end_year = year_range_match.groups()
            mentioned_years = [start_year, end_year]
        
        if not mentioned_years:
            mentioned_years = ['2023']  # Default to most recent
        
        # Extract metric type
        metric_type = None
        for metric, keywords in self.metrics.items():
            if any(keyword in query_lower for keyword in keywords):
                metric_type = metric
                break
        
        # Generate sub-queries based on pattern
        if 'compare' in query_lower and len(mentioned_companies) > 1:
            for company in mentioned_companies:
                for year in mentioned_years:
                    if metric_type:
                        sub_queries.append(f"{company} {metric_type} {year}")
                    else:
                        sub_queries.append(f"{company} financial metrics {year}")
        
        elif 'grow' in query_lower or 'growth' in query_lower:
            company = mentioned_companies[0] if mentioned_companies else 'NVDA'
            if len(mentioned_years) >= 2:
                for year in mentioned_years:
                    sub_queries.append(f"{company} {metric_type or 'revenue'} {year}")
            else:
                sub_queries.append(f"{company} {metric_type or 'revenue'} 2022")
                sub_queries.append(f"{company} {metric_type or 'revenue'} 2023")
        
        elif 'which company' in query_lower:
            for company in self.companies:
                year = mentioned_years[0] if mentioned_years else '2023'
                sub_queries.append(f"{company} {metric_type or 'operating margin'} {year}")
        
        else:
            # Fallback: create simple sub-queries
            sub_queries = [query]
        
        return sub_queries if sub_queries else [query]
    
    def execute_query(self, query: str, k: int = 3) -> List[Tuple[Document, float]]:
        """Execute a single query against vector store"""
        return self.vector_store.search(query, k=k)
    
    def synthesize_answer(self, query: str, sub_queries: List[str], 
                         all_results: List[List[Tuple[Document, float]]]) -> str:
        """Synthesize final answer from multiple retrieval results"""
        
        # Collect all relevant information
        all_info = {}
        
        for sub_query, results in zip(sub_queries, all_results):
            for doc, score in results:
                key = f"{doc.company}_{doc.year}"
                if key not in all_info:
                    all_info[key] = []
                
                # Extract relevant snippets
                snippet = self.extract_relevant_snippet(sub_query, doc.content)
                if snippet:
                    all_info[key].append({
                        'content': snippet,
                        'score': score,
                        'sub_query': sub_query
                    })
        
        # Generate answer based on query type
        query_lower = query.lower()
        
        if 'compare' in query_lower or 'which company' in query_lower:
            return self.generate_comparison_answer(query, all_info)
        elif 'grow' in query_lower or 'growth' in query_lower:
            return self.generate_growth_answer(query, all_info)
        else:
            return self.generate_simple_answer(query, all_info)
    
    def extract_relevant_snippet(self, query: str, content: str, 
                               max_length: int = 200) -> str:
        """Extract most relevant snippet from content"""
        # Simple approach: find sentences with query keywords
        sentences = content.split('.')
        query_words = set(query.lower().split())
        
        best_sentence = ""
        best_score = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:  # Skip very short sentences
                continue
                
            sentence_words = set(sentence.lower().split())
            score = len(query_words.intersection(sentence_words))
            
            if score > best_score:
                best_score = score
                best_sentence = sentence
        
        return best_sentence[:max_length] if best_sentence else content[:max_length]
    
    def generate_comparison_answer(self, query: str, all_info: Dict) -> str:
        """Generate answer for comparison queries"""
        if not all_info:
            return "Unable to find sufficient information to make comparison."
        
        # Extract company performance from collected info
        companies_data = {}
        
        for key, info_list in all_info.items():
            company, year = key.split('_')
            
            if company not in companies_data:
                companies_data[company] = {}
            
            # Look for numerical values in the content
            for info in info_list:
                content = info['content']
                # Simple extraction of percentages and numbers
                numbers = re.findall(r'(\d+\.?\d*)%?', content)
                if numbers:
                    companies_data[company][year] = {
                        'value': numbers[0],
                        'context': content[:100]
                    }
        
        # Generate comparison
        if companies_data:
            answer_parts = []
            for company, data in companies_data.items():
                if data:
                    year_data = list(data.values())[0]  # Get first available year
                    answer_parts.append(f"{company}: {year_data['value']}")
            
            if answer_parts:
                return f"Based on available data: {', '.join(answer_parts)}"
        
        return "Found relevant information but unable to extract specific comparison values."
    
    def generate_growth_answer(self, query: str, all_info: Dict) -> str:
        """Generate answer for growth/change queries"""
        if not all_info:
            return "Unable to find sufficient information to calculate growth."
        
        # Look for data from different years
        company_years = {}
        
        for key, info_list in all_info.items():
            company, year = key.split('_')
            if company not in company_years:
                company_years[company] = {}
                
            for info in info_list:
                content = info['content']
                numbers = re.findall(r'(\d+\.?\d*)', content)
                if numbers:
                    company_years[company][year] = {
                        'value': float(numbers[0]),
                        'context': content[:100]
                    }
        
        # Calculate growth if we have multiple years
        for company, years_data in company_years.items():
            if len(years_data) >= 2:
                years = sorted(years_data.keys())
                start_value = years_data[years[0]]['value']
                end_value = years_data[years[-1]]['value']
                
                if start_value > 0:
                    growth_pct = ((end_value - start_value) / start_value) * 100
                    return f"{company} grew from {start_value} to {end_value} ({growth_pct:.1f}% growth) from {years[0]} to {years[-1]}"
        
        return "Found relevant information but unable to calculate specific growth rate."
    
    def generate_simple_answer(self, query: str, all_info: Dict) -> str:
        """Generate answer for simple queries"""
        if not all_info:
            return "No relevant information found."
        
        # Get the best matching information
        best_info = None
        best_score = 0
        
        for key, info_list in all_info.items():
            for info in info_list:
                if info['score'] > best_score:
                    best_score = info['score']
                    best_info = info
        
        if best_info:
            return f"Based on the filing: {best_info['content']}"
        
        return "Found some relevant information but unable to provide specific answer."
    
    def process_query(self, query: str) -> QueryResult:
        """Main method to process a query with agent capabilities"""
        logger.info(f"Processing query: {query}")
        
        # Determine if decomposition is needed
        if self.needs_decomposition(query):
            logger.info("Query requires decomposition")
            sub_queries = self.decompose_query(query)
        else:
            logger.info("Simple query, no decomposition needed")
            sub_queries = [query]
        
        logger.info(f"Sub-queries: {sub_queries}")
        
        # Execute all sub-queries
        all_results = []
        all_sources = []
        
        for sub_query in sub_queries:
            results = self.execute_query(sub_query, k=3)
            all_results.append(results)
            
            # Collect sources
            for doc, score in results:
                source = {
                    "company": doc.company,
                    "year": doc.year, 
                    "excerpt": doc.content[:200] + "...",
                    "score": round(score, 3),
                    "chunk_id": doc.chunk_id
                }
                all_sources.append(source)
        
        # Remove duplicate sources
        unique_sources = []
        seen_chunks = set()
        for source in all_sources:
            if source["chunk_id"] not in seen_chunks:
                unique_sources.append(source)
                seen_chunks.add(source["chunk_id"])
        
        # Synthesize final answer
        answer = self.synthesize_answer(query, sub_queries, all_results)
        
        reasoning = f"Executed {len(sub_queries)} sub-queries and synthesized results from {len(unique_sources)} document chunks."
        
        return QueryResult(
            query=query,
            answer=answer,
            reasoning=reasoning,
            sub_queries=sub_queries,
            sources=unique_sources[:5]  # Limit to top 5 sources
        )

def main():
    """Main execution function"""
    print("🏦 Financial RAG System with Agent Capabilities")
    print("=" * 50)
    
    # Test queries
    test_queries = [
        "What was NVIDIA's total revenue in fiscal year 2024?",
        "What percentage of Google's 2023 revenue came from advertising?",
        "How much did Microsoft's cloud revenue grow from 2022 to 2023?",
        "Which of the three companies had the highest gross margin in 2023?",
        "Compare the R&D spending as a percentage of revenue across all three companies in 2023"
    ]
    
    try:
        # Initialize components
        print("\n1. Initializing Data Collector...")
        collector = SECDataCollector()
        
        print("\n2. Collecting SEC Filings...")
        # For demo purposes, we'll assume some files exist or create mock data
        # In real implementation, uncomment the next line:
        # all_files = collector.collect_all_filings()
        
        print("\n3. Processing Documents...")
        processor = DocumentProcessor()
        
        # Create some mock documents for demo
        mock_documents = []
        for company in ['GOOGL', 'MSFT', 'NVDA']:
            for year in ['2022', '2023', '2024']:
                # In real implementation, you would process actual files
                mock_content = f"""
                {company} Annual Report {year}
                
                Total revenue for fiscal year {year} was ${'100' if company == 'GOOGL' else '80' if company == 'MSFT' else '60'} billion.
                Operating margin improved to {'29.8' if company == 'GOOGL' else '42.1' if company == 'MSFT' else '32.5'}% in {year}.
                
                Research and development expenses were ${'25' if company == 'GOOGL' else '20' if company == 'MSFT' else '15'} billion, 
                representing {'16' if company == 'GOOGL' else '14' if company == 'MSFT' else '18'}% of total revenue.
                
                Cloud revenue {'Google Cloud generated $26 billion' if company == 'GOOGL' else 
                'Microsoft Azure and cloud services revenue was $45 billion' if company == 'MSFT' else 
                'Data center revenue was $47 billion'} in fiscal {year}.
                """
                
                doc = Document(
                    content=mock_content,
                    company=company,
                    year=year,
                    chunk_id=f"{company}_{year}_mock"
                )
                mock_documents.append(doc)
        
        print(f"Created {len(mock_documents)} mock document chunks")
        
        print("\n4. Building Vector Store...")
        vector_store = VectorStore()
        vector_store.add_documents(mock_documents)
        
        print("\n5. Initializing Financial Agent...")
        agent = FinancialAgent(vector_store)
        
        print("\n6. Testing Queries...")
        print("=" * 50)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🔍 Query {i}: {query}")
            print("-" * 40)
            
            try:
                result = agent.process_query(query)
                
                print(f"💡 Answer: {result.answer}")
                print(f"🧠 Reasoning: {result.reasoning}")
                print(f"📋 Sub-queries: {', '.join(result.sub_queries)}")
                print(f"📚 Sources: {len(result.sources)} documents")
                
                # Show JSON output
                result_json = asdict(result)
                print(f"\n📄 JSON Response:")
                print(json.dumps(result_json, indent=2))
                
            except Exception as e:
                print(f"❌ Error processing query: {e}")
            
            print("\n" + "=" * 50)
        
        print("\n✅ Financial RAG System Demo Complete!")
        print("\nTo run with real SEC data:")
        print("1. Uncomment the data collection lines")
        print("2. Install additional dependencies: pip install beautifulsoup4 lxml")
        print("3. Wait for SEC filings to download")
        
    except Exception as e:
        print(f"❌ Error in main execution: {e}")
        logger.error(f"Main execution error: {e}")

if __name__ == "__main__":
    main()