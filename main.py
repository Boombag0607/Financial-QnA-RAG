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
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Document:
    """Document chunk with metadata and HTML awareness"""
    content: str
    company: str
    year: str
    page: Optional[int] = None
    section: Optional[str] = None
    chunk_id: Optional[str] = None

@dataclass
class QueryResult:
    """Result from a query with enhanced source attribution"""
    query: str
    answer: str
    reasoning: str
    sub_queries: List[str]
    sources: List[Dict[str, Any]]

class SECDataCollector:
    """Collects 10-K filings from SEC EDGAR database with HTML focus"""
    
    BASE_URL = "https://www.sec.gov/Archives/edgar/data/"
    HEADERS = {
        'User-Agent': 'Financial RAG System educational@example.com'
    }
    
    COMPANY_CIKS = {
        'GOOGL': '1652044',
        'MSFT': '789019', 
        'NVDA': '1045810'
    }
    
    def __init__(self, data_dir: str = "data/sec_filings"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
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
        """Download a specific filing in HTML format"""
        # Clean accession number for URL
        clean_accession = accession_number.replace('-', '')
        url = f"{self.BASE_URL}{cik}/{clean_accession}/{primary_document}"
        
        filename = f"{company}_{year}_10K.html"
        filepath = self.data_dir / filename
        
        if filepath.exists():
            logger.info(f"HTML file already exists: {filepath}")
            return str(filepath)
        
        try:
            response = requests.get(url, headers=self.HEADERS)
            response.raise_for_status()
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(response.text)
            
            logger.info(f"Downloaded HTML filing: {filepath}")
            time.sleep(1)  # Be respectful to SEC servers
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error downloading {url}: {e}")
            return None
    
    def collect_all_filings(self) -> Dict[str, List[str]]:
        """Collect all required filings in HTML format"""
        all_files = {}
        
        for company, cik in self.COMPANY_CIKS.items():
            logger.info(f"Collecting HTML filings for {company}")
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
    """Enhanced HTML document processor with table extraction capabilities"""
    
    def __init__(self):
        self.financial_sections = [
            "item 7", "management's discussion", "md&a", 
            "item 8", "financial statements", "consolidated statements",
            "revenue", "operating income", "net income", "gross profit",
            "operating margin", "data center", "cloud", "azure", "gcp", "aws"
        ]
        
        # HTML table patterns for financial data
        self.table_indicators = [
            "consolidated statements", "income statement", "balance sheet",
            "cash flow", "revenue", "expenses", "assets", "liabilities",
            "fiscal year", "three months ended", "year ended"
        ]
        
    def extract_html_content(self, filepath: str) -> dict:
        """Extract structured content from HTML filing preserving table structure"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove scripts and styles but keep structure
            for script in soup(["script", "style"]):
                script.decompose()
            
            content_parts = {
                'text_content': '',
                'tables': [],
                'raw_html': html_content[:50000]  # Keep first 50k chars for context
            }
            
            # Extract tables with context
            tables = soup.find_all('table')
            logger.info(f"Found {len(tables)} tables in HTML document")
            
            for i, table in enumerate(tables):
                table_text = self._extract_table_content(table)
                if self._is_financial_table(table_text):
                    content_parts['tables'].append({
                        'table_id': i,
                        'content': table_text,
                        'html': str(table)[:2000],  # First 2000 chars of table HTML
                        'type': self._classify_table_type(table_text)
                    })
            
            # Extract regular text content (non-table)
            for table in soup.find_all('table'):
                table.decompose()  # Remove tables from text extraction
            
            content_parts['text_content'] = soup.get_text(separator=' ', strip=True)
            
            return content_parts
            
        except Exception as e:
            logger.error(f"Error extracting HTML content from {filepath}: {e}")
            return {'text_content': '', 'tables': [], 'raw_html': ''}
    
    def _extract_table_content(self, table) -> str:
        """Extract meaningful content from HTML table"""
        rows = []
        
        for row in table.find_all('tr'):
            cells = []
            for cell in row.find_all(['td', 'th']):
                cell_text = cell.get_text(strip=True)
                if cell_text:
                    cells.append(cell_text)
            
            if cells:  # Only add non-empty rows
                rows.append(' | '.join(cells))
        
        return '\n'.join(rows)
    
    def _is_financial_table(self, table_content: str) -> bool:
        """Determine if table contains financial data"""
        content_lower = table_content.lower()
        
        # Look for financial indicators
        financial_indicators = [
            'revenue', 'income', 'expense', 'margin', 'profit', 'loss',
            'assets', 'liabilities', 'equity', 'cash', 'million', 'billion',
            '$', 'fiscal', 'quarter', 'year ended', 'three months'
        ]
        
        indicator_count = sum(1 for indicator in financial_indicators 
                            if indicator in content_lower)
        
        # Also check for numerical patterns
        number_patterns = len(re.findall(r'\$?[\d,]+\.?\d*', table_content))
        
        return indicator_count >= 2 and number_patterns >= 3
    
    def _classify_table_type(self, table_content: str) -> str:
        """Classify the type of financial table"""
        content_lower = table_content.lower()
        
        if any(term in content_lower for term in ['income statement', 'operations', 'revenue', 'operating income']):
            return 'income_statement'
        elif any(term in content_lower for term in ['balance sheet', 'assets', 'liabilities']):
            return 'balance_sheet'
        elif any(term in content_lower for term in ['cash flow', 'cash flows', 'operating activities']):
            return 'cash_flow'
        elif any(term in content_lower for term in ['segment', 'revenue by', 'geographic']):
            return 'segment_data'
        else:
            return 'financial_data'
    
    def create_html_aware_chunks(self, content_parts: dict, company: str, year: str,
                               chunk_size: int = 1000, overlap: int = 150) -> List[Document]:
        """Create chunks that preserve HTML table structure and context"""
        chunks = []
        
        # Process tables as separate chunks (they're highly structured)
        for table_info in content_parts['tables']:
            table_content = f"""
            Financial Table - {table_info['type'].replace('_', ' ').title()}
            Company: {company} | Year: {year}
            
            Table Data:
            {table_info['content']}
            
            HTML Structure Context:
            {table_info['html'][:500]}...
            """
            
            if len(table_content.strip()) > 100:
                doc = Document(
                    content=table_content,
                    company=company,
                    year=year,
                    section=f"table_{table_info['table_id']}",
                    chunk_id=f"{company}_{year}_table_{table_info['table_id']}"
                )
                chunks.append(doc)
        
        # Process regular text content in overlapping chunks
        text_content = content_parts['text_content']
        if text_content:
            words = text_content.split()
            
            for i in range(0, len(words), chunk_size - overlap):
                chunk_words = words[i:i + chunk_size]
                chunk_text = ' '.join(chunk_words)
                
                if len(chunk_text.strip()) > 100 and self.is_financial_relevant(chunk_text):
                    # Add HTML context for better understanding
                    enhanced_content = f"""
                    Text Content - {company} {year}
                    
                    {chunk_text}
                    
                    [Document contains {len(content_parts['tables'])} financial tables]
                    """
                    
                    doc = Document(
                        content=enhanced_content,
                        company=company,
                        year=year,
                        section="narrative_text",
                        chunk_id=f"{company}_{year}_text_{i//chunk_size}"
                    )
                    chunks.append(doc)
        
        return chunks
    
    def is_financial_relevant(self, text: str) -> bool:
        """Enhanced financial relevance check"""
        text_lower = text.lower()
        
        # Original keyword check
        keyword_match = any(keyword in text_lower for keyword in self.financial_sections)
        
        # Check for financial numbers
        has_financial_numbers = bool(re.search(r'\$[\d,]+\.?\d*\s*(million|billion|thousand)', text_lower))
        
        # Check for percentage metrics
        has_percentages = bool(re.search(r'\d+\.?\d*%', text))
        
        return keyword_match or has_financial_numbers or has_percentages
    
    def process_filing(self, filepath: str) -> List[Document]:
        """Process a single HTML filing into document chunks with table awareness"""
        # Extract company and year from filename
        filename = Path(filepath).stem  # e.g., "GOOGL_2023_10K"
        parts = filename.split('_')
        company = parts[0]
        year = parts[1]
        
        logger.info(f"Processing HTML filing: {filepath}")
        
        # Extract HTML content with table structure
        content_parts = self.extract_html_content(filepath)
        
        if not content_parts['text_content'] and not content_parts['tables']:
            logger.warning(f"No content extracted from {filepath}")
            return []
        
        # Create HTML-aware chunks
        chunks = self.create_html_aware_chunks(content_parts, company, year)
        
        logger.info(f"Created {len(chunks)} chunks for {company} {year} "
                   f"({len(content_parts['tables'])} table chunks, "
                   f"{len(chunks) - len(content_parts['tables'])} text chunks)")
        
        return chunks

class VectorStore:
    """Enhanced FAISS-based vector store with table awareness"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = None
        self.index = None
        self.dimension = None
    
    def add_documents(self, documents: List[Document]):
        """Add documents to the vector store with enhanced indexing"""
        logger.info(f"Adding {len(documents)} documents to vector store")
        
        # Separate table and text documents for enhanced processing
        table_docs = [doc for doc in documents if 'table_' in doc.chunk_id]
        text_docs = [doc for doc in documents if 'table_' not in doc.chunk_id]
        
        logger.info(f"  - {len(table_docs)} table documents")
        logger.info(f"  - {len(text_docs)} text documents")
        
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
               year_filter: str = None, prefer_tables: bool = False) -> List[Tuple[Document, float]]:
        """Enhanced search with table preference option"""
        if self.index is None:
            return []
        
        # Encode query
        query_embedding = self.encoder.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search for more results initially if we need to filter
        search_k = min(k * 3, len(self.documents))
        scores, indices = self.index.search(query_embedding.astype('float32'), search_k)
        
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
                
                # Boost table documents if preferred
                final_score = float(score)
                if prefer_tables and 'table_' in doc.chunk_id:
                    final_score *= 1.2
                
                results.append((doc, final_score))
        
        # Sort by final score and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

class FinancialAgent:
    """Enhanced agent for HTML-aware financial analysis with advanced table processing"""
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        
        self.companies = ['GOOGL', 'MSFT', 'NVDA']
        self.years = ['2022', '2023', '2024']
        
        # Enhanced patterns for HTML-aware queries
        self.comparison_patterns = [
            r'compare.*across.*companies',
            r'which company.*highest|lowest',
            r'how.*grow.*from.*to',
            r'growth.*from.*to',
            r'between.*and.*'
        ]
        
        # Enhanced metrics with table-specific patterns
        self.metrics = {
            'revenue': ['revenue', 'net revenues', 'total revenue', 'sales'],
            'operating_margin': ['operating margin', 'operating income margin', 'margin from operations'],
            'gross_margin': ['gross margin', 'gross profit margin'], 
            'rd_spending': ['research and development', 'r&d', 'research & development'],
            'cloud_revenue': ['cloud', 'azure', 'google cloud', 'gcp', 'aws', 'cloud services'],
            'data_center': ['data center', 'datacenter', 'compute'],
            'advertising': ['advertising', 'ads', 'search advertising'],
            'total_assets': ['total assets', 'assets'],
            'cash': ['cash and cash equivalents', 'cash', 'liquid assets']
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
    
    def enhanced_query_search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """Enhanced search that prioritizes table content for structured queries"""
        # Check if this is a numerical query that would benefit from table data
        prefer_tables = self._is_numerical_query(query)
        
        # Use enhanced search with table preference
        return self.vector_store.search(query, k=k, prefer_tables=prefer_tables)
    
    def _is_numerical_query(self, query: str) -> bool:
        """Check if query is asking for numerical/tabular data"""
        numerical_indicators = [
            'revenue', 'margin', 'income', 'profit', 'loss', 'assets', 'cash',
            'percentage', '%', 'billion', 'million', 'growth', 'compare',
            'highest', 'lowest', 'total', 'expenses', 'breakdown', 'segment'
        ]
        
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in numerical_indicators)
    
    def extract_table_values(self, content: str, metric_type: str) -> dict:
        """Enhanced extraction from HTML table content"""
        results = {}
        
        # Table-specific patterns that work with HTML structure
        table_patterns = {
            'revenue': [
                r'(?:total\s+)?(?:net\s+)?revenues?\s*[|:]\s*\$?([\d,]+\.?\d*)',
                r'(?:net\s+)?revenues?\s*\$?([\d,]+\.?\d*)\s*(?:million|billion)',
                r'revenues?\s*[|]\s*\$?([\d,]+\.?\d*)'
            ],
            'operating_margin': [
                r'operating\s+margin\s*[|:]\s*(\d+\.?\d*)%',
                r'margin\s*[|]\s*(\d+\.?\d*)%',
                r'(\d+\.?\d*)%\s*[|]\s*operating'
            ],
            'rd_spending': [
                r'research\s+(?:and\s+)?development\s*[|:]\s*\$?([\d,]+\.?\d*)',
                r'r&d\s*[|:]\s*\$?([\d,]+\.?\d*)',
                r'(\d+\.?\d*)%?\s*[|]\s*research'
            ],
            'cloud_revenue': [
                r'cloud\s*(?:services)?\s*[|:]\s*\$?([\d,]+\.?\d*)',
                r'azure\s*[|:]\s*\$?([\d,]+\.?\d*)',
                r'(?:google\s+)?cloud\s*[|]\s*\$?([\d,]+\.?\d*)'
            ],
            'data_center': [
                r'data\s+center\s*[|:]\s*\$?([\d,]+\.?\d*)',
                r'datacenter\s*[|:]\s*\$?([\d,]+\.?\d*)'
            ]
        }
        
        patterns = table_patterns.get(metric_type, [])
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
            if matches:
                # Clean and convert the first match
                value_str = matches[0].replace(',', '')
                try:
                    if '%' in pattern:
                        results['value'] = float(value_str)
                        results['type'] = 'percentage'
                    else:
                        results['value'] = float(value_str)
                        results['type'] = 'currency'
                        
                        # Detect units from context
                        if 'billion' in content.lower():
                            results['unit'] = 'billion'
                        elif 'million' in content.lower():
                            results['unit'] = 'million'
                        else:
                            results['unit'] = 'unknown'
                    
                    results['raw_match'] = matches[0]
                    break
                except ValueError:
                    continue
        
        return results
    
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
    
    def synthesize_table_answer(self, query: str, sub_queries: List[str],
                              all_results: List[List[Tuple[Document, float]]]) -> str:
        """Enhanced synthesis that leverages table structure"""
        
        # Collect structured data from tables
        structured_data = {}
        
        for sub_query, results in zip(sub_queries, all_results):
            # Extract metric type from sub-query
            metric_type = self._extract_metric_type(sub_query)
            
            for doc, score in results:
                company = doc.company
                year = doc.year
                
                key = f"{company}_{year}"
                
                if key not in structured_data:
                    structured_data[key] = {}
                
                # Use enhanced table extraction
                if 'table_' in doc.chunk_id:
                    table_values = self.extract_table_values(doc.content, metric_type)
                    if table_values:
                        structured_data[key][metric_type] = table_values
                        structured_data[key][metric_type]['source_type'] = 'table'
                        structured_data[key][metric_type]['confidence'] = score * 1.2
                else:
                    # Fallback to regular text extraction
                    snippet = self.extract_relevant_snippet(sub_query, doc.content)
                    if snippet:
                        structured_data[key][metric_type] = {
                            'content': snippet,
                            'source_type': 'text',
                            'confidence': score
                        }
        
        # Generate analysis based on structured data
        return self._generate_structured_analysis(query, structured_data)
    
    def _extract_metric_type(self, query: str) -> str:
        """Extract the type of metric being queried"""
        query_lower = query.lower()
        
        for metric, keywords in self.metrics.items():
            if any(keyword in query_lower for keyword in keywords):
                return metric
        
        return 'general'
    
    def _generate_structured_analysis(self, query: str, structured_data: dict) -> str:
        """Generate analysis from structured table data"""
        query_lower = query.lower()
        
        if not structured_data:
            return "Unable to find sufficient structured data to answer the query."
        
        # Different analysis patterns based on query type
        if 'compare' in query_lower or 'which company' in query_lower:
            return self._generate_table_comparison(structured_data)
        elif 'grow' in query_lower or 'growth' in query_lower:
            return self._generate_table_growth_analysis(structured_data)
        elif 'breakdown' in query_lower or 'segment' in query_lower:
            return self._generate_segment_analysis(structured_data)
        else:
            return self._generate_table_direct_answer(structured_data)
    
    def _generate_table_comparison(self, structured_data: dict) -> str:
        """Generate comparison from table data"""
        company_metrics = {}
        
        for key, data in structured_data.items():
            company, year = key.split('_')
            
            for metric, values in data.items():
                if 'value' in values:
                    if company not in company_metrics:
                        company_metrics[company] = {}
                    
                    company_metrics[company][metric] = values
        
        if not company_metrics:
            return "Unable to extract comparable metrics from the data."
        
        # Create comparison for the first available metric
        first_metric = list(list(company_metrics.values())[0].keys())[0]
        comparison_data = {}
        
        for company, metrics in company_metrics.items():
            if first_metric in metrics:
                comparison_data[company] = metrics[first_metric]['value']
        
        if comparison_data:
            sorted_companies = sorted(comparison_data.items(), key=lambda x: x[1], reverse=True)
            winner = sorted_companies[0]
            
            answer = f"Based on table analysis: {winner[0]} leads with {winner[1]}"
            
            # Add unit information
            sample_data = list(company_metrics.values())[0][first_metric]
            if sample_data.get('type') == 'percentage':
                answer += "%"
            elif sample_data.get('unit'):
                answer += f" {sample_data['unit']}"
            
            # Add other companies
            if len(sorted_companies) > 1:
                others = [f"{comp}: {val}" for comp, val in sorted_companies[1:]]
                if sample_data.get('type') == 'percentage':
                    others = [f"{comp}: {val}%" for comp, val in sorted_companies[1:]]
                elif sample_data.get('unit'):
                    others = [f"{comp}: {val} {sample_data['unit']}" for comp, val in sorted_companies[1:]]
                answer += f", followed by {', '.join(others)}"
            
            return answer