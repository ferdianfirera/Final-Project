import json
import re
import os
from typing import List, Dict, Optional


class QueryRecommender:
    """
    Query recommendation engine using collaborative filtering and context-based ranking.
    Recommends relevant business questions based on conversation history and current context.
    """
    
    def __init__(self, catalog_path: str = "question_catalog.json"):
        """Initialize the recommender with question catalog."""
        self.catalog_path = catalog_path
        self.questions = self._load_catalog()
        self.category_map = self._build_category_map()
        
    def _load_catalog(self) -> List[Dict]:
        """Load question catalog from JSON file."""
        try:
            with open(self.catalog_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("questions", [])
        except Exception as e:
            print(f"Error loading question catalog: {e}")
            return []
    
    def _build_category_map(self) -> Dict[str, List[str]]:
        """Build a map of categories to related categories."""
        return {
            "revenue": ["product", "financial", "temporal", "location"],
            "customer": ["satisfaction", "location", "product"],
            "operational": ["seller", "location", "temporal"],
            "product": ["revenue", "satisfaction", "customer"],
            "financial": ["revenue", "customer", "payment"],
            "temporal": ["revenue", "operational", "product"],
            "seller": ["operational", "location", "satisfaction"],
            "satisfaction": ["customer", "product", "seller"],
            "location": ["revenue", "customer", "operational", "seller"]
        }
    
    def get_recommendations(
        self, 
        current_query: str, 
        chat_history: List[Dict], 
        num_recommendations: int = 6
    ) -> List[Dict]:
        """
        Get top N question recommendations based on current context.
        
        Args:
            current_query: The most recent user query
            chat_history: List of previous messages
            num_recommendations: Number of recommendations to return
            
        Returns:
            List of recommended questions with metadata
        """
        # Extract context from query and history
        context = self.extract_context(current_query, chat_history)
        
        # Filter questions based on context
        candidates = self.filter_by_context(context, chat_history)
        
        # Rank candidates
        ranked = self.rank_recommendations(candidates, context, chat_history)
        
        # Select top N
        top_recommendations = ranked[:num_recommendations]
        
        # Substitute template variables
        final_recommendations = []
        for rec in top_recommendations:
            rec_copy = rec.copy()
            rec_copy["text"] = self.substitute_template_vars(rec_copy, context)
            final_recommendations.append(rec_copy)
        
        return final_recommendations
    
    def extract_context(self, query: str, history: List[Dict]) -> Dict:
        """
        Extract context from current query and chat history.
        
        Returns dict with:
            - location: Detected city/state name
            - time_range: Detected time period
            - metric_type: COUNT, SUM, AVG, etc.
            - category: Detected business category
            - last_agent: Last agent used (SQL/RAG)
        """
        context = {
            "location": None,
            "time_range": None,
            "metric_type": None,
            "category": None,
            "last_agent": None,
            "last_question_id": None
        }
        
        # Extract location from current query
        location = self._extract_location(query)
        if location:
            context["location"] = location
        
        # Extract time range from current query
        time_range = self._extract_time_range(query)
        if time_range:
            context["time_range"] = time_range
        
        # Scan recent history (last 5 messages)
        recent_history = history[-5:] if len(history) > 5 else history
        
        for msg in reversed(recent_history):
            # Extract from SQL queries
            if msg.get("sql"):
                sql = msg.get("sql", "")
                
                # Extract location from SQL if not found in query
                if not context["location"]:
                    context["location"] = self._extract_location_from_sql(sql)
                
                # Extract metric type
                if not context["metric_type"]:
                    context["metric_type"] = self._extract_metric_from_sql(sql)
            
            # Extract agent type
            if msg.get("agent"):
                context["last_agent"] = msg.get("agent")
                break
        
        # Detect category from query keywords
        context["category"] = self._detect_category(query)
        
        return context
    
    def _extract_location(self, text: str) -> Optional[str]:
        """Extract Brazilian city/state names from text."""
        # Common Brazilian cities
        cities = [
            "campinas", "são paulo", "rio de janeiro", "curitiba", 
            "brasília", "salvador", "fortaleza", "belo horizonte",
            "manaus", "recife", "porto alegre", "goiânia"
        ]
        
        text_lower = text.lower()
        for city in cities:
            if city in text_lower:
                return city.title()
        
        # State codes (2 letters)
        state_match = re.search(r'\b([A-Z]{2})\b', text)
        if state_match:
            return state_match.group(1)
        
        return None
    
    def _extract_location_from_sql(self, sql: str) -> Optional[str]:
        """Extract location from SQL WHERE clause."""
        sql_upper = sql.upper()
        
        # Look for city in WHERE clause
        city_pattern = r"(?:CUSTOMER_CITY|GEOLOCATION_CITY).*?(?:=|LIKE).*?['\"]%?([^'\"]+?)%?['\"]"
        city_match = re.search(city_pattern, sql_upper, re.IGNORECASE)
        if city_match:
            return city_match.group(1).title()
        
        # Look for state
        state_pattern = r"(?:CUSTOMER_STATE|GEOLOCATION_STATE).*?(?:=|LIKE).*?['\"]%?(\w{2})%?['\"]"
        state_match = re.search(state_pattern, sql_upper, re.IGNORECASE)
        if state_match:
            return state_match.group(1).upper()
        
        return None
    
    def _extract_time_range(self, text: str) -> Optional[str]:
        """Extract time range mentions from text."""
        text_lower = text.lower()
        
        # Year patterns
        year_match = re.search(r'\b(20\d{2})\b', text)
        if year_match:
            return f"year_{year_match.group(1)}"
        
        # Month patterns
        if any(word in text_lower for word in ["bulan", "month", "bulanan", "monthly"]):
            return "monthly"
        
        # Quarter patterns
        if any(word in text_lower for word in ["kuartal", "quarter", "quarterly"]):
            return "quarterly"
        
        # Week patterns
        if any(word in text_lower for word in ["minggu", "week", "weekly"]):
            return "weekly"
        
        # Day patterns
        if any(word in text_lower for word in ["hari", "day", "daily", "today"]):
            return "daily"
        
        return None
    
    def _extract_metric_from_sql(self, sql: str) -> Optional[str]:
        """Extract metric type from SQL SELECT clause."""
        sql_upper = sql.upper()
        
        if "COUNT(" in sql_upper:
            return "COUNT"
        elif "SUM(" in sql_upper:
            return "SUM"
        elif "AVG(" in sql_upper:
            return "AVG"
        elif "MAX(" in sql_upper:
            return "MAX"
        elif "MIN(" in sql_upper:
            return "MIN"
        
        return None
    
    def _detect_category(self, text: str) -> Optional[str]:
        """Detect business category from query keywords."""
        text_lower = text.lower()
        
        # Category keyword mapping
        category_keywords = {
            "revenue": ["revenue", "pendapatan", "penjualan", "sales", "menguntungkan"],
            "customer": ["customer", "pelanggan", "pembeli"],
            "operational": ["pesanan", "order", "delivery", "pengiriman", "status"],
            "product": ["produk", "product", "kategori", "category"],
            "financial": ["payment", "pembayaran", "credit", "boleto", "installment"],
            "temporal": ["tren", "trend", "bulan", "month", "tahun", "year"],
            "seller": ["seller", "penjual", "toko"],
            "satisfaction": ["review", "rating", "score", "satisfaction", "kepuasan"],
            "location": ["kota", "city", "state", "region", "wilayah"]
        }
        
        for category, keywords in category_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return category
        
        return None
    
    def filter_by_context(self, context: Dict, history: List[Dict]) -> List[Dict]:
        """Filter questions based on extracted context."""
        candidates = []
        
        # If location detected, prioritize location-specific questions
        if context.get("location"):
            location_questions = [q for q in self.questions if "location_detected" in q.get("context_triggers", [])]
            candidates.extend(location_questions)
        
        # Add questions from detected category
        if context.get("category"):
            category = context["category"]
            category_questions = [q for q in self.questions if q.get("category") == category]
            candidates.extend(category_questions)
            
            # Add related category questions
            related_categories = self.category_map.get(category, [])
            for rel_cat in related_categories[:2]:  # Limit to 2 related categories
                related_questions = [q for q in self.questions if q.get("category") == rel_cat]
                candidates.extend(related_questions[:3])  # Max 3 per related category
        
        # If no specific context, use high-priority general questions
        if not candidates:
            candidates = [q for q in self.questions if q.get("priority", 0) >= 8]
        
        # Remove duplicates
        seen_ids = set()
        unique_candidates = []
        for q in candidates:
            if q["id"] not in seen_ids:
                unique_candidates.append(q)
                seen_ids.add(q["id"])
        
        return unique_candidates
    
    def rank_recommendations(
        self, 
        candidates: List[Dict], 
        context: Dict, 
        history: List[Dict]
    ) -> List[Dict]:
        """
        Rank candidate questions using scoring algorithm.
        
        Scoring:
            - Context match: 40%
            - Sequential pattern: 30%
            - Recency: 15%
            - Priority: 15%
        """
        scored_candidates = []
        
        for question in candidates:
            score = 0
            
            # Context match (40 points max)
            context_score = self._calculate_context_score(question, context)
            score += context_score * 0.4
            
            # Sequential pattern (30 points max)
            sequential_score = self._calculate_sequential_score(question, history)
            score += sequential_score * 0.3
            
            # Recency (15 points max) - higher if not recently asked
            recency_score = self._calculate_recency_score(question, history)
            score += recency_score * 0.15
            
            # Priority weight (15 points max)
            priority_score = question.get("priority", 5) * 10  # Scale 1-10 to 10-100
            score += priority_score * 0.15
            
            scored_candidates.append({
                **question,
                "score": score
            })
        
        # Sort by score descending
        scored_candidates.sort(key=lambda x: x["score"], reverse=True)
        
        return scored_candidates
    
    def _calculate_context_score(self, question: Dict, context: Dict) -> float:
        """Calculate context match score (0-100)."""
        score = 0
        triggers = question.get("context_triggers", [])
        
        # Location match
        if context.get("location") and "location_detected" in triggers:
            score += 50
        
        # Category match
        if context.get("category") and context["category"] == question.get("category"):
            score += 30
        
        # Time range match
        if context.get("time_range"):
            time_keywords = ["monthly", "bulanan", "tren", "trend", "quarterly", "kuartal"]
            if any(kw in triggers for kw in time_keywords):
                score += 20
        
        return min(score, 100)
    
    def _calculate_sequential_score(self, question: Dict, history: List[Dict]) -> float:
        """Calculate sequential pattern score (0-100)."""
        if not history:
            return 0
        
        score = 0
        recent_categories = []
        
        for msg in history[-3:]:  # Last 3 messages
            content = msg.get("content", "").lower()
            # Simple category detection from content
            for cat in ["revenue", "customer", "product", "seller"]:
                if cat in content:
                    recent_categories.append(cat)
        
        # If question category matches recent conversation
        if question.get("category") in recent_categories:
            score += 60
        
        return min(score, 100)
    
    def _calculate_recency_score(self, question: Dict, history: List[Dict]) -> float:
        """Calculate recency score - higher if not recently asked (0-100)."""
        # Check if similar question was asked recently
        question_text = question.get("text", "").lower()
        
        for msg in history[-5:]:  # Last 5 messages
            if msg.get("role") == "user":
                user_text = msg.get("content", "").lower()
                # Simple similarity check
                if self._text_similarity(question_text, user_text) > 0.6:
                    return 0
        
        return 100
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity (Jaccard similarity)."""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def substitute_template_vars(self, question: Dict, context: Dict) -> str:
        """Substitute template variables in question text."""
        text = question.get("text", "")
        template_vars = question.get("template_vars", [])
        
        if not template_vars:
            return text
        
        # Substitute {location}
        if "location" in template_vars and context.get("location"):
            text = text.replace("{location}", context["location"])
        
        # Substitute {category}
        if "category" in template_vars and context.get("category"):
            text = text.replace("{category}", context["category"])
        
        # If template vars not substituted, return original or skip
        # If location var exists but no context, use generic placeholder
        if "{location}" in text:
            text = text.replace("{location}", "lokasi ini")
        
        return text
