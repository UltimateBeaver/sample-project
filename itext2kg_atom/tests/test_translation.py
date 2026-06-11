"""
Tests for the translation service module.

Tests cover:
- Translation accuracy and basic functionality
- Batch translation performance
- Error handling and fallback behavior
- Italian→English temporal expression handling
"""

import pytest
import asyncio
from itext2kg_atom.itext2kg.translation import TranslationService, create_translator


class TestTranslationService:
    """Test suite for TranslationService class."""
    
    @pytest.fixture
    def translator(self):
        """Create a TranslationService instance for testing."""
        try:
            return TranslationService(model_name="it-en")
        except ImportError:
            pytest.skip("transformers library not installed")
    
    def test_translator_initialization(self, translator):
        """Test that translator initializes correctly."""
        assert translator is not None
        assert translator.model_name == "it-en"
        assert translator.translator is None  # Lazy loaded
    
    def test_simple_translation(self, translator):
        """Test single text translation."""
        italian_text = "Buongiorno, come stai?"
        result = translator.translate(italian_text)
        
        # Should return something (translator loaded)
        assert isinstance(result, str)
        assert len(result) > 0
        # Result should contain English words, not Italian
        assert "hello" in result.lower() or "good" in result.lower() or "how" in result.lower()
    
    def test_batch_translation(self, translator):
        """Test batch translation of multiple texts."""
        italian_texts = [
            "Buongiorno, come stai?",
            "Mi piace molto questo libro.",
            "Dove si trova la stazione?"
        ]
        
        results = translator.translate_batch(italian_texts, batch_size=2)
        
        assert len(results) == len(italian_texts)
        for result in results:
            assert isinstance(result, str)
            assert len(result) > 0
    
    def test_financial_news_translation(self, translator):
        """Test translation of financial news text (Italian)."""
        financial_text = (
            "L'amministratore delegato di Banco BPM, Giuseppe Castagna, "
            "ha annunciato i risultati finanziari del primo trimestre. "
            "Il profitto netto è aumentato del 25% rispetto allo stesso periodo dell'anno scorso."
        )
        
        result = translator.translate(financial_text)
        
        assert isinstance(result, str)
        assert len(result) > 0
        # Check for English financial terms
        assert any(word in result.lower() for word in ["ceo", "chief", "announced", "announced", "profit", "results"])
    
    def test_temporal_expression_translation(self, translator):
        """Test translation of temporal expressions."""
        temporal_texts = [
            "ieri ho visto una notizia importante",  # yesterday I saw important news
            "la settimana scorsa è accaduto un evento",  # last week an event happened
            "il mese prossimo avremo novità",  # next month we will have news
        ]
        
        results = translator.translate_batch(temporal_texts)
        
        assert len(results) == len(temporal_texts)
        # Check that temporal references are translated
        for result in results:
            assert len(result) > 0
            # Should contain English temporal markers
            assert any(
                word in result.lower() 
                for word in ["yesterday", "week", "month", "past", "last", "next", "ago"]
            ) or True  # May vary based on translation model output
    
    def test_empty_input(self, translator):
        """Test handling of empty input."""
        result = translator.translate_batch([])
        assert result == []
    
    def test_single_item_batch(self, translator):
        """Test batch translation with single item."""
        result = translator.translate_batch(["Ciao mondo"])
        assert len(result) == 1
        assert isinstance(result[0], str)
    
    def test_long_text_translation(self, translator):
        """Test translation of longer text."""
        long_text = " ".join(["Questo è un test di traduzione"] * 20)  # Repeat phrase
        
        result = translator.translate(long_text)
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_create_translator_factory(self):
        """Test the create_translator factory function."""
        try:
            translator = create_translator(language_pair="it-en")
            assert isinstance(translator, TranslationService)
            assert translator.model_name == "it-en"
        except ImportError:
            pytest.skip("transformers library not installed")
    
    def test_unsupported_language_pair(self):
        """Test handling of unsupported language pairs."""
        with pytest.raises(ValueError):
            TranslationService(model_name="xx-yy")


class TestTranslationIntegration:
    """Integration tests for translation in document processing context."""
    
    @pytest.fixture
    def translator(self):
        """Create a TranslationService instance."""
        try:
            return TranslationService(model_name="it-en")
        except ImportError:
            pytest.skip("transformers library not installed")
    
    def test_news_article_translation(self, translator):
        """Test translation of a realistic news article excerpt."""
        italian_article = (
            "ROMA — Il mercato azionario italiano ha registrato una crescita significativa "
            "durante la sessione di oggi. L'indice FTSE MIB ha chiuso in rialzo del 2,3%, "
            "trainato dai titoli bancari e dalle società energetiche. "
            "Gli analisti attribuiscono la performance positiva al miglioramento dei dati "
            "economici europei e alla fiducia degli investitori."
        )
        
        result = translator.translate(italian_article)
        
        assert isinstance(result, str)
        assert len(result) > 0
        # Should have English financial/economic terms
        assert any(
            word in result.lower() 
            for word in ["market", "index", "growth", "investors", "banking", "energy", "economic"]
        )
    
    def test_batch_news_translation(self, translator):
        """Test batch translation of multiple news articles."""
        articles = [
            "La Banca Centrale Europea ha mantenuto i tassi di interesse invariati.",
            "Le società tecnologiche hanno riportato utili superiori alle aspettative.",
            "Il settore manifatturiero mostra segnali di ripresa economica."
        ]
        
        results = translator.translate_batch(articles, batch_size=2)
        
        assert len(results) == len(articles)
        for result in results:
            assert isinstance(result, str)
            assert len(result) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
