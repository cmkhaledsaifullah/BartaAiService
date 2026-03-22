from app.constants import (
    SERVICE_NAME,
    SERVICE_ID,
    SERVICE_VERSION,
    COLLECTION_USERS,
    COLLECTION_NEWS_ARTICLES,
    DATE_FORMAT_YMD,
    HEALTH_STATUS_HEALTHY,
    ERROR_INVALID_TOKEN,
    ERROR_EMAIL_ALREADY_EXISTS,
    ERROR_UNKNOWN_LLM_PROVIDER,
    ERROR_UNKNOWN_EMBEDDING_PROVIDER,
    MSG_NO_ARTICLES_FOUND,
    MSG_NO_ARTICLES_BY_CATEGORY,
    MSG_INVALID_DATE_FORMAT,
    LLM_SYSTEM_PROMPT,
    AGENT_SYSTEM_PROMPT,
)


class TestServiceMetadata:
    def test_service_name(self):
        assert SERVICE_NAME == "Barta AI Service"

    def test_service_id(self):
        assert SERVICE_ID == "barta-ai-service"

    def test_service_version(self):
        assert SERVICE_VERSION == "1.0.0"


class TestCollectionNames:
    def test_users_collection(self):
        assert COLLECTION_USERS == "users"

    def test_news_articles_collection(self):
        assert COLLECTION_NEWS_ARTICLES == "news_articles"


class TestFormatConstants:
    def test_date_format(self):
        assert DATE_FORMAT_YMD == "%Y-%m-%d"

    def test_health_status(self):
        assert HEALTH_STATUS_HEALTHY == "healthy"


class TestErrorMessages:
    def test_error_messages_are_non_empty_strings(self):
        errors = [
            ERROR_INVALID_TOKEN,
            ERROR_EMAIL_ALREADY_EXISTS,
            MSG_NO_ARTICLES_FOUND,
            MSG_INVALID_DATE_FORMAT,
        ]
        for err in errors:
            assert isinstance(err, str)
            assert len(err) > 0

    def test_template_messages_have_placeholders(self):
        assert "{category_id}" in MSG_NO_ARTICLES_BY_CATEGORY
        assert "{provider}" in ERROR_UNKNOWN_LLM_PROVIDER
        assert "{supported}" in ERROR_UNKNOWN_LLM_PROVIDER
        assert "{provider}" in ERROR_UNKNOWN_EMBEDDING_PROVIDER


class TestSystemPrompts:
    def test_llm_prompt_non_empty(self):
        assert len(LLM_SYSTEM_PROMPT) > 100

    def test_agent_prompt_non_empty(self):
        assert len(AGENT_SYSTEM_PROMPT) > 100

    def test_llm_prompt_contains_key_rules(self):
        assert "Grounded" in LLM_SYSTEM_PROMPT
        assert "No Fabrication" in LLM_SYSTEM_PROMPT
        assert "Privacy" in LLM_SYSTEM_PROMPT

    def test_agent_prompt_contains_tool_references(self):
        assert "semantic_news_search" in AGENT_SYSTEM_PROMPT
        assert "get_latest_news" in AGENT_SYSTEM_PROMPT
