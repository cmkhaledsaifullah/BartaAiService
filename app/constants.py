"""
Centralized string constants for the Barta AI Service.

All user-facing messages, error strings, collection names, service metadata,
and system prompts are defined here. This makes future localization
(i18n/l10n) straightforward — swap this file or load from a resource bundle.
"""

# ==========================================================================
# Service Metadata
# ==========================================================================

SERVICE_NAME = "Barta AI Service"
SERVICE_ID = "barta-ai-service"
SERVICE_DESCRIPTION = "Agentic RAG AI backend for Bangladesh news chat"
SERVICE_VERSION = "1.0.0"
DOCS_DISABLED_MESSAGE = "Docs disabled in production"

# ==========================================================================
# MongoDB Collections
# ==========================================================================

COLLECTION_USERS = "users"
COLLECTION_NEWS_ARTICLES = "news_articles"

# ==========================================================================
# Date Formats
# ==========================================================================

DATE_FORMAT_YMD = "%Y-%m-%d"

# ==========================================================================
# Auth Error Messages
# ==========================================================================

ERROR_INVALID_TOKEN = "Invalid or expired authentication token"
ERROR_TOKEN_EXPIRED = "Token has expired"
ERROR_USER_DEACTIVATED = "User account is deactivated"
ERROR_EMAIL_ALREADY_EXISTS = "An account with this email already exists"
ERROR_USERNAME_ALREADY_TAKEN = "This username is already taken"
ERROR_INVALID_CREDENTIALS = "Incorrect email or password"

# ==========================================================================
# Validation Error Messages
# ==========================================================================

ERROR_INVALID_EMAIL_FORMAT = "Invalid email format"
ERROR_PASSWORD_NO_UPPERCASE = "Password must contain at least one uppercase letter"
ERROR_PASSWORD_NO_LOWERCASE = "Password must contain at least one lowercase letter"
ERROR_PASSWORD_NO_DIGIT = "Password must contain at least one digit"

# ==========================================================================
# Server Error Messages
# ==========================================================================

ERROR_INTERNAL_SERVER = "An internal error occurred. Please try again later."
ERROR_MONGODB_NOT_CONNECTED = "MongoDB is not connected. Call connect_to_mongodb() first."

# ==========================================================================
# Agent / Tool Response Messages
# ==========================================================================

MSG_AGENT_PROCESSING_FAILED = "I was unable to process your request."
MSG_NO_ARTICLES_FOUND = "No relevant news articles found for this query."
MSG_NO_ARTICLES_BY_CATEGORY = "No articles found for category: {category_id}"
MSG_NO_ARTICLES_BY_DATE_RANGE = "No articles found between {start_date} and {end_date}."
MSG_NO_ARTICLES_BY_TAGS = "No articles found with tags: {tags}"
MSG_NO_ARTICLES_BY_NEWSPAPER = "No articles found from newspaper: {newspaper_id}"
MSG_NO_RECENT_ARTICLES = "No recent articles found."
MSG_NO_TAGS_PROVIDED = "No tags provided."
MSG_INVALID_DATE_FORMAT = "Invalid date format. Please use YYYY-MM-DD."

# ==========================================================================
# Provider Error Messages
# ==========================================================================

ERROR_UNKNOWN_LLM_PROVIDER = (
    "Unknown LLM provider: '{provider}'. Supported: {supported}"
)
ERROR_UNKNOWN_EMBEDDING_PROVIDER = (
    "Unknown embedding provider: '{provider}'. Supported: {supported}"
)

# ==========================================================================
# System Prompts
# ==========================================================================

LLM_SYSTEM_PROMPT = """You are Barta AI, an intelligent news assistant specializing in Bangladesh news.
You help users find, understand, and analyze news articles from Bangladesh.

IMPORTANT RULES YOU MUST FOLLOW:
1. **Grounded Responses**: Only provide information that is supported by the retrieved news articles.
   If the retrieved context does not contain enough information, say so honestly.
2. **Citations**: Always cite the source articles when providing information. Include the article
   title and source URL when available.
3. **No Fabrication**: Never fabricate news, statistics, quotes, or events. If you don't know, say so.
4. **Privacy**: Never reveal personal information about private individuals mentioned in articles
   beyond what is publicly reported. Do not speculate about individuals' private lives.
5. **Neutrality**: Present news objectively. Do not inject personal opinions or political bias.
   When covering sensitive topics, present multiple perspectives if available.
6. **Safety**: Refuse to generate content that promotes violence, hate speech, discrimination,
   or illegal activities. If asked to do so, politely decline.
7. **Language**: Respond in the same language the user uses. Support both Bangla and English.
8. **Date Awareness**: When discussing news, always note the publish date so users understand
   the recency of the information.
9. **Scope**: You are a news assistant. Politely redirect off-topic questions back to
   news-related discussions.
10. **Transparency**: If asked about your capabilities or limitations, be honest. You are an AI
    that retrieves and summarizes news — you do not have real-time access to news beyond what
    is in your database.

When presenting news information, structure your response clearly with relevant details,
dates, and sources."""

AGENT_SYSTEM_PROMPT = """You are Barta AI, an intelligent news assistant specializing in Bangladesh news.

Your job is to help users find, understand, and analyze news from Bangladesh by using
the available tools to search the news database.

## Decision Making Process

1. **Understand the Query**: Determine what the user is asking about.
2. **Choose Tools**: Select the most appropriate tool(s):
   - For general topic queries → use `semantic_news_search`
   - For category-specific requests → use `search_news_by_category`
   - For date-specific queries → use `search_news_by_date_range`
   - For specific topic/entity searches → use `search_news_by_tags`
   - For newspaper-specific requests → use `search_news_by_newspaper`
   - For "what's new" or "latest" queries → use `get_latest_news`
3. **Synthesize**: After retrieval, synthesize a clear, well-organized response.

## Response Guidelines

- ALWAYS cite sources with article title and URL when available.
- NEVER fabricate information. If tools return no results, say so honestly.
- Present information neutrally and objectively.
- Note publication dates so users understand recency.
- Respond in the same language the user uses (Bangla or English).
- If the query is off-topic (not news-related), politely redirect.
- For ambiguous queries, use semantic search first, then refine if needed.
- You may call multiple tools if the query requires combining different searches.
"""

# ==========================================================================
# Health Check
# ==========================================================================

HEALTH_STATUS_HEALTHY = "healthy"
