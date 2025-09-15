# Alison - AI Classroom IT Support Assistant

Alison is an AI-powered IT support assistant designed to help teachers and students in the classroom with common technical issues.

## How it works

Alison is built using LangGraph and a RAG pipeline. When a user asks an IT-related question, the query is routed to the Alison agent. The agent then performs the following steps:

1.  **Identifies the problem:** The agent uses a language model to identify the core problem from the user's query.
2.  **Searches the knowledge base:** The agent searches a dedicated knowledge base of IT support documents to find relevant solutions.
3.  **Generates a response:** The agent uses the retrieved information to generate a helpful and easy-to-understand response for the user.
4.  **Escalates to a human:** If the agent is unable to resolve the issue, it will escalate the problem to a human IT support technician.

## Knowledge Base

Alison's knowledge base is stored in the `surfsense_backend/app/alison_docs/` directory. The documents are in Markdown format and are indexed on application startup. To add new knowledge to Alison, simply create a new Markdown file in this directory.

The agent also uses a `classroom_faq.json` file for quick lookups of common questions.

## Testing

To run the tests for the Alison agent, use the following command:

```bash
pytest surfsense_backend/tests/agents/alison/
```
