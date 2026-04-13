import concurrent.futures
import json
import os
import sys
from typing import Callable

from anthropic import Anthropic
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

load_dotenv(override=True)

JUDGE_MODEL = "o3-mini"
MERGE_MODEL = "gpt-4o-mini"
CLAUDE_MAX_TOKENS = 4096

Messages = list[dict[str, str]]

DESIGN_REVIEW_REQUEST = """For the following requirements and architectural diagram, please perform a full security design review which includes the following 7 steps
1. Define scope and system boundaries.
2. Create detailed data flow diagrams.
3. Apply threat frameworks (like STRIDE) to identify threats.
4. Rate and prioritize identified threats.
5. Document-specific security controls and mitigations.
6. Rank the threats based on their severity and likelihood of occurrence.
7. Provide a summary of the security review and recommendations.

Here are the requirements and mermaid architectural diagram:
Software Requirements Specification (SRS) - Juice Shop: Secure E-Commerce Platform
This document outlines the functional and non-functional requirements for the Juice Shop, a secure online retail platform.

1. Introduction

1.1 Purpose: To define the requirements for a robust and secure e-commerce platform that allows customers to purchase products online safely and efficiently.
1.2 Scope: The system will be a web-based application providing a full range of e-commerce functionalities, from user registration and product browsing to secure payment processing and order management.
1.3 Intended Audience: This document is intended for project managers, developers, quality assurance engineers, and stakeholders involved in the development and maintenance of the Juice Shop platform.
2. Overall Description

2.1 Product Perspective: A customer-facing, scalable, and secure e-commerce website with a comprehensive administrative backend.
2.2 Product Features:
Secure user registration and authentication with multi-factor authentication (MFA).
A product catalog with detailed descriptions, images, pricing, and stock levels.
Advanced search and filtering capabilities for products.
A secure shopping cart and checkout process integrating with a trusted payment gateway.
User profile management, including order history, shipping addresses, and payment information.
An administrative dashboard for managing products, inventory, orders, and customer data.
2.3 User Classes and Characteristics:
Customer: A registered or guest user who can browse products, make purchases, and manage their account.
Administrator: An authorized employee who can manage the platform's content and operations.
Customer Service Representative: An authorized employee who can assist customers with orders and account issues.
3. System Features

3.1 Functional Requirements:
User Management:
Users shall be able to register for a new account with a unique email address and a strong password.
The system shall enforce strong password policies (e.g., length, complexity, and expiration).
Users shall be able to log in securely and enable/disable MFA.
Users shall be able to reset their password through a secure, token-based process.
Product Management:
The system shall display products with accurate information, including price, description, and availability.
Administrators shall be able to add, update, and remove products from the catalog.
Order Processing:
The system shall process orders through a secure, PCI-compliant payment gateway.
The system shall encrypt all sensitive customer and payment data.
Customers shall receive email confirmations for orders and shipping updates.
3.2 Non-Functional Requirements:
Security:
All data transmission shall be encrypted using TLS 1.2 or higher.
The system shall be protected against common web vulnerabilities, including the OWASP Top 10 (e.g., SQL Injection, XSS, CSRF).
Regular security audits and penetration testing shall be conducted.
Performance:
The website shall load in under 3 seconds on a standard broadband connection.
The system shall handle at least 1,000 concurrent users without significant performance degradation.
Reliability: The system shall have an uptime of 99.9% or higher.
Usability: The user interface shall be intuitive and easy to navigate for all user types.

and here is the mermaid architectural diagram:

graph TB
    subgraph "Client Layer"
        Browser[Web Browser]
        Mobile[Mobile App]
    end

    subgraph "Frontend Layer"
        Angular[Angular SPA Frontend]
        Static[Static Assets<br/>CSS, JS, Images]
    end

    subgraph "Application Layer"
        Express[Express.js Server]
        Routes[REST API Routes]
        Auth[Authentication Module]
        Middleware[Security Middleware]
        Challenges[Challenge Engine]
    end

    subgraph "Business Logic"
        UserMgmt[User Management]
        ProductCatalog[Product Catalog]
        OrderSystem[Order System]
        Feedback[Feedback System]
        FileUpload[File Upload Handler]
        Payment[Payment Processing]
    end

    subgraph "Data Layer"
        SQLite[(SQLite Database)]
        FileSystem[File System<br/>Uploaded Files]
        Memory[In-Memory Storage<br/>Sessions, Cache]
    end

    subgraph "Security Features (Intentionally Vulnerable)"
        XSS[DOM Manipulation]
        SQLi[Database Queries]
        AuthBypass[Login System]
        CSRF[State Changes]
        CryptoVuln[Password Hashing]
        IDOR[Resource Access]
    end

    subgraph "External Dependencies"
        NPM[NPM Packages]
        JWT[JWT Libraries]
        CryptoLib[Crypto Libraries]
        Sequelize[Sequelize ORM]
    end

    %% Client connections
    Browser --> Angular
    Mobile --> Routes

    %% Frontend connections
    Angular --> Static
    Angular --> Routes

    %% Application layer connections
    Express --> Routes
    Routes --> Auth
    Routes --> Middleware
    Routes --> Challenges

    %% Business logic connections
    Routes --> UserMgmt
    Routes --> ProductCatalog
    Routes --> OrderSystem
    Routes --> Feedback
    Routes --> FileUpload
    Routes --> Payment

    %% Data layer connections
    UserMgmt --> SQLite
    ProductCatalog --> SQLite
    OrderSystem --> SQLite
    Feedback --> SQLite
    FileUpload --> FileSystem
    Auth --> Memory

    %% Security vulnerabilities (dotted lines indicate vulnerable paths)
    Angular -.-> XSS
    Routes -.-> SQLi
    Auth -.-> AuthBypass
    Angular -.-> CSRF
    UserMgmt -.-> CryptoVuln
    Routes -.-> IDOR

    %% External dependencies
    Express --> NPM
    Auth --> JWT
    UserMgmt --> CryptoLib
    SQLite --> Sequelize"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def validate_keys() -> None:
    keys = [
        ("OpenAI", os.getenv("OPENAI_API_KEY"), 8, True),
        ("Anthropic", os.getenv("ANTHROPIC_API_KEY"), 7, False),
        ("Google", os.getenv("GOOGLE_API_KEY"), 2, False),
        ("DeepSeek", os.getenv("DEEPSEEK_API_KEY"), 3, False),
        ("Groq", os.getenv("GROQ_API_KEY"), 4, False),
    ]
    for name, key, prefix, required in keys:
        if key:
            print(f"{name} API Key exists and begins {key[:prefix]}")
        elif required:
            sys.exit(f"{name} API Key is required but not set")
        else:
            print(f"{name} API Key not set (optional)")


def call_openai_compat(client: OpenAI, model: str, msgs: Messages) -> str:
    return client.chat.completions.create(model=model, messages=msgs).choices[0].message.content or ""


def call_claude(client: Anthropic, model: str, msgs: Messages) -> str:
    return client.messages.create(model=model, messages=msgs, max_tokens=CLAUDE_MAX_TOKENS).content[0].text


def query_competitor(name: str, fn, msgs: Messages) -> tuple[str, str]:
    try:
        answer = fn(msgs)
        print(f"  [{name}] responded.")
        return name, answer
    except Exception as e:
        print(f"  [{name}] failed: {e}")
        return name, f"[ERROR: {e}]"


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def collect_reviews(competitors: list[tuple[str, Callable]]) -> list[tuple[str, str]]:
    """Stage 1 — Parallelization: fan-out the security review request to all models concurrently."""
    print("\n--- Stage 1: Collecting security design reviews (parallel) ---")
    msgs: Messages = [{"role": "user", "content": DESIGN_REVIEW_REQUEST}]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(query_competitor, name, fn, msgs): name for name, fn in competitors}
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    order = {name: i for i, (name, _) in enumerate(competitors)}
    results.sort(key=lambda r: order[r[0]])
    return results


def judge_reviews(openai_client: OpenAI, results: list[tuple[str, str]]) -> None:
    """Stage 2 — Prompt Chaining: judge ranks all security reviews."""
    print("\n--- Stage 2: Judging security reviews ---")
    combined = "\n\n".join(
        f"# Response from competitor {i + 1}\n\n{answer}"
        for i, (_, answer) in enumerate(results)
    )
    judge_prompt = f"""You are judging a competition between {len(results)} competitors.
Each model has been given this question:

{DESIGN_REVIEW_REQUEST}

Your job is to evaluate each response for completeness and accuracy, and rank them in order of best to worst.
Respond with JSON, and only JSON, with the following format:
{{"results": ["best competitor number", "second best competitor number", "third best competitor number", ...]}}

Here are the responses from each competitor:

{combined}

Now respond with the JSON with the ranked order of the competitors, nothing else. Do not include markdown formatting or code blocks."""

    raw = call_openai_compat(openai_client, JUDGE_MODEL, [{"role": "user", "content": judge_prompt}])
    ranks = json.loads(raw)["results"]

    print("\n--- Final Rankings ---")
    competitor_names = [name for name, _ in results]
    for rank, result in enumerate(ranks, start=1):
        print(f"Rank {rank}: {competitor_names[int(result) - 1]}")


def merge_reviews(openai_client: OpenAI, results: list[tuple[str, str]]) -> str:
    """Stage 3 — Prompt Chaining: synthesize all reviews into a single comprehensive report."""
    print("\n--- Stage 3: Merging reviews into a final report ---")
    combined = "\n\n".join(
        f"# Response from competitor {i + 1}\n\n{answer}"
        for i, (_, answer) in enumerate(results)
    )
    merge_prompt = f"""Here are design reviews from {len(results)} LLMs:

{combined}

Your task is to synthesize these reviews into a single, comprehensive design review and threat model that:

1. **Includes all identified threats**, consolidating any duplicates with unified wording.
2. **Preserves the strongest insights** from each review, especially nuanced or unique observations.
3. **Highlights conflicting or divergent findings**, if any, and explains which interpretation seems more likely and why.
4. **Organizes the final output** in a clear format, with these sections:
   - Scope and System Boundaries
   - Data Flow Overview
   - Identified Threats (categorized using STRIDE or equivalent)
   - Risk Ratings and Prioritization
   - Suggested Mitigations
   - Final Comments and Open Questions

Be concise but thorough. Treat this as a final report for a real-world security audit."""

    report = call_openai_compat(openai_client, MERGE_MODEL, [{"role": "user", "content": merge_prompt}])
    print("\n--- Merged Security Design Review ---")
    print(report)
    return report


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    validate_keys()

    openai_client = OpenAI()
    claude_client = Anthropic()
    gemini_client = OpenAI(
        api_key=os.getenv("GOOGLE_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    deepseek_client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1",
    )
    groq_client = OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
    )
    ollama_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

    competitors: list[tuple[str, Callable]] = [
        ("gpt-4o-mini", lambda msgs: call_openai_compat(openai_client, "gpt-4o-mini", msgs)),
        ("claude-sonnet-4-5", lambda msgs: call_claude(claude_client, "claude-sonnet-4-5", msgs)),
        ("gemini-2.0-flash", lambda msgs: call_openai_compat(gemini_client, "gemini-2.0-flash", msgs)),
        ("deepseek-chat", lambda msgs: call_openai_compat(deepseek_client, "deepseek-chat", msgs)),
        ("llama-3.3-70b-versatile", lambda msgs: call_openai_compat(groq_client, "llama-3.3-70b-versatile", msgs)),
        ("llama3.2", lambda msgs: call_openai_compat(ollama_client, "llama3.2", msgs)),
    ]

    try:
        results = collect_reviews(competitors)
        judge_reviews(openai_client, results)
        merge_reviews(openai_client, results)
    except OpenAIError as exc:
        sys.exit(f"OpenAI API error: {exc}")


if __name__ == "__main__":
    main()
