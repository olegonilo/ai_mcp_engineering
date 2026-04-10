"""
Personal AI agent for Oleh Onilo's website.

Agentic patterns used:
  1. Tool-use loop      – iterates until the model stops calling tools
  2. RAG retrieval      – semantic search over personal knowledge chunks
  3. SQL knowledge base – 100 Q&A pairs for Automation QA topics (read + write)
  4. Pushover tools     – record contacts and unknown questions
  5. Evaluator + rerun  – quality-control loop that regenerates low-quality replies
"""

import hashlib
import json
import math
import os
import sqlite3
import sys
from pathlib import Path

import gradio as gr
import requests
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from pydantic import BaseModel
from pypdf import PdfReader

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
ME_DIR = ROOT / "me"
LINKEDIN_PDF = ME_DIR / "linkedin.pdf"
SUMMARY_PATH = ME_DIR / "summary.txt"
DB_PATH = ME_DIR / "qa_knowledge.db"
EMBED_CACHE = ME_DIR / "embed_cache.json"

NAME = "Oleh Onilo"
CHAT_MODEL = "gpt-4o-mini"
EVALUATOR_MODEL = "gpt-4.1-nano"
EMBED_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 400   # characters per chunk
CHUNK_OVERLAP = 80
TOP_K_CHUNKS = 4
MAX_TOOL_ITERATIONS = 10

# ---------------------------------------------------------------------------
# Pushover notifications
# ---------------------------------------------------------------------------


def push(text: str) -> None:
    token = os.getenv("PUSHOVER_TOKEN")
    user = os.getenv("PUSHOVER_USER")
    if not token or not user:
        print("[Pushover] Missing credentials — notification skipped.", flush=True)
        return
    try:
        r = requests.post(
            "https://api.pushover.net/1/messages.json",
            data={"token": token, "user": user, "message": text},
            timeout=10,
        )
        r.raise_for_status()
    except requests.RequestException as exc:
        print(f"[Pushover] Notification failed: {exc}", flush=True)


def record_user_details(
    email: str,
    name: str = "Name not provided",
    notes: str = "not provided",
) -> dict:
    push(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}


def record_unknown_question(question: str) -> dict:
    push(f"Recording unknown question: {question}")
    return {"recorded": "ok"}


# ---------------------------------------------------------------------------
# RAG – text chunking, embeddings, cosine search
# ---------------------------------------------------------------------------


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunk = text[start : start + size].strip()
        if chunk:
            chunks.append(chunk)
        start += size - overlap
    return chunks


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return 0.0 if norm_a == 0 or norm_b == 0 else dot / (norm_a * norm_b)


class VectorStore:
    def __init__(self) -> None:
        self.chunks: list[str] = []
        self.embeddings: list[list[float]] = []

    def add(self, chunks: list[str], embeddings: list[list[float]]) -> None:
        self.chunks.extend(chunks)
        self.embeddings.extend(embeddings)

    def search(self, query_embedding: list[float], top_k: int = TOP_K_CHUNKS) -> list[str]:
        scored = [
            (cosine_similarity(query_embedding, emb), chunk)
            for emb, chunk in zip(self.embeddings, self.chunks)
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored[:top_k]]


def _get_embeddings(client: OpenAI, texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in response.data]


def _docs_fingerprint(docs: list[str]) -> str:
    return hashlib.sha256("\n".join(docs).encode()).hexdigest()


def build_vector_store(client: OpenAI, documents: list[str]) -> VectorStore:
    fingerprint = _docs_fingerprint(documents)

    if EMBED_CACHE.is_file():
        cached = json.loads(EMBED_CACHE.read_text(encoding="utf-8"))
        if cached.get("fingerprint") == fingerprint:
            store = VectorStore()
            store.add(cached["chunks"], cached["embeddings"])
            print(f"[RAG] Loaded {len(store.chunks)} chunks from cache.", flush=True)
            return store

    all_chunks: list[str] = []
    for doc in documents:
        all_chunks.extend(chunk_text(doc))

    if not all_chunks:
        print("[RAG] No chunks to embed — returning empty store.", flush=True)
        return VectorStore()

    all_embeddings: list[list[float]] = []
    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        all_embeddings.extend(_get_embeddings(client, all_chunks[i : i + batch_size]))

    store = VectorStore()
    store.add(all_chunks, all_embeddings)
    print(f"[RAG] Built vector store with {len(store.chunks)} chunks.", flush=True)

    EMBED_CACHE.write_text(
        json.dumps({"fingerprint": fingerprint, "chunks": all_chunks, "embeddings": all_embeddings}),
        encoding="utf-8",
    )
    return store


# ---------------------------------------------------------------------------
# SQL Q&A knowledge base – 100 Automation QA pairs
# ---------------------------------------------------------------------------

_QA_SEED: list[tuple[str, str, str]] = [
    # (question, answer, topic)
    # --- Fundamentals ---
    ("What is test automation?",
     "Test automation uses software tools to execute pre-scripted tests automatically, comparing actual outcomes with expected ones. It reduces manual effort, increases test coverage, enables regression testing at scale, and speeds up feedback cycles in CI/CD pipelines.",
     "fundamentals"),
    ("What is the difference between functional and non-functional testing?",
     "Functional testing verifies that features work as specified (e.g., login, checkout). Non-functional testing covers performance, security, usability, and reliability — aspects beyond feature correctness.",
     "fundamentals"),
    ("What is regression testing?",
     "Regression testing re-runs existing tests after code changes to ensure nothing that previously worked is broken. Automation makes regression suites cost-effective by eliminating manual re-execution effort.",
     "fundamentals"),
    ("What is smoke testing?",
     "Smoke testing is a shallow pass covering core functionality to quickly determine whether a build is stable enough for further testing. It acts as a sanity check before running the full suite.",
     "fundamentals"),
    ("What is the difference between unit, integration, and end-to-end tests?",
     "Unit tests verify individual functions in isolation. Integration tests check that components interact correctly. E2E tests simulate real user flows through the full system stack.",
     "fundamentals"),
    ("What is the test pyramid?",
     "The test pyramid (Mike Cohn) recommends many fast unit tests at the base, fewer integration tests in the middle, and a small number of slow E2E tests at the top. It optimizes for speed and cost of feedback.",
     "fundamentals"),
    ("What is a flaky test and how do you handle it?",
     "A flaky test passes and fails intermittently without code changes — usually due to timing, shared state, or external dependencies. Fix by adding proper waits, isolating state, using mocks for external services, and tracking flakiness metrics.",
     "fundamentals"),
    ("What are the key qualities of a good automated test?",
     "Good automated tests are: deterministic, independent, readable, fast, maintainable, and provide clear failure messages. They follow the AAA pattern: Arrange, Act, Assert.",
     "fundamentals"),
    ("What is test coverage and why does it matter?",
     "Test coverage measures how much of the codebase is exercised by tests (line, branch, path, mutation). Higher coverage reduces the risk of undetected defects, though 100% coverage doesn't guarantee bug-free software.",
     "fundamentals"),
    ("What is the difference between verification and validation?",
     "Verification checks that the product is built correctly (does it match specifications?). Validation checks that the correct product is built (does it meet user needs?). Both are essential in a quality process.",
     "fundamentals"),

    # --- Selenium ---
    ("What is Selenium WebDriver?",
     "Selenium WebDriver is an open-source API for browser automation. It sends commands directly to the browser via ChromeDriver/GeckoDriver and enables programmatic control of web applications across browsers.",
     "selenium"),
    ("What is the Page Object Model (POM)?",
     "POM is a design pattern where each web page is a class containing element locators and interaction methods. It separates test logic from UI structure, making tests more maintainable and reducing duplication.",
     "selenium"),
    ("What is the difference between implicit and explicit waits in Selenium?",
     "Implicit wait sets a global timeout for all find_element calls. Explicit wait (WebDriverWait + ExpectedConditions) waits for a specific condition on a specific element — more precise and reliable.",
     "selenium"),
    ("What are the different locator strategies in Selenium?",
     "Selenium supports: ID, name, class name, tag name, link text, partial link text, CSS selector, and XPath. CSS selectors and XPath are the most powerful for complex element identification.",
     "selenium"),
    ("What is XPath and when should you use it?",
     "XPath is a query language for navigating XML/HTML. Use it when elements lack unique IDs/classes or when you need DOM traversal (e.g., 'button inside the third table row'). Prefer CSS selectors for simplicity.",
     "selenium"),
    ("How do you handle dynamic elements in Selenium?",
     "Use explicit waits with conditions like element_to_be_clickable. For dynamic IDs, use partial attribute matching in CSS (e.g., [id^='prefix']) or XPath contains(). Avoid hardcoded sleeps.",
     "selenium"),
    ("How do you handle iframes in Selenium?",
     "Switch with driver.switch_to.frame(frame). Interact with elements inside, then return with driver.switch_to.default_content(). Always switch back to avoid context errors.",
     "selenium"),
    ("What is Selenium Grid?",
     "Selenium Grid enables distributed test execution across multiple machines and browsers in parallel. A Hub distributes commands to registered Nodes, each running a specific browser/OS combination.",
     "selenium"),
    ("How do you take a screenshot in Selenium?",
     "Use driver.save_screenshot('path.png') or element.screenshot('element.png'). Screenshots are invaluable for debugging failures in CI pipelines.",
     "selenium"),
    ("What are common challenges with Selenium automation?",
     "Flaky tests from timing issues, maintaining selectors when UI changes, dynamic content, browser-specific behavior, slow E2E execution. POM pattern and proper waits address most of these.",
     "selenium"),

    # --- API Testing ---
    ("What is API testing and why is it important?",
     "API testing validates that APIs work correctly, reliably, and securely. It's faster and more stable than UI testing, catches integration issues early, and is essential in microservice architectures.",
     "api_testing"),
    ("What is the difference between REST and SOAP APIs?",
     "REST uses HTTP verbs and JSON/XML, is stateless and lightweight. SOAP is a protocol with XML messaging and formal WSDL contracts. REST is simpler and dominant in modern APIs.",
     "api_testing"),
    ("What HTTP status codes should QA engineers know?",
     "200 OK, 201 Created, 204 No Content, 400 Bad Request, 401 Unauthorized, 403 Forbidden, 404 Not Found, 409 Conflict, 422 Unprocessable Entity, 500 Internal Server Error, 503 Service Unavailable.",
     "api_testing"),
    ("What tools are used for API testing?",
     "Postman (manual and automated), pytest + requests/httpx (Python), RestAssured (Java), Karate DSL, Playwright API testing, k6 (performance), and Newman (Postman CLI for CI integration).",
     "api_testing"),
    ("What is contract testing?",
     "Contract testing verifies that services adhere to agreed-upon API contracts. Tools like Pact record consumer expectations and verify providers meet them, catching integration breaks early.",
     "api_testing"),
    ("How do you test authentication in APIs?",
     "Test valid and invalid credentials. Verify protected endpoints return 401 without a token, 403 with insufficient permissions. Check token expiry, refresh flows, and that sensitive endpoints reject forged tokens.",
     "api_testing"),
    ("What is idempotency and how do you test it?",
     "An idempotent operation produces the same result regardless of how many times it's called (GET, PUT, DELETE). Test by calling the operation multiple times and verifying the final state is identical.",
     "api_testing"),
    ("What is the difference between a mock and a stub?",
     "A stub returns hardcoded responses for specific requests. A mock additionally verifies how it was called (assertions on interactions). Both isolate the system from real dependencies for faster, deterministic tests.",
     "api_testing"),
    ("How do you test API performance?",
     "Use k6, Locust, or JMeter to simulate load and measure: response time (p50, p95, p99), throughput (RPS), error rate, and resource consumption. Establish baselines and set performance budgets in CI.",
     "api_testing"),
    ("What should you verify in an API test?",
     "Status code, response body structure and values, headers (Content-Type, CORS), response time, error messages for invalid inputs, data persistence (GET after POST/PUT), and security (authorization, injection prevention).",
     "api_testing"),

    # --- CI/CD ---
    ("What is CI/CD and how does it relate to test automation?",
     "CI automatically builds and tests code on every commit. CD automates releases. Test automation is the engine of CI/CD — without automated tests, pipelines can't safely validate changes.",
     "ci_cd"),
    ("What CI/CD tools do you have experience with?",
     "Jenkins, GitHub Actions, GitLab CI, CircleCI, Azure DevOps, Buildkite, TeamCity. Key concepts: triggers, stages, jobs, artifacts, caching, and environment secrets.",
     "ci_cd"),
    ("How do you integrate automated tests into a CI pipeline?",
     "Define test stages in pipeline YAML (unit → integration → E2E). Run on PRs to block merges on failures. Use JUnit XML for reporting, collect coverage artifacts, and send notifications on failures.",
     "ci_cd"),
    ("What is the difference between Continuous Delivery and Continuous Deployment?",
     "Continuous Delivery means code is always ready to deploy but requires a manual approval step. Continuous Deployment automatically deploys every passing build with no human gate.",
     "ci_cd"),
    ("How do you manage test environments in CI/CD?",
     "Use environment-specific config or environment variables. Docker Compose or Kubernetes spins up isolated environments per test run. Infrastructure-as-Code ensures reproducibility. Tear down after tests to save costs.",
     "ci_cd"),
    ("What are quality gates in a CI pipeline?",
     "Quality gates are pass/fail thresholds that block pipeline progression: minimum coverage, maximum failure rate, performance benchmarks, security scan pass. They enforce standards automatically.",
     "ci_cd"),
    ("How do you handle slow test suites in CI?",
     "Run tests in parallel (pytest-xdist, Selenium Grid), split by type (unit first), cache dependencies, use test impact analysis to run only affected tests, and optimize the slowest E2E tests.",
     "ci_cd"),
    ("What is shift-left testing?",
     "Shift-left moves testing earlier: writing tests alongside code, running unit tests on pre-commit hooks, static analysis in the IDE, and security scanning in PRs. It reduces the cost of finding defects.",
     "ci_cd"),
    ("What is test parallelization?",
     "Parallelization runs tests concurrently to reduce total execution time. In pytest: use pytest-xdist (-n auto). Ensure tests are independent with no shared mutable state between workers.",
     "ci_cd"),
    ("How do you report test results in CI?",
     "Generate JUnit XML (pytest --junitxml), HTML reports (pytest-html), Allure for rich dashboards. Integrate with Xray or TestRail. Slack/email notifications for critical failures. Track trends over time.",
     "ci_cd"),

    # --- Python Testing ---
    ("What is pytest and what are its advantages?",
     "pytest is Python's most popular testing framework. Advantages: simple assert syntax, powerful fixtures, parametrize for data-driven tests, extensive plugin ecosystem (xdist, cov, mock), and great test discovery.",
     "python_testing"),
    ("What are pytest fixtures?",
     "Fixtures are @pytest.fixture functions providing setup/teardown and shared resources via dependency injection. They support scope control (function/class/module/session) and yield for teardown.",
     "python_testing"),
    ("What is the difference between pytest fixture scope levels?",
     "function: new instance per test (default). class: shared across methods in a class. module: shared across all tests in a file. session: shared across the entire test run. Higher scope = faster but risk of shared-state pollution.",
     "python_testing"),
    ("How do you use pytest parametrize?",
     "Use @pytest.mark.parametrize('arg', [val1, val2]) to run the same test with multiple inputs. Combine arguments with tuples. Use pytest.param with marks to xfail or skip specific combinations.",
     "python_testing"),
    ("What is unittest.mock and how do you use it?",
     "unittest.mock provides Mock, MagicMock, and patch for replacing dependencies. Use @patch to replace an object during a test. Verify calls with assert_called_with, call_count, and call_args.",
     "python_testing"),
    ("What is the difference between Mock and MagicMock?",
     "Mock is a basic mock object. MagicMock also supports magic methods (__len__, __str__, __enter__, etc.), making it compatible with context managers and iteration. Use MagicMock by default.",
     "python_testing"),
    ("How do you measure test coverage in Python?",
     "Use pytest-cov: pytest --cov=src --cov-report=html. Configure .coveragerc or pyproject.toml to set minimum thresholds (fail_under = 80) and exclude generated code.",
     "python_testing"),
    ("What is tox?",
     "tox creates virtual environments and runs tests across multiple Python versions. Define environments in tox.ini with dependencies and commands. Use in CI to verify compatibility across Python 3.10, 3.11, 3.12.",
     "python_testing"),
    ("What is property-based testing?",
     "Property-based testing generates random inputs to test invariants. Hypothesis is the main Python library. Define @given strategies and assertions about properties that must always hold. It finds edge cases automatically.",
     "python_testing"),
    ("How do you test async code in Python?",
     "Use pytest-asyncio with @pytest.mark.asyncio. Define async test functions. Use AsyncMock for coroutines. For httpx: use AsyncClient in a fixture. For SQLAlchemy: use async session fixtures.",
     "python_testing"),

    # --- Test Design ---
    ("What is equivalence partitioning?",
     "Equivalence partitioning divides inputs into groups expected to behave the same way. Test one value per partition. For age 18-65: test <18 (invalid), 18-65 (valid), >65 (invalid).",
     "test_design"),
    ("What is boundary value analysis?",
     "BVA tests at the edges of equivalence partitions where defects are most common. For range 1-100: test 0, 1, 2, 99, 100, 101. Combine with equivalence partitioning for efficient coverage.",
     "test_design"),
    ("What is decision table testing?",
     "Decision table testing systematically captures combinations of conditions and their expected actions. Each column is a rule, rows are conditions and actions. Ensures all relevant combinations are tested.",
     "test_design"),
    ("What is state transition testing?",
     "State transition testing models systems with state-dependent behavior. Draw a state diagram and derive test cases covering valid transitions, invalid transitions, and multi-step sequences.",
     "test_design"),
    ("What is exploratory testing?",
     "Exploratory testing is simultaneous learning, test design, and execution guided by a charter. It finds unexpected defects that scripted tests miss. Automation handles regression; exploratory handles novel scenarios.",
     "test_design"),
    ("What is risk-based testing?",
     "Risk-based testing prioritizes tests by likelihood and impact of failure. High-risk areas (complex business logic, security, payment) get more thorough testing. Ensures critical paths are always verified.",
     "test_design"),
    ("What is the difference between black-box and white-box testing?",
     "Black-box treats the system as opaque — tests based on specifications. White-box examines internal code paths (unit tests, coverage). Gray-box combines both, using some internal knowledge.",
     "test_design"),
    ("What is mutation testing?",
     "Mutation testing introduces small code changes (mutations) and checks if tests detect them. Tools: mutmut (Python), PIT (Java). It measures test effectiveness beyond coverage.",
     "test_design"),
    ("What makes a good bug report?",
     "Clear title, steps to reproduce, expected vs actual behavior, environment details, screenshots/logs, severity/priority, and reproducibility rate. Any developer should be able to reproduce it independently.",
     "test_design"),
    ("What is the difference between severity and priority?",
     "Severity is the technical impact (critical crash vs minor typo). Priority is the business urgency (a typo on the homepage may be high priority despite low severity). QA sets severity; product sets priority.",
     "test_design"),

    # --- Performance Testing ---
    ("What is the difference between load, stress, and spike testing?",
     "Load testing checks behavior under expected concurrent users. Stress testing finds the breaking point. Spike testing applies sudden large load increases. Together they define capacity limits and failure modes.",
     "performance"),
    ("What is k6?",
     "k6 is a modern load testing tool with JavaScript-based scripts. Define VUs, scenarios, and thresholds. Outputs p95 response time, RPS, error rate. Integrates with Grafana for visualization.",
     "performance"),
    ("What are key performance metrics to monitor?",
     "Response time (p50, p95, p99), throughput (RPS), error rate, concurrent users, CPU/memory utilization, database query times, and cache hit rates. Establish baselines and alert on regressions.",
     "performance"),
    ("What is a performance baseline?",
     "A baseline is measured performance under known conditions (e.g., 100 VUs, normal data volume). Future tests compare against it to detect regressions. Without a baseline, you can't measure degradation.",
     "performance"),
    ("What is the difference between horizontal and vertical scaling?",
     "Vertical scaling adds resources (CPU/RAM) to existing servers. Horizontal scaling adds more instances behind a load balancer. Performance tests validate both and identify which bottlenecks prevent scaling.",
     "performance"),

    # --- Mobile Testing ---
    ("What is Appium?",
     "Appium is an open-source mobile automation framework using WebDriver protocol. It controls native, hybrid, and web apps on iOS and Android without app recompilation — uses XCTest for iOS, UIAutomator2 for Android.",
     "mobile"),
    ("What is the difference between native, hybrid, and web mobile apps?",
     "Native apps use platform SDKs (Swift/Kotlin). Hybrid apps wrap web content in a native container (Ionic, Cordova). Web apps run in the mobile browser. Each requires a different automation approach.",
     "mobile"),
    ("What are key considerations for mobile test automation?",
     "Device fragmentation, OS versions, network conditions (3G, offline), device-specific gestures, battery state, permissions handling, push notifications, deep linking, and app lifecycle transitions.",
     "mobile"),
    ("What is the difference between real devices and emulators?",
     "Emulators are faster and cheaper but miss real hardware behavior (battery, sensors, memory pressure). Real devices catch device-specific bugs. Use emulators for CI, real devices for release validation.",
     "mobile"),
    ("What mobile testing tools exist beyond Appium?",
     "XCUITest (iOS), Espresso (Android), Detox (React Native), Flutter integration_test, Maestro (YAML-based), BrowserStack Automate, and Sauce Labs for cloud device farms.",
     "mobile"),

    # --- Database Testing ---
    ("What is database testing and what do you verify?",
     "DB testing validates data integrity, schema, query performance, stored procedures, triggers, and security. Verify CRUD correctness, constraints (NOT NULL, UNIQUE, FK), and ACID transaction compliance.",
     "database"),
    ("How do you manage test data?",
     "Use factories/fixtures to create test-specific data. Clean up after each test (teardown or transactions). Seed scripts for consistent state. Avoid production data. Tools: factory_boy, Faker.",
     "database"),
    ("What is a database migration and how do you test it?",
     "Migrations are versioned schema changes (Alembic for Python, Flyway for Java). Test by running migrations on production data copies, verifying schema and data transformations, and ensuring rollback works.",
     "database"),
    ("What SQL queries should every QA engineer know?",
     "SELECT with JOIN (INNER, LEFT, RIGHT), GROUP BY with HAVING, subqueries, window functions (ROW_NUMBER, RANK), EXPLAIN ANALYZE for query plans, and transaction control (BEGIN, COMMIT, ROLLBACK).",
     "database"),
    ("How do you test database performance?",
     "Analyze slow query logs, use EXPLAIN ANALYZE, check for missing indexes, measure execution time under load, test with production-like data volumes, and monitor for long-running queries.",
     "database"),

    # --- Security Testing ---
    ("What is OWASP Top 10?",
     "OWASP Top 10 lists the most critical web security risks (injection, broken auth, XSS, IDOR, etc.). QA should include security tests in test plans, use DAST tools like OWASP ZAP, and validate security controls.",
     "security"),
    ("What is SQL injection and how do you test for it?",
     "SQL injection inserts malicious SQL via inputs to manipulate queries. Test by submitting ' OR 1=1-- in form fields and API params. Verify parameterized queries are used — responses should be error messages, not data leaks.",
     "security"),
    ("What is XSS and how do you test for it?",
     "XSS injects malicious scripts via user input that execute in other users' browsers. Test by submitting <script>alert(1)</script> in inputs. Verify the app escapes output. Three types: reflected, stored, DOM-based.",
     "security"),
    ("What is OWASP ZAP?",
     "OWASP ZAP is a free DAST tool for scanning web apps for vulnerabilities. Use as an intercepting proxy during manual testing or in active scan mode in CI. The ZAP API enables headless automation.",
     "security"),
    ("What is the difference between authentication and authorization testing?",
     "Authentication testing verifies identity (login, MFA, session management). Authorization testing verifies access control (can user A access user B's data? Can a regular user hit admin endpoints?).",
     "security"),

    # --- Agile / QA Process ---
    ("What is the role of QA in an Agile team?",
     "QA collaborates throughout the sprint: writing acceptance criteria during planning, automating tests alongside development, participating in code reviews, and providing rapid feedback rather than gate-keeping at the end.",
     "agile"),
    ("What is Definition of Done and how does QA contribute?",
     "DoD is a checklist that must be met before a story is 'done' (code reviewed, unit tests written, integration tests passing, feature tested, docs updated). QA defines and enforces test-related DoD criteria.",
     "agile"),
    ("How do you estimate testing effort?",
     "Use story points (relative complexity), task decomposition (hours per test case type), historical velocity, risk assessment (complex features need more testing), and automation potential.",
     "agile"),
    ("What metrics should QA track?",
     "Defect density, escaped defects (found in production), test coverage, automation coverage, flaky test rate, mean time to detect (MTTD), test execution time, and defect resolution time.",
     "agile"),
    ("What is the difference between QA and QC?",
     "QA (Quality Assurance) is process-focused — preventing defects through standards and strategy. QC (Quality Control) is product-focused — detecting defects through testing. QA happens throughout; QC validates the final product.",
     "agile"),
    ("How do you handle testing when requirements are unclear?",
     "Ask for clarification using BDD Given-When-Then examples. Document assumptions and get sign-off. Use exploratory testing to discover actual behavior. Mark unclear areas for follow-up and test what you know.",
     "agile"),
    ("What is Three Amigos meeting?",
     "Three Amigos brings together developer, QA, and product owner to discuss a story before implementation. It surfaces misunderstandings early, aligns acceptance criteria, and lets QA prepare tests in advance.",
     "agile"),
    ("What is TDD and how does it affect QA?",
     "TDD means writing a failing test before implementing code (Red → Green → Refactor). It produces more testable, modular code and comprehensive unit tests, letting QA focus on integration and exploratory testing.",
     "agile"),
    ("What is BDD?",
     "BDD (Behavior Driven Development) expresses tests as plain-language scenarios (Given-When-Then). Tools: Cucumber, Behave (Python), SpecFlow. It aligns business and technical stakeholders on acceptance criteria.",
     "agile"),
    ("What is a sprint retrospective from a QA perspective?",
     "QA discusses test bottlenecks, automation wins, flaky test issues, escaped defects, and proposes process improvements. It's a chance to continuously improve quality practices within the team.",
     "agile"),
]


def init_database(db_path: Path) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS qa_knowledge (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                question    TEXT NOT NULL,
                answer      TEXT NOT NULL,
                topic       TEXT DEFAULT 'general',
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        if conn.execute("SELECT COUNT(*) FROM qa_knowledge").fetchone()[0] == 0:
            conn.executemany(
                "INSERT INTO qa_knowledge (question, answer, topic) VALUES (?, ?, ?)",
                _QA_SEED,
            )
            print(f"[DB] Seeded {len(_QA_SEED)} Q&A pairs.", flush=True)


def search_qa_database(query: str, limit: int = 5) -> dict:
    """Full-text search the Q&A database by keyword matching."""
    terms = query.lower().split()
    clauses = " OR ".join(
        ["LOWER(question) LIKE ? OR LOWER(answer) LIKE ? OR LOWER(topic) LIKE ?" for _ in terms]
    )
    params: list = []
    for term in terms:
        params.extend([f"%{term}%", f"%{term}%", f"%{term}%"])
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            f"SELECT question, answer, topic FROM qa_knowledge WHERE {clauses} LIMIT ?",
            params + [limit],
        ).fetchall()
    if not rows:
        return {"results": [], "message": "No matching Q&A found."}
    return {"results": [{"question": q, "answer": a, "topic": t} for q, a, t in rows]}


def add_qa_pair(question: str, answer: str, topic: str = "general") -> dict:
    """Persist a new Q&A pair learned during conversation."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO qa_knowledge (question, answer, topic) VALUES (?, ?, ?)",
            (question, answer, topic),
        )
    push(f"[New Q&A] {question[:80]}")
    return {"added": "ok", "question": question}


# ---------------------------------------------------------------------------
# Tools – definitions and registry
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "record_user_details",
            "description": "Record that a user wants to stay in touch and has provided an email address.",
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {"type": "string", "description": "The email address of this user"},
                    "name": {"type": "string", "description": "The user's name, if provided"},
                    "notes": {"type": "string", "description": "Any additional context worth recording"},
                },
                "required": ["email"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "record_unknown_question",
            "description": "Record any question you couldn't answer even after searching your tools.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The question that couldn't be answered"},
                },
                "required": ["question"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Semantic search over Oleh's personal knowledge base (LinkedIn, summary, professional context). "
                "Use this to find specific details about his background, skills, or experience."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for"},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_qa_database",
            "description": (
                "Search the Automation QA knowledge database for technical questions and detailed answers. "
                "Always use this before answering any QA or testing question to ensure accuracy."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The topic or question to look up"},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_qa_pair",
            "description": "Save a new useful Q&A pair to the knowledge database for future conversations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The question"},
                    "answer": {"type": "string", "description": "The answer"},
                    "topic": {
                        "type": "string",
                        "description": "Topic category (e.g. selenium, api_testing, ci_cd, python_testing)",
                    },
                },
                "required": ["question", "answer"],
                "additionalProperties": False,
            },
        },
    },
]


# ---------------------------------------------------------------------------
# PDF / text loading
# ---------------------------------------------------------------------------


def load_linkedin_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    parts: list[str] = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            parts.append(text.strip())
    return "\n\n".join(parts)


def load_me_documents() -> list[str]:
    """Load summary, LinkedIn PDF, and all .txt/.md files from me/."""
    docs: list[str] = []
    if SUMMARY_PATH.is_file():
        docs.append(SUMMARY_PATH.read_text(encoding="utf-8"))
    if LINKEDIN_PDF.is_file():
        docs.append(load_linkedin_text(LINKEDIN_PDF))
    for path in sorted(ME_DIR.glob("*.txt")):
        if path == SUMMARY_PATH:
            continue
        docs.append(path.read_text(encoding="utf-8"))
    for path in sorted(ME_DIR.glob("*.md")):
        docs.append(path.read_text(encoding="utf-8"))
    return docs


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class Evaluation(BaseModel):
    is_acceptable: bool
    feedback: str


def evaluate(
    client: OpenAI,
    reply: str,
    message: str,
    history: list,
    evaluator_system: str,
) -> Evaluation:
    history_text = json.dumps(history, ensure_ascii=False, indent=2, default=str)
    user_prompt = (
        f"Conversation so far:\n{history_text}\n\n"
        f"Latest user message:\n{message}\n\n"
        f"Agent's latest response:\n{reply}\n\n"
        "Evaluate the response: is it acceptable? Provide feedback if not."
    )
    response = client.beta.chat.completions.parse(
        model=EVALUATOR_MODEL,
        messages=[
            {"role": "system", "content": evaluator_system},
            {"role": "user", "content": user_prompt},
        ],
        response_format=Evaluation,
    )
    parsed = response.choices[0].message.parsed
    return parsed if parsed is not None else Evaluation(is_acceptable=True, feedback="Parse failed; accepting.")


def build_evaluator_system(name: str, ctx: str) -> str:
    return (
        "You are a quality evaluator for an AI agent acting as a professional on a personal website. "
        f"The agent plays the role of {name}, a Principal QA Automation Engineer. "
        "Evaluate whether the latest response is: accurate, professional, persona-consistent, "
        "technically correct for QA/automation topics, and engaging. "
        f"Context about {name}:\n\n{ctx}\n"
        "Reply with is_acceptable (bool) and feedback (string with specific improvement suggestions if not acceptable)."
    )


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------


def context_block(summary: str, linkedin: str) -> str:
    return f"## Summary:\n{summary}\n\n## LinkedIn Profile:\n{linkedin}\n"


def build_system_prompt(name: str, ctx: str) -> str:
    return (
        f"You are acting as {name}, a Principal QA Automation Engineer, answering questions on "
        f"{name}'s personal website. Represent {name} faithfully and professionally.\n\n"
        "**Tool usage (use proactively):**\n"
        "- search_knowledge_base: fetch specific details about Oleh's background when needed\n"
        "- search_qa_database: ALWAYS use this for technical QA/automation questions to give accurate, detailed answers\n"
        "- record_unknown_question: use when you genuinely cannot answer after searching\n"
        "- record_user_details: use when the user shares their email\n"
        "- add_qa_pair: save new useful knowledge you encounter during conversation\n\n"
        "Be professional and engaging — this visitor may be a potential employer or client. "
        "Steer interested visitors towards sharing their email so you can follow up.\n\n"
        f"{ctx}"
        f"\nAlways stay in character as {name}."
    )


# ---------------------------------------------------------------------------
# Tool-use loop + evaluator loop (agentic patterns)
# ---------------------------------------------------------------------------


def _run_tool_loop(
    client: OpenAI,
    messages: list,
    tool_registry: dict,
) -> str:
    """Pattern 1: tool-use loop — iterate until finish_reason != 'tool_calls'."""
    for _ in range(MAX_TOOL_ITERATIONS):
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            tools=TOOLS,
        )
        choice = response.choices[0]
        if choice.finish_reason != "tool_calls":
            return choice.message.content or ""

        tool_msg = choice.message
        results: list[dict] = []
        for tc in tool_msg.tool_calls:
            fn_name = tc.function.name
            args = json.loads(tc.function.arguments)
            print(f"[Tool] {fn_name}({list(args.keys())})", flush=True)
            fn = tool_registry.get(fn_name)
            result = fn(**args) if fn else {"error": f"Unknown tool: {fn_name}"}
            results.append({"role": "tool", "content": json.dumps(result), "tool_call_id": tc.id})

        messages.append(tool_msg)
        messages.extend(results)

    print(f"[Tool] Max iterations ({MAX_TOOL_ITERATIONS}) reached — returning last content.", flush=True)
    return messages[-1].get("content", "") if isinstance(messages[-1], dict) else ""


def chat(
    message: str,
    history: list,
    *,
    client: OpenAI,
    system_prompt: str,
    evaluator_system: str,
    tool_registry: dict,
) -> str:
    """Pattern 2: evaluator + rerun — regenerate low-quality replies."""
    messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": message}]
    reply = _run_tool_loop(client, messages, tool_registry)

    evaluation = evaluate(client, reply, message, history, evaluator_system)
    print(f"[Eval] acceptable={evaluation.is_acceptable}", flush=True)
    if evaluation.is_acceptable:
        return reply

    # Rerun with feedback injected into the system prompt
    print(f"[Eval] Regenerating. Feedback: {evaluation.feedback}", flush=True)
    updated_system = (
        f"{system_prompt}\n\n## Previous answer was rejected by quality control\n"
        f"Your attempted answer:\n{reply}\n\n"
        f"Reason for rejection:\n{evaluation.feedback}\n\n"
        "Please provide an improved response."
    )
    messages2 = [{"role": "system", "content": updated_system}] + history + [{"role": "user", "content": message}]
    return _run_tool_loop(client, messages2, tool_registry)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    if not LINKEDIN_PDF.is_file():
        sys.exit(f"LinkedIn PDF not found: {LINKEDIN_PDF}")
    if not SUMMARY_PATH.is_file():
        sys.exit(f"Summary file not found: {SUMMARY_PATH}")
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set.")

    client = OpenAI()

    # Initialise SQL knowledge base
    ME_DIR.mkdir(exist_ok=True)
    init_database(DB_PATH)

    # Load all personal documents and build RAG vector store
    print("[RAG] Loading documents...", flush=True)
    documents = load_me_documents()
    vector_store = build_vector_store(client, documents)

    # Build prompts
    summary = SUMMARY_PATH.read_text(encoding="utf-8")
    linkedin = load_linkedin_text(LINKEDIN_PDF)
    ctx = context_block(summary, linkedin)
    system_prompt = build_system_prompt(NAME, ctx)
    evaluator_system = build_evaluator_system(NAME, ctx)

    # Assemble tool registry (search_knowledge_base is a closure over client + vector_store)
    def _search_knowledge_base(query: str) -> dict:
        query_emb = _get_embeddings(client, [query])[0]
        chunks = vector_store.search(query_emb)
        return {"results": chunks}

    tool_registry: dict = {
        "record_user_details": record_user_details,
        "record_unknown_question": record_unknown_question,
        "search_knowledge_base": _search_knowledge_base,
        "search_qa_database": search_qa_database,
        "add_qa_pair": add_qa_pair,
    }

    def _chat(message: str, history: list) -> str:
        return chat(
            message,
            history,
            client=client,
            system_prompt=system_prompt,
            evaluator_system=evaluator_system,
            tool_registry=tool_registry,
        )

    try:
        gr.ChatInterface(_chat, type="messages").launch()
    except OpenAIError as exc:
        sys.exit(f"OpenAI error: {exc}")


if __name__ == "__main__":
    main()
