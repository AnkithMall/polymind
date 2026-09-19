"""Demo seed script — creates the 5 custom demo domains with suites.

Usage:
    polymind demo seed          # Create all 5 demo domains
    polymind demo seed --reset  # Delete all custom domains first, then seed

This script creates:
  - frontend     (3 suites, ~20 questions)
  - backend      (3 suites, ~20 questions)
  - system-design (3 suites, ~20 questions)
  - architecture  (3 suites, ~20 questions)
  - cybersecurity (3 suites, ~20 questions)
"""

from __future__ import annotations

import typer
from rich.console import Console

from polymind.core.confidence.artifact import (
    load_all_domains,
    save_custom_domain,
)
from polymind.core.confidence.types import Domain, TestQuestion, TestSuite

app = typer.Typer()
console = Console()


def _make_frontend() -> Domain:
    return Domain(
        id="frontend",
        name="Frontend Development",
        description="UI engineering, React, TypeScript, browser architecture and frontend debugging",
        custom=True,
        aliases=["ui", "react", "typescript", "browser", "css", "html", "web frontend"],
        suites=[
            TestSuite(
                id="frontend-core",
                name="Frontend Core",
                description="Core frontend concepts: DOM, components, state management",
                difficulty="medium",
                questions=[
                    TestQuestion(
                        id="fc1",
                        prompt="What is the virtual DOM and why do React and similar frameworks use it?",
                        expected="The virtual DOM is a lightweight in-memory representation of the real DOM. Frameworks use it to batch and minimize expensive real DOM updates by diffing the virtual tree and applying only necessary changes.",
                        evaluation="keyword_match",
                        keywords=["virtual DOM", "diff", "real DOM", "lightweight", "update"],
                    ),
                    TestQuestion(
                        id="fc2",
                        prompt="Explain the difference between controlled and uncontrolled components in React.",
                        expected="In controlled components, form data is handled by React state (value and onChange). In uncontrolled components, the DOM itself holds the state, accessed via refs.",
                        evaluation="keyword_match",
                        keywords=["controlled", "uncontrolled", "state", "refs", "onChange"],
                    ),
                    TestQuestion(
                        id="fc3",
                        prompt="What are React hooks and name three commonly used ones.",
                        expected="Hooks are functions that let functional components use state and lifecycle features. Common hooks include useState, useEffect, and useRef.",
                        evaluation="keyword_match",
                        keywords=["hooks", "useState", "useEffect", "functional", "state"],
                    ),
                    TestQuestion(
                        id="fc4",
                        prompt="What is CSS specificity and how is it calculated?",
                        expected="CSS specificity determines which rule wins when multiple rules target the same element. It is calculated as a tuple (inline, IDs, classes/attributes, elements) where higher values override lower ones.",
                        evaluation="keyword_match",
                        keywords=["specificity", "inline", "IDs", "classes", "override"],
                    ),
                    TestQuestion(
                        id="fc5",
                        prompt="Explain the CSS box model.",
                        expected="The CSS box model describes how elements are rendered as boxes with content, padding, border, and margin. box-sizing controls whether width/height include padding and border.",
                        evaluation="keyword_match",
                        keywords=["content", "padding", "border", "margin", "box-sizing"],
                    ),
                ],
            ),
            TestSuite(
                id="frontend-architecture",
                name="Frontend Architecture",
                description="Component patterns, state management, and application architecture",
                difficulty="hard",
                questions=[
                    TestQuestion(
                        id="fa1",
                        prompt="Compare Redux, Zustand, and Jotai for state management in a React application.",
                        expected="Redux is a predictable state container with reducers and actions. Zustand is a minimal, hook-based store. Jotai is an atomic state management library. Redux suits large apps with complex state, Zustand for moderate apps, Jotai for fine-grained reactivity.",
                        evaluation="keyword_match",
                        keywords=["Redux", "Zustand", "Jotai", "reducer", "atomic", "hook"],
                    ),
                    TestQuestion(
                        id="fa2",
                        prompt="What is server-side rendering (SSR) and how does it differ from static site generation (SSG)?",
                        expected="SSR renders pages on each request on the server. SSG pre-renders pages at build time. SSR is dynamic and fresh per request; SSG is faster at serving but content is static until rebuild.",
                        evaluation="keyword_match",
                        keywords=["SSR", "SSG", "server", "build time", "request", "static"],
                    ),
                    TestQuestion(
                        id="fa3",
                        prompt="Explain the concept of code splitting and lazy loading in frontend applications.",
                        expected="Code splitting divides JavaScript bundles into smaller chunks loaded on demand. Lazy loading defers loading of non-critical resources. Together they reduce initial page load time and memory usage.",
                        evaluation="keyword_match",
                        keywords=[
                            "code splitting",
                            "lazy loading",
                            "chunks",
                            "bundles",
                            "on demand",
                        ],
                    ),
                ],
            ),
            TestSuite(
                id="frontend-debugging",
                name="Frontend Debugging",
                description="Browser dev tools, performance profiling, and common bugs",
                difficulty="medium",
                questions=[
                    TestQuestion(
                        id="fd1",
                        prompt="How do you debug a memory leak in a single-page application?",
                        expected="Use Chrome DevTools heap snapshots to compare memory over time, take snapshots before and after actions, look for detached DOM nodes and event listener leaks, and use the Performance tab to track allocations.",
                        evaluation="keyword_match",
                        keywords=[
                            "heap snapshot",
                            "memory leak",
                            "detached",
                            "DevTools",
                            "allocations",
                        ],
                    ),
                    TestQuestion(
                        id="fd2",
                        prompt="What causes a layout thrash and how do you prevent it?",
                        expected="Layout thrash occurs when JavaScript repeatedly reads layout properties (offsetHeight, getBoundingClientRect) and then writes to the DOM, forcing synchronous reflows. Prevent by batching reads and writes, using requestAnimationFrame, and minimizing forced reflows.",
                        evaluation="keyword_match",
                        keywords=[
                            "layout thrash",
                            "reflow",
                            "offsetHeight",
                            "batch",
                            "requestAnimationFrame",
                        ],
                    ),
                    TestQuestion(
                        id="fd3",
                        prompt="What is a CORS error and how do you fix it?",
                        expected="CORS (Cross-Origin Resource Sharing) errors occur when a browser blocks requests to a different origin due to missing or restrictive Access-Control-Allow-Origin headers. Fix by configuring the server to send proper CORS headers or using a proxy.",
                        evaluation="keyword_match",
                        keywords=[
                            "CORS",
                            "origin",
                            "Access-Control-Allow-Origin",
                            "header",
                            "proxy",
                        ],
                    ),
                ],
            ),
        ],
    )


def _make_backend() -> Domain:
    return Domain(
        id="backend",
        name="Backend Development",
        description="APIs, databases, server architecture, and backend debugging",
        custom=True,
        aliases=["api", "server", "rest", "graphql", "database", "sql", "backend services"],
        suites=[
            TestSuite(
                id="backend-api",
                name="Backend API Design",
                description="REST API design, authentication, and API best practices",
                difficulty="medium",
                questions=[
                    TestQuestion(
                        id="ba1",
                        prompt="What are the key principles of RESTful API design?",
                        expected="REST APIs should use proper HTTP methods (GET, POST, PUT, DELETE), return appropriate status codes, use resource-based URLs, be stateless, support pagination, and use consistent naming conventions.",
                        evaluation="keyword_match",
                        keywords=[
                            "HTTP methods",
                            "stateless",
                            "resource",
                            "status codes",
                            "pagination",
                        ],
                    ),
                    TestQuestion(
                        id="ba2",
                        prompt="Explain the difference between authentication and authorization.",
                        expected="Authentication verifies identity (who you are). Authorization determines permissions (what you can do). Authentication comes first, then authorization checks are applied.",
                        evaluation="keyword_match",
                        keywords=["authentication", "authorization", "identity", "permissions"],
                    ),
                    TestQuestion(
                        id="ba3",
                        prompt="What is a JWT and what are its advantages and disadvantages?",
                        expected="JWT (JSON Web Token) is a stateless token containing header, payload, and signature. Advantages: no server-side session storage, scalable, self-contained. Disadvantages: can't be revoked easily, larger token size, sensitive data in payload if not encrypted.",
                        evaluation="keyword_match",
                        keywords=["JWT", "stateless", "payload", "signature", "token"],
                    ),
                    TestQuestion(
                        id="ba4",
                        prompt="How would you design a rate limiting system for an API?",
                        expected="Use algorithms like token bucket, sliding window, or fixed window counters. Store counts in Redis for distributed systems. Return 429 status code with Retry-After header when limits are exceeded.",
                        evaluation="keyword_match",
                        keywords=[
                            "rate limiting",
                            "token bucket",
                            "sliding window",
                            "Redis",
                            "429",
                        ],
                    ),
                ],
            ),
            TestSuite(
                id="backend-database",
                name="Backend Database",
                description="Database design, queries, indexing, and optimization",
                difficulty="hard",
                questions=[
                    TestQuestion(
                        id="bd1",
                        prompt="When would you choose a SQL database over NoSQL, and vice versa?",
                        expected="SQL for structured data with complex relationships, ACID compliance, and complex queries. NoSQL for flexible schemas, horizontal scaling, high write throughput, and document/key-value access patterns.",
                        evaluation="keyword_match",
                        keywords=[
                            "SQL",
                            "NoSQL",
                            "ACID",
                            "schema",
                            "horizontal scaling",
                            "relationships",
                        ],
                    ),
                    TestQuestion(
                        id="bd2",
                        prompt="What is database indexing and when should you create one?",
                        expected="An index is a data structure (B-tree, hash) that speeds up data retrieval at the cost of additional storage and slower writes. Create indexes on columns used in WHERE, JOIN, and ORDER BY clauses frequently.",
                        evaluation="keyword_match",
                        keywords=["index", "B-tree", "WHERE", "JOIN", "storage", "retrieval"],
                    ),
                    TestQuestion(
                        id="bd3",
                        prompt="Explain database transactions and ACID properties.",
                        expected="A transaction is a logical unit of work. ACID: Atomicity (all or nothing), Consistency (valid state transitions), Isolation (concurrent transactions don't interfere), Durability (committed data persists).",
                        evaluation="keyword_match",
                        keywords=[
                            "transaction",
                            "atomicity",
                            "consistency",
                            "isolation",
                            "durability",
                        ],
                    ),
                ],
            ),
            TestSuite(
                id="backend-debugging",
                name="Backend Debugging",
                description="Server debugging, profiling, and performance optimization",
                difficulty="medium",
                questions=[
                    TestQuestion(
                        id="bdbg1",
                        prompt="How do you diagnose a slow API endpoint?",
                        expected="Check database query performance (explain plans), measure network latency, profile CPU/memory usage, check for N+1 query problems, review caching strategy, and use distributed tracing for microservices.",
                        evaluation="keyword_match",
                        keywords=["query", "latency", "N+1", "caching", "profiling", "tracing"],
                    ),
                    TestQuestion(
                        id="bdbg2",
                        prompt="What is connection pooling and why is it important?",
                        expected="Connection pooling maintains a cache of reusable database connections. It reduces the overhead of opening/closing connections per request, improves performance, and prevents connection exhaustion under load.",
                        evaluation="keyword_match",
                        keywords=["connection pool", "cache", "reuse", "overhead", "performance"],
                    ),
                    TestQuestion(
                        id="bdbg3",
                        prompt="How would you handle a production database outage?",
                        expected="Activate failover to read replica, check connection pool health, implement circuit breaker pattern, serve cached data if available, notify stakeholders, investigate root cause, and implement post-mortem improvements.",
                        evaluation="keyword_match",
                        keywords=[
                            "failover",
                            "replica",
                            "circuit breaker",
                            "cached",
                            "post-mortem",
                        ],
                    ),
                ],
            ),
        ],
    )


def _make_system_design() -> Domain:
    return Domain(
        id="system-design",
        name="System Design",
        description="Distributed systems, scalability, reliability engineering",
        custom=True,
        aliases=[
            "distributed systems",
            "scalability",
            "reliability",
            "high availability",
            "distributed",
        ],
        suites=[
            TestSuite(
                id="distributed-systems",
                name="Distributed Systems",
                description="CAP theorem, consensus, distributed data stores",
                difficulty="hard",
                questions=[
                    TestQuestion(
                        id="ds1",
                        prompt="Explain the CAP theorem and its implications for distributed systems.",
                        expected="CAP theorem states a distributed system can only guarantee two of three properties: Consistency (all nodes see the same data), Availability (every request gets a response), Partition tolerance (system works despite network failures). In practice, networks always partition, so you choose between CP and AP.",
                        evaluation="keyword_match",
                        keywords=["CAP", "consistency", "availability", "partition", "CP", "AP"],
                    ),
                    TestQuestion(
                        id="ds2",
                        prompt="What is eventual consistency and when is it acceptable?",
                        expected="Eventual consistency means all replicas will converge to the same state eventually, but not immediately. Acceptable for non-critical reads (social media feeds, analytics) where slight staleness is tolerable for better availability and performance.",
                        evaluation="keyword_match",
                        keywords=[
                            "eventual consistency",
                            "replicas",
                            "converge",
                            "staleness",
                            "availability",
                        ],
                    ),
                    TestQuestion(
                        id="ds3",
                        prompt="Compare message queues (Kafka, RabbitMQ) and when to use each.",
                        expected="Kafka is a distributed event streaming platform for high-throughput, ordered, replayable event logs. RabbitMQ is a traditional message broker for task queues with complex routing. Use Kafka for event sourcing and streaming; RabbitMQ for work queues and RPC patterns.",
                        evaluation="keyword_match",
                        keywords=[
                            "Kafka",
                            "RabbitMQ",
                            "event streaming",
                            "message broker",
                            "throughput",
                        ],
                    ),
                ],
            ),
            TestSuite(
                id="scalability",
                name="Scalability Patterns",
                description="Horizontal vs vertical scaling, load balancing, caching",
                difficulty="medium",
                questions=[
                    TestQuestion(
                        id="sc1",
                        prompt="What is the difference between horizontal and vertical scaling?",
                        expected="Vertical scaling (scale up) adds more resources to a single machine. Horizontal scaling (scale out) adds more machines. Horizontal is preferred for cloud-native apps as it provides better fault tolerance and is more cost-effective at scale.",
                        evaluation="keyword_match",
                        keywords=[
                            "horizontal",
                            "vertical",
                            "scale up",
                            "scale out",
                            "fault tolerance",
                        ],
                    ),
                    TestQuestion(
                        id="sc2",
                        prompt="Explain common caching strategies and their trade-offs.",
                        expected="Strategies include: cache-aside (lazy loading), write-through (write to cache and DB), write-behind (async DB write), and read-through. Trade-offs involve consistency, complexity, cache invalidation, and data freshness.",
                        evaluation="keyword_match",
                        keywords=[
                            "cache-aside",
                            "write-through",
                            "write-behind",
                            "invalidation",
                            "freshness",
                        ],
                    ),
                    TestQuestion(
                        id="sc3",
                        prompt="How would you design a system that handles 100K concurrent users?",
                        expected="Use load balancers (L7), horizontally scaled application servers, database read replicas, Redis for caching and sessions, CDN for static assets, message queues for async processing, and connection pooling.",
                        evaluation="keyword_match",
                        keywords=[
                            "load balancer",
                            "horizontal",
                            "read replicas",
                            "Redis",
                            "CDN",
                            "queue",
                        ],
                    ),
                ],
            ),
            TestSuite(
                id="reliability",
                name="Reliability Engineering",
                description="Fault tolerance, circuit breakers, monitoring",
                difficulty="hard",
                questions=[
                    TestQuestion(
                        id="re1",
                        prompt="What is the circuit breaker pattern and when should you use it?",
                        expected="Circuit breaker prevents cascading failures by stopping calls to a failing service after a threshold is reached. States: closed (normal), open (failing, fast-fail), half-open (testing recovery). Use when calling external services that may be unreliable.",
                        evaluation="keyword_match",
                        keywords=[
                            "circuit breaker",
                            "cascading",
                            "closed",
                            "open",
                            "half-open",
                            "threshold",
                        ],
                    ),
                    TestQuestion(
                        id="re2",
                        prompt="Explain the difference between SLA, SLO, and SLI.",
                        expected="SLI (Service Level Indicator) is a measurable metric like uptime. SLO (Service Level Objective) is the target value for an SLI like 99.9% uptime. SLA (Service Level Agreement) is a contract defining consequences of missing the SLO.",
                        evaluation="keyword_match",
                        keywords=["SLA", "SLO", "SLI", "uptime", "target", "contract"],
                    ),
                    TestQuestion(
                        id="re3",
                        prompt="How do you implement graceful degradation in a microservices architecture?",
                        expected="Implement fallback responses, feature flags, timeout and retry policies, degraded functionality modes, and health checks. Use bulkhead pattern to isolate failures and return partial results when dependencies fail.",
                        evaluation="keyword_match",
                        keywords=[
                            "fallback",
                            "timeout",
                            "bulkhead",
                            "health check",
                            "degraded",
                            "isolation",
                        ],
                    ),
                ],
            ),
        ],
    )


def _make_architecture() -> Domain:
    return Domain(
        id="architecture",
        name="Architecture",
        description="Software architecture patterns, service boundaries, deployment architecture",
        custom=True,
        aliases=[
            "software architecture",
            "service boundaries",
            "deployment",
            "design patterns",
            "microservices",
        ],
        suites=[
            TestSuite(
                id="software-architecture",
                name="Software Architecture",
                description="Architecture patterns, design decisions, and trade-offs",
                difficulty="hard",
                questions=[
                    TestQuestion(
                        id="sa1",
                        prompt="Compare monolithic, microservices, and modular monolith architectures.",
                        expected="Monolithic is a single deployable unit (simple but hard to scale independently). Microservices are independently deployable services (flexible but complex). Modular monolith is a well-structured monolith with clear module boundaries (compromise between simplicity and modularity).",
                        evaluation="keyword_match",
                        keywords=[
                            "monolithic",
                            "microservices",
                            "modular",
                            "deployable",
                            "boundaries",
                        ],
                    ),
                    TestQuestion(
                        id="sa2",
                        prompt="What is Domain-Driven Design (DDD) and when should you use it?",
                        expected="DDD is a software design approach focusing on modeling software to match business domains. Use bounded contexts, aggregates, and ubiquitous language. Best for complex domains with rich business logic where alignment between code and domain matters.",
                        evaluation="keyword_match",
                        keywords=[
                            "DDD",
                            "bounded context",
                            "aggregate",
                            "ubiquitous language",
                            "domain",
                        ],
                    ),
                    TestQuestion(
                        id="sa3",
                        prompt="Explain event-driven architecture and its benefits.",
                        expected="Event-driven architecture uses events to communicate between components. Benefits include loose coupling, scalability, auditability, and temporal decoupling. Components publish events without knowing subscribers. Enables CQRS and event sourcing patterns.",
                        evaluation="keyword_match",
                        keywords=[
                            "event-driven",
                            "loose coupling",
                            "CQRS",
                            "event sourcing",
                            "decoupling",
                        ],
                    ),
                ],
            ),
            TestSuite(
                id="service-boundaries",
                name="Service Boundaries",
                description="Decomposing systems into services, API design between services",
                difficulty="hard",
                questions=[
                    TestQuestion(
                        id="sb1",
                        prompt="How do you decide where to draw service boundaries in a microservices architecture?",
                        expected="Use domain-driven design bounded contexts. Identify cohesive business capabilities. Consider data ownership, team boundaries, deployment independence, and failure isolation. Start with fewer services and split when needed.",
                        evaluation="keyword_match",
                        keywords=[
                            "bounded context",
                            "cohesive",
                            "data ownership",
                            "team boundaries",
                            "split",
                        ],
                    ),
                    TestQuestion(
                        id="sb2",
                        prompt="What is the strangler fig pattern and when would you use it?",
                        expected="Strangler fig pattern incrementally replaces parts of a legacy monolith by routing traffic to new services. Use when migrating from monolith to microservices without a big-bang rewrite. The new system grows around the old one until the old is replaced.",
                        evaluation="keyword_match",
                        keywords=[
                            "strangler fig",
                            "incremental",
                            "legacy",
                            "monolith",
                            "migration",
                            "rewrite",
                        ],
                    ),
                    TestQuestion(
                        id="sb3",
                        prompt="How do services communicate and what are the trade-offs of synchronous vs asynchronous communication?",
                        expected="Synchronous (REST, gRPC): simple, immediate response, but creates coupling and cascading failures. Asynchronous (queues, events): decoupled, resilient, but eventual consistency and complexity. Use sync for queries, async for commands and events.",
                        evaluation="keyword_match",
                        keywords=[
                            "synchronous",
                            "asynchronous",
                            "REST",
                            "gRPC",
                            "queues",
                            "coupling",
                        ],
                    ),
                ],
            ),
            TestSuite(
                id="deployment-architecture",
                name="Deployment Architecture",
                description="Container orchestration, CI/CD, cloud deployment strategies",
                difficulty="medium",
                questions=[
                    TestQuestion(
                        id="da1",
                        prompt="Compare blue-green, canary, and rolling deployment strategies.",
                        expected="Blue-green runs two identical environments, switching traffic instantly. Canary gradually routes a percentage of traffic to the new version. Rolling updates instances incrementally. Blue-green is fastest rollback, canary minimizes risk, rolling is simplest.",
                        evaluation="keyword_match",
                        keywords=[
                            "blue-green",
                            "canary",
                            "rolling",
                            "traffic",
                            "rollback",
                            "incremental",
                        ],
                    ),
                    TestQuestion(
                        id="da2",
                        prompt="What are the key considerations for designing a disaster recovery plan?",
                        expected="Define RPO (Recovery Point Objective) and RTO (Recovery Time Objective). Choose backup strategy (full, incremental, continuous replication). Design multi-region failover. Test recovery procedures regularly. Consider data consistency and DNS failover.",
                        evaluation="keyword_match",
                        keywords=[
                            "RPO",
                            "RTO",
                            "backup",
                            "failover",
                            "multi-region",
                            "disaster recovery",
                        ],
                    ),
                    TestQuestion(
                        id="da3",
                        prompt="Explain the twelve-factor app methodology.",
                        expected="12 factors: codebase, dependencies, config in env, backing services, build/release/run, processes, port binding, concurrency, disposability, dev/prod parity, logs as streams, admin processes. It's a methodology for building cloud-native, scalable SaaS apps.",
                        evaluation="keyword_match",
                        keywords=[
                            "12-factor",
                            "config",
                            "backing services",
                            "processes",
                            "cloud-native",
                            "SaaS",
                        ],
                    ),
                ],
            ),
        ],
    )


def _make_cybersecurity() -> Domain:
    return Domain(
        id="cybersecurity",
        name="Cybersecurity",
        description="Threat modeling, secure coding, application security",
        custom=True,
        aliases=["security", "threat modeling", "secure coding", "appsec", "infosec", "cyber"],
        suites=[
            TestSuite(
                id="threat-modeling",
                name="Threat Modeling",
                description="STRIDE, DREAD, attack trees, and threat identification",
                difficulty="hard",
                questions=[
                    TestQuestion(
                        id="tm1",
                        prompt="Explain the STRIDE threat modeling framework.",
                        expected="STRIDE categorizes threats: Spoofing (identity), Tampering (data), Repudiation (denial), Information disclosure (confidentiality), Denial of service (availability), Elevation of privilege (authorization). Each maps to a security property.",
                        evaluation="keyword_match",
                        keywords=[
                            "STRIDE",
                            "spoofing",
                            "tampering",
                            "repudiation",
                            "disclosure",
                            "elevation",
                        ],
                    ),
                    TestQuestion(
                        id="tm2",
                        prompt="How would you threat model a multi-tenant SaaS application?",
                        expected="Identify entry points (APIs, auth), assets (data, credentials), trust boundaries (tenant isolation). Key threats: cross-tenant data access, privilege escalation, token theft, API abuse. Mitigate with tenant isolation, RBAC, rate limiting, input validation.",
                        evaluation="keyword_match",
                        keywords=[
                            "multi-tenant",
                            "trust boundary",
                            "isolation",
                            "RBAC",
                            "cross-tenant",
                        ],
                    ),
                    TestQuestion(
                        id="tm3",
                        prompt="What is an attack tree and how do you use one?",
                        expected="An attack tree is a hierarchical diagram of attack vectors against a system. Root node is the attacker's goal, branches are attack methods, leaves are specific techniques. Used to systematically identify, prioritize, and communicate security risks.",
                        evaluation="keyword_match",
                        keywords=[
                            "attack tree",
                            "hierarchical",
                            "attack vector",
                            "goal",
                            "techniques",
                            "prioritize",
                        ],
                    ),
                ],
            ),
            TestSuite(
                id="secure-coding",
                name="Secure Coding",
                description="Input validation, authentication, cryptography best practices",
                difficulty="medium",
                questions=[
                    TestQuestion(
                        id="sc_sec1",
                        prompt="What are the OWASP Top 10 and why do they matter?",
                        expected="OWASP Top 10 lists the most critical web application security risks: injection, broken authentication, sensitive data exposure, XSS, broken access control, security misconfiguration, insecure deserialization, using vulnerable components, insufficient logging, and SSRF.",
                        evaluation="keyword_match",
                        keywords=[
                            "OWASP",
                            "injection",
                            "XSS",
                            "authentication",
                            "access control",
                            "SSRF",
                        ],
                    ),
                    TestQuestion(
                        id="sc_sec2",
                        prompt="Explain SQL injection and how to prevent it.",
                        expected="SQL injection inserts malicious SQL into queries via user input. Prevent with parameterized queries/prepared statements, ORM usage, input validation, least-privilege database permissions, and stored procedures.",
                        evaluation="keyword_match",
                        keywords=[
                            "SQL injection",
                            "parameterized",
                            "prepared statements",
                            "ORM",
                            "validation",
                        ],
                    ),
                    TestQuestion(
                        id="sc_sec3",
                        prompt="How should passwords be stored securely?",
                        expected="Use adaptive hashing algorithms like bcrypt, scrypt, or Argon2 with appropriate work factors. Never store plaintext or use fast hashes like MD5/SHA-1 alone. Add salt (handled by bcrypt/Argon2). Implement account lockout and rate limiting.",
                        evaluation="keyword_match",
                        keywords=["bcrypt", "Argon2", "salt", "hashing", "work factor", "lockout"],
                    ),
                ],
            ),
            TestSuite(
                id="application-security",
                name="Application Security",
                description="Security architecture, API security, and incident response",
                difficulty="hard",
                questions=[
                    TestQuestion(
                        id="as1",
                        prompt="How would you secure a REST API against common attacks?",
                        expected="Implement HTTPS/TLS, JWT with short expiry, rate limiting, input validation, CORS policy, request signing, API keys for service accounts, OWASP security headers, and logging of suspicious activity.",
                        evaluation="keyword_match",
                        keywords=[
                            "HTTPS",
                            "JWT",
                            "rate limiting",
                            "CORS",
                            "input validation",
                            "logging",
                        ],
                    ),
                    TestQuestion(
                        id="as2",
                        prompt="What is a security incident response plan and what are its phases?",
                        expected="Phases: Preparation (tools, team, playbooks), Identification (detect and analyze), Containment (limit damage), Eradication (remove threat), Recovery (restore systems), Lessons Learned (improve). Time is critical in each phase.",
                        evaluation="keyword_match",
                        keywords=[
                            "preparation",
                            "identification",
                            "containment",
                            "eradication",
                            "recovery",
                            "lessons",
                        ],
                    ),
                    TestQuestion(
                        id="as3",
                        prompt="Explain defense in depth and how to apply it to a web application.",
                        expected="Defense in depth uses multiple layers of security controls: network firewall, WAF, authentication, authorization, input validation, output encoding, encryption, logging, and monitoring. No single layer is sufficient; each compensates for others' weaknesses.",
                        evaluation="keyword_match",
                        keywords=[
                            "defense in depth",
                            "layers",
                            "firewall",
                            "WAF",
                            "encryption",
                            "monitoring",
                        ],
                    ),
                ],
            ),
        ],
    )


ALL_DEMO_DOMAINS = [
    _make_frontend,
    _make_backend,
    _make_system_design,
    _make_architecture,
    _make_cybersecurity,
]


@app.callback(invoke_without_command=True)
def demo_callback(
    ctx: typer.Context,
) -> None:
    """Demo preparation commands.

    Without a subcommand, shows available demo commands.
    """
    if ctx.invoked_subcommand is not None:
        return

    console.print("[bold]PolyMind Demo Commands[/]")
    console.print()
    console.print("  polymind demo seed          — Create 5 demo domains with suites")
    console.print("  polymind demo seed --reset  — Reset and re-seed demo domains")
    console.print()
    console.print("  Demo 1: Multi-domain pipeline with built-in domains")
    console.print("  Demo 2: Custom domain pipeline (requires 'polymind demo seed')")


@app.command("seed")
def seed_domains(
    reset: bool = typer.Option(False, "--reset", help="Delete all custom domains before seeding."),
) -> None:
    """Create the 5 demo custom domains with test suites.

    Seeds: frontend, backend, system-design, architecture, cybersecurity.
    Each domain has 3 suites with 3-5 questions each.

    Use --reset to delete all existing custom domains first.

    Examples:

        polymind demo seed

        polymind demo seed --reset
    """
    if reset:
        # Delete all existing custom domains
        from polymind.core.confidence.artifact import custom_domains_path

        custom_dir = custom_domains_path()
        if custom_dir.exists():
            deleted = 0
            for yaml_file in custom_dir.glob("*.yaml"):
                yaml_file.unlink()
                deleted += 1
            console.print(f"  Deleted {deleted} custom domain(s).")

    created = 0
    for factory in ALL_DEMO_DOMAINS:
        domain = factory()
        save_custom_domain(domain)
        q_count = sum(len(s.questions) for s in domain.suites)
        console.print(
            f"  [green]✓[/] Created domain '[cyan]{domain.id}[/]' "
            f"({len(domain.suites)} suites, {q_count} questions)"
        )
        created += 1

    console.print()
    console.print(f"  [bold green]Seeded {created} demo domains.[/]")
    console.print()
    console.print("  Next steps:")
    console.print("    polymind confidence compute -d frontend")
    console.print("    polymind confidence compute -d backend")
    console.print("    polymind confidence compute -d system-design")
    console.print("    polymind confidence compute -d architecture")
    console.print("    polymind confidence compute -d cybersecurity")
    console.print()
    console.print("    polymind capability show")
    console.print()
    console.print('    polymind pipeline run "<prompt>" -v')


@app.command("status")
def demo_status() -> None:
    """Show which demo domains have been created and their status.

    Examples:

        polymind demo status
    """
    from polymind.core.confidence.artifact import load_confidence

    demo_ids = {"frontend", "backend", "system-design", "architecture", "cybersecurity"}
    all_domains = load_all_domains()
    confidence = load_confidence()

    console.print("[bold]Demo Domain Status[/]")
    console.print()

    for domain_id in sorted(demo_ids):
        domain = next((d for d in all_domains if d.id == domain_id and d.custom), None)
        if domain is None:
            console.print(f"  [red]✗[/] {domain_id}: not created")
            continue

        q_count = sum(len(s.questions) for s in domain.suites)

        # Check confidence
        scored = False
        for conf in confidence.values():
            if domain_id in conf.domains:
                scored = True
                ds = conf.domains[domain_id]
                console.print(
                    f"  [green]✓[/] {domain_id}: {len(domain.suites)} suites, "
                    f"{q_count} questions, [cyan]scored {ds.overall:.1f}%[/]"
                )
                break
        if not scored:
            console.print(
                f"  [yellow]~[/] {domain_id}: {len(domain.suites)} suites, "
                f"{q_count} questions, [dim]not yet scored[/]"
            )
