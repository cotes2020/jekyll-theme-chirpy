---
title: "Why your Kubernetes platform needs a security baseline (not just another tool)"
date: 2025-12-10 09:00:00 +0100
categories:
  - kubernetes
  - cloud security
  - platform security
tags:
  - kubernetes
  - security
  - baseline
  - hardening
  - saas
  - platform
---

Most Kubernetes security articles stop at generic “best practices”:

- “Use RBAC”
- “Use NetworkPolicies”
- “Don’t run containers as root”

All true. But if you run a real platform with real customers, you’ve probably asked yourself a different question:

> **“If someone *really* looked at our Kubernetes setup – an auditor, a customer, or an attacker – could we defend it?”**

Tools alone won’t answer that.

In this post we’ll look at why a **clear, opinionated security baseline for your Kubernetes platform** is more valuable than “one more scanner, one more dashboard, one more product”.

We’ll focus mainly on **SaaS and digital platform teams** running on Kubernetes, but most of this applies just as well to healthcare, government and high‑tech.

---

## Tools don’t fix the lack of a standard

A typical Kubernetes landscape in 2025:

- Multiple clusters (dev/test/acc/prod)
- Multiple cloud accounts/subscriptions
- Multiple teams deploying things
- Multiple tools for security, logging and monitoring

And still:

- Nobody can summarize in one sentence **“what normal looks like”** in terms of access, network and secrets.
- Nobody can show **“this is how it’s supposed to be”** and **“here is where we deviate”**.
- Security work is driven by **incidents and ad-hoc findings**, not by a clear standard.

In that situation, the problem is usually not “we need another tool”, but:

> **“We don’t have a shared, enforced minimum standard for our platform.”**

That minimum standard is what we call a **Kubernetes security baseline**.

---

## What is a Kubernetes security baseline?

A security baseline is not a 70‑page PDF.  
It’s a **set of technical opinions, written down and applied in your platform**:

- _How_ things must be configured
- _How_ you prove that
- _How_ you detect drifts and exceptions

For Kubernetes this typically covers:

### 1. Identity & access (RBAC)

- Who is allowed to be **cluster-admin** (and more importantly: who isn’t)?
- What roles do we use by default for:
  - human users,
  - service accounts,
  - CI/CD pipelines?
- How do we handle `kubectl` access and kubeconfigs?

### 2. Network & segmentation

- How do we separate **environments** (dev/test/acc/prod)?
- Which namespaces may talk to which?
- Which traffic must go through ingress or an API gateway?
- What is our default stance: open by default or **deny by default**?

### 3. Secrets & configuration

- Where do secrets live?  
  Kubernetes Secrets? External secret stores?
- How do we prevent secrets from ending up in Git?
- Which configuration is allowed in plain text manifests and which is not?

### 4. Workload hardening

- What are our default resource limits/requests?
- Are containers allowed to run as root? If so: where and why?
- What is our policy on:
  - privileged pods,
  - hostPath mounts,
  - extra Linux capabilities?

### 5. Logging & detection

- Which logs must we have **by design** (control plane, audit, ingress, workloads)?
- Where do those logs end up (cloud logging, SIEM, Wazuh, …)?
- Which security use cases must we at least be able to detect?

The baseline expresses all of this in a clear way:

> **“This is our default. If we deviate, we do so on purpose and we document why.”**

---

## Why this matters for audits (ISO, SOC2, NIS2, …)

Most compliance frameworks in the end ask three practical questions:

1. Have you identified your risks?
2. Have you implemented controls?
3. Can you show that they’re actually in place?

With a Kubernetes security baseline you can:

- **Link risks to technical controls**  
  (e.g. “abuse of broad admin rights → strict RBAC + audit on RBAC changes”)
- **Show the controls in the live configuration**, not just in Word
- **Demonstrate how you keep the baseline in place**  
  through policies, automation and monitoring

That feels very different to an auditor than:

> “We use this tool and here is a screenshot.”

Instead you can show:

- **“This is our baseline.”**
- **“These controls implement it in the cluster.”**
- **“These checks and reports show where we drift and how we correct.”**

Exactly what you need for **ISO27001**, **SOC2** or **NIS2** when your product is built on Kubernetes.

---

## From “we hope it’s fine” to a baseline: the journey

Every organization is different, but the path is remarkably similar:

### Step 1 – As‑is: how does it actually run?

Not in diagrams, but in the cluster:

- Which roles and role bindings exist?
- Who is effectively cluster-admin?
- How are namespaces structured?
- Which NetworkPolicies exist (if any)?
- How do critical workloads actually run (privileges, mounts, secrets)?
- What do we log today – and where does it go?

This typically reveals a few surprises:

- accounts with far more access than anyone realized,
- workloads with more privileges than strictly needed,
- namespaces that are effectively flat networks.

### Step 2 – To‑be: what would make sense for *your* platform?

Here we sketch a **realistic** target architecture:

- environment separation (dev/test/acc/prod),
- namespace strategy,
- access model (RBAC, service accounts, CI/CD),
- network and ingress layout,
- logging and monitoring layer.

Important: not a perfect textbook diagram, but a design that:

- fits your **team structure**,
- fits your **roadmap**,
- is actually **operable**.

### Step 3 – Define the baseline

For each domain (RBAC, network, secrets, workloads, logging) we define:

- which settings are “normal”,
- which are explicitly banned,
- where exceptions are allowed (and how we handle them).

Some of this will be written down, but wherever possible we encode it in:

- Kubernetes manifests (Helm, Kustomize),
- policies (Gatekeeper/Kyverno),
- CI/CD checks,
- scripts and automation.

### Step 4 – Pilot: one cluster really on baseline

Instead of “fixing everything at once”, we pick:

- one cluster (or one environment),
- a set of critical workloads,

and bring those **fully in line** with the baseline.

That gives immediate feedback:

- where the baseline is too strict,
- where it is too vague,
- which processes you need (e.g. for exceptions and approvals).

### Step 5 – Scale & embed

Only after the pilot do we:

- roll the baseline out to more clusters/environments,
- move checks into CI/CD,
- wire logging and monitoring into the baseline.

From there, you can start thinking about:

- managed platform security,
- continuous baseline monitoring,
- deeper detection & response.

First a solid baseline, then continuous improvement.

---

## What this means for your team

If you run a SaaS product or digital platform on Kubernetes, a clear baseline is more than a “nice to have”.

It is a prerequisite for:

- **Scaling your platform safely**  
  Onboarding new services and teams without reinventing security every time.

- **Onboarding engineers**  
  New people know what “normal” is from day one.

- **Enterprise & compliance conversations**  
  When customers, auditors or partners ask “How is your platform secured?”, you have a story backed by actual config.

- **Sleeping at night**  
  You don’t just hope things are fine – you know:
  > “We have a standard. We measure against it. We correct when we drift.”

---

## Where you can start tomorrow

You don’t need a full-blown project to get value. A few concrete starting points:

1. **List your cluster admins**  
   Can you name them all? Do they all really need that level of access?

2. **Write down how you want to use namespaces**  
   Per environment, what types of namespaces do you want (platform vs app vs shared)?

3. **Pick one way to manage secrets**  
   And document what is acceptable and what’s not (e.g. no secrets in plain text in Git).

4. **Decide what you minimally want to log**  
   RBAC changes, namespace and network changes, ingress traffic, etc.

5. **Schedule a baseline session with your platform team**  
   No tool talk. Just one question: “What should our minimum standard look like?”

---

## How Sudo can help

At Sudo Cybersecurity, we work with platform and engineering teams that recognize this situation:

- Kubernetes is in production.
- There is pressure from customers and frameworks like ISO, SOC2, NIS2.
- But there is no strong, coherent story about platform security yet.

That’s exactly why we created our **Kubernetes Security Review & Hardening** engagement:

- we map the current state of your clusters,
- define a realistic security baseline for your platform,
- bring a pilot environment in line with that baseline,
- and deliver a **concrete backlog** that your own team can pick up.

Whether you do this yourself or with us, the goal is the same:

> Move from “we hope it’s secure” to “we know what our standard is – and we can prove it.”
