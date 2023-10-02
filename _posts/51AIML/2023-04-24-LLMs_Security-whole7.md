---
title: AIML - 7
date: 2023-04-24 11:11:11 -0400
description:
categories: [51AIML]
# img: /assets/img/sample/rabbit.png
tags: [AIML]
---

# OWASP Top 10 for LLM
- [OWASP Top 10 for LLM](#owasp-top-10-for-llm)
    - [LLM07: Insecure Plugin Design](#llm07-insecure-plugin-design)
      - [Vulnerability Examples](#vulnerability-examples)
      - [Attack Scenario Examples](#attack-scenario-examples)
      - [Prevention Solution](#prevention-solution)


---

### LLM07: Insecure Plugin Design

LLM plugins
- extensions that, when enabled, are called automatically by the model during user interactions.
- They are `driven by the model`, and there is `no application control over the execution.`
- Furthermore, to deal with context-size limitations, `plugins are likely to implement free-text inputs from the model with no validation or type checking`.
  - This allows a potential attacker to construct a malicious request to the plugin, which could result in a wide range of undesired behaviors, up to and including remote code execution.

- The harm of malicious inputs often depends on `insufficient access controls and the failure to track authorization across plugins`.
  - Inadequate access control allows a plugin to blindly trust other plugins and assume that the end user provided the inputs. Such inadequate access control can enable malicious inputs to have harmful consequences ranging from data exfiltration, remote code execution, and privilege escalation.

- This item focuses on the creation of LLM plugins rather than using third-party plugins, which is covered by [LLM-Supply-Chain-Vulnerabilities].

#### Vulnerability Examples

- A plugin `accepts all parameters in a single text field instead of distinct input parameters`

- A plugin `accepts configuration strings, instead of parameters, that can override entire configuration settings`

- A plugin `accepts raw SQL or programming statements instead of parameters`

- `Authentication is performed without explicit authorization to a particular plugin`

- A plugin `treats all LLM content as being created entirely by the user and performs any requested actions without requiring additional authorization`

#### Attack Scenario Examples

- A plugin accepts a base URL and `instructs the LLM to combine the URL with a query to obtain weather forecasts which are included in handling the user request`.
  - A malicious user can craft a request such that the URL points to a domain they control, which allows them to inject their own content into the LLM system via their domainM

- A plugin `accepts a free-form input into a single field that it does not validate`.
  - An attacker supplies carefully crafted payloads to perform reconnaissance 侦察 from error messages. It then exploits known third-party vulnerabilities to execute code and perform data exfiltration or privilege escalationM

- A plugin used to `retrieve embeddings from a vector store accepts configuration parameters as a connection string without any validation`.
  - This allows an attacker to experiment and access other vector stores by changing names or host parameters and exfiltrate embeddings they should not have access to

- A plugin `accepts SQL WHERE clauses as advanced filters, which are then appended to the filtering SQL`.
  - This allows an attacker to stage a SQL attack

- An attacker uses indirect prompt injection to `exploit an insecure code management plugin with no input validation and weak access control to transfer repository ownership and lock out the user from their repositories.`

Reference Links

- [OpenAI ChatGPT Plugins](https://platform.openai.com/docs/plugins/introduction)
- [OpenAI ChatGPT Plugins - Plugin Flow](https://platform.openai.com/docs/plugins/introduction/plugin-flow)

- [OpenAI ChatGPT Plugins - Authentication](https://platform.openai.com/docs/plugins/authentication/service-level)

- [OpenAI Semantic Search Plugin Sample](https://github.com/openai/chatgpt-retrieval-plugin)

- [Plugin Vulnerabilities: Visit a Website and Have the Source Code Stolen](https://embracethered.com/blog/posts/2023/chatgpt-plugin-vulns-chat-with-code)

- [ChatGPT Plugin Exploit Explained: From Prompt Injection to Accessing Private Data](https://embracethered.com/blog/posts/2023/chatgpt-cross-plugin-request-forgery-and-prompt-injection)

- [OWASP ASVS - 5 Validation, Sanitization and Encoding](https://owasp-aasvs4.readthedocs.io/en/latest/V5.html#validation-sanitization-and-encoding)
- [OWASP ASVS 4.1 General Access Control Design](https://owasp-aasvs4.readthedocs.io/en/latest/V4.1.html#general-access-control-design)
- [OWASP Top 10 API Security Risks – 2023](https://owasp.org/API-Security/editions/2023/en/0x11-t10/)


#### Prevention Solution

- **Plugins should enforce strict parameterized input wherever possible and include type and range checks on inputs**.
  - When this is not possible, a second layer of typed calls should be introduced, parsing requests and applying validation and sanitization.
  - When freeform input must be accepted because of application semantics, it should be carefully inspected to ensure that no potentially harmful methods are being called

- **Plugin developers should apply OWASP’s recommendations in ASVS (Application Security Verification Standard)** to ensure effective input validation and sanitization.

- **Plugins should be inspected and tested thoroughly** to ensure adequate validation. Use `Static Application Security Testing (SAST) scans` as well as `Dynamic and Interactive application testing (DAST, IAST)` in development pipelines

- Plugins should **be designed to minimize the impact of any insecure input parameter exploitation following the OWASP ASVS Access Control Guidelines**.
  - This includes least-privilege access control, exposing as little functionality as possible while still performing its desired function

- Plugins should **use appropriate authentication identities**, such as OAuth2, to apply effective authorization and access control.
  - Additionally, API Keys should be used to provide context for custom authorization decisions which reflect the plugin route rather than the default interactive user

- Require **manual user authorization and confirmation of any action taken by sensitive plugins**

- Plugins are, typically, REST APIs, so developers should **apply OWASP Top 10 API Security Risks – 2023 to minimize generic vulnerabilities**
