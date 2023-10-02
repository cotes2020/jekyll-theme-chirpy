---
title: AIML - 8
date: 2023-04-24 11:11:11 -0400
description:
categories: [51AIML]
# img: /assets/img/sample/rabbit.png
tags: [AIML]
---

# OWASP Top 10 for LLM
- [OWASP Top 10 for LLM](#owasp-top-10-for-llm)
    - [LLM08: Excessive Agency 过度代理](#llm08-excessive-agency-过度代理)
      - [Vulnerability Examples](#vulnerability-examples)
      - [Attack Scenario Example](#attack-scenario-example)
      - [Prevention Solution](#prevention-solution)


---

### LLM08: Excessive Agency 过度代理

- An LLM-based system is often granted a degree of agency by its developer - the ability to interface with other systems and undertake actions in response to a prompt.

- The decision over which functions to invoke may also be delegated to an LLM 'agent' to dynamically determine based on input prompt or LLM output.

- `Excessive Agency is the vulnerability` that `enables damaging actions to be performed in response` to unexpected/ambiguous outputs from an LLM
  - regardless of what is causing the LLM to malfunction;
  - be it hallucination/confabulation,
  - direct/indirect prompt injection,
  - malicious plugin,
  - poorly-engineered benign prompts,
  - or just a poorly-performing model

- The root cause of Excessive Agency is typically one or more of: `excessive functionality, excessive permissions or excessive autonomy`.

- Excessive Agency can lead to a broad range of impacts across the confidentiality, integrity and availability spectrum, and is dependent on which systems an LLM-based app is able to interact with.

#### Vulnerability Examples

- `Excessive Functionality`:

  - An LLM agent has access to plugins which `include functions that are not needed for the intended operation of the system`.
    - For example,
    - a developer needs to grant an LLM agent the ability to read documents from a repository, but the 3rd-party plugin they choose to use also includes the ability to modify and delete documents.
    - a plugin may have been trialled during a development phase and dropped in favour of a better alternative, but the original plugin remains available to the LLM agent

  - An LLM plugin with `open-ended functionality fails to properly filter the input instructions` for commands outside what's necessary for the intended operation of the application.
    - E.g., a plugin to run one specific shell command fails to properly prevent other shell commands from being executed

- `Excessive Permissions`:
  - An LLM plugin `has permissions on other systems that are not needed` for the intended operation of the application.
    - E.g., a plugin intended to read data connects to a database server using an identity that not only has SELECT permissions, but also UPDATE, INSERT and DELETE permissions

  - An LLM plugin that is designed to perform operations on behalf of a user accesses downstream systems `with a generic high-privileged identity`.
    - E.g., a plugin to read the current user's document store connects to the document repository with a privileged account that has access to all users' files.


- `Excessive Autonomy`:
  - An LLM-based application or plugin `fails to independently verify and approve high-impact actions`.
    - E.g., a plugin that allows a user's documents to be deleted performs deletions without any confirmation from the user.


#### Attack Scenario Example

- An LLM-based personal assistant app is granted access to an individual’s mailbox via a plugin in order to summarise the content of incoming emails.
  - To achieve this functionality, the email plugin requires the ability to read messages, however the plugin that the system developer has chosen to use also contains functions for sending messages.
  - The LLM is vulnerable to an indirect prompt injection attack, whereby a maliciously-crafted incoming email tricks the LLM into commanding the email plugin to call the 'send message' function to send spam from the user's mailbox.
  - This could be avoided by:
    - (a) eliminating excessive functionality by using a plugin that only offered mail-reading capabilities,
    - (b) eliminating excessive permissions by authenticating to the user's email service via an OAuth session with a read-only scope,
    - (c) eliminating excessive autonomy by requiring the user to manually review and hit 'send' on every mail drafted by the LLM plugin.
    - Alternatively, the damage caused could be reduced by implementing rate limiting on the mail-sending interface.


Reference Links
- [EmbracetheRed:ConfusedDeputyProblem](https://embracethered.com/blog/posts/2023/chatgpt-cross-plugin-request-forgery-and-prompt-injection)
- [NeMo-GuardrailsInterfaceGuidelines](https://github.com/NVIDIA/NeMo-Guardrails/blob/main/docs/security/guidelines)
- [LangChain:Human-approvalfortools](https://python.langchain.com/docs/modules/agents/tools/how_to/human_approval)
- [SimonWillison:DualLLMPattern](https://simonwillison.net/2023/Apr/25/dual-llm- pattern/)



#### Prevention Solution

The following actions can prevent Excessive Agency.

- **Limit the plugins/tools that LLM agents are allowed to call** to only the minimum functions necessary.
  - For example, if an LLM-based system does not require the ability to fetch the contents of a URL then such a plugin should not be offered to the LLM agent

- **Limit the functions that are implemented in LLM plugins/tools to the minimum necessary**.
  - For example, a plugin that accesses a user's mailbox to summarize emails may only require the ability to read emails, so the plugin should not contain other functionality such as deleting or sending messages6

- **Avoid open-ended functions where possible** (e.g., run a shell command, fetch a URL, etc) and use plugins/tools with more granular functionality.
  - For example, an LLM- based app may need to write some output to a file. If this were implemented using a plugin to run a shell function then the scope for undesirable actions is very large (any other shell command could be executed).
  - A more secure alternative would be to build a file-writing plugin that could only support that specific functionality6

- **Limit the permissions that LLM plugins/tools are granted to other systems** the minimum necessary in order to limit the scope of undesirable actions.
  - For example, an LLM agent that uses a product database in order to make purchase recommendations to a customer might only need read access to a 'products' table; it should not have access to other tables, nor the ability to insert, update or delete records. This should be enforced by applying appropriate database permissions for the identity that the LLM plugin uses to connect to the database6

- **Track user authorization and security scope to ensure actions taken on behalf of a user are executed on downstream systems in the context of that specific user**, and with the minimum privileges necessary.
  - For example, an LLM plugin that reads a user's code repo should require the user to authenticate via OAuth and with the minimum scope required6

- **Utilize human-in-the-loop control** to require a human to approve all actions before they are taken.
  - This may be implemented in a downstream system (outside the scope of the LLM application) or within the LLM plugin/tool itself.
  - For example, an LLM-based app that creates and posts social media content on behalf of a user should include a user approval routine within the plugin/tool/API that implements the 'post' operation6

- **Implement authorization in downstream systems rather than relying on an LLM** to decide if an action is allowed or not.
  - When implementing tools/plugins enforce the complete mediation principle so that all requests made to downstream systems via the plugins/tools are validated against security policies.


The following options will not prevent Excessive Agency, but can limit the level of damage caused

- **Log and monitor the activity** of LLM plugins/tools and downstream systems to identify where undesirable actions are taking place, and respond accordingly!

- **Implement rate-limiting** to reduce the number of undesirable actions that can take place within a given time period, increasing the opportunity to discover undesirable actions through monitoring before significant damage can occur.

---
