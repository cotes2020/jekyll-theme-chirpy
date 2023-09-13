---
title: AIML - Security
date: 2023-04-24 11:11:11 -0400
description:
categories: [51AIML]
# img: /assets/img/sample/rabbit.png
tags: [AIML]
---

# AIML - Security

- [AIML - Security](#aiml---security)
  - [OWAPS Top10 for LLM](#owaps-top10-for-llm)
  - [Vendor - Data Security and AI Language Models](#vendor---data-security-and-ai-language-models)
    - [controls](#controls)
    - [Security Practices Review](#security-practices-review)
    - [Contractual Protections 合同的](#contractual-protections-合同的)
      - [Data Processing Addendum (DPA)](#data-processing-addendum-dpa)
      - [Business Associate Agreement (BAA)](#business-associate-agreement-baa)
    - [Customer Data Rights](#customer-data-rights)
      - [Data Retention Policy](#data-retention-policy)
    - [Transparency](#transparency)
    - [Regulatory Compliance](#regulatory-compliance)
    - [Ethics](#ethics)
  - [Hallucinations](#hallucinations)
    - [Hallucinations in Large Language Models](#hallucinations-in-large-language-models)
      - [Using Hallucinations](#using-hallucinations)
    - [Mitigating Hallucinations](#mitigating-hallucinations)

ref:
- [OWAPS Top10 for LLM v1](https://owasp.org/www-project-top-10-for-large-language-model-applications/assets/PDF/OWASP-Top-10-for-LLMs-2023-v1_0.pdf)


---

## OWAPS Top10 for LLM

link:

---


## Vendor - Data Security and AI Language Models

- As a steward of the customer's data, vendor should rigorously reviews the systems and services of the third parties that process data on the behalf (known as "subprocessors").

- Large language models (LLMs) like OpenAI represent an emerging category of subprocessor, which necessitates both applying existing security review criteria and developing new ones to address the unique technology and risks associated with LLMs.

  - vendor should will continuously refine its security and privacy assessments to include these innovative partners and the capabilities they bring to the table.
  - When evaluating an LLM provider, vendor should examines the following security, privacy, and compliance aspects:

    - Security practices review
    - Contractual protections
    - Customer data rights
    - Transparency
    - Regulatory Compliance
    - Ethics


---

### controls

- LLM interfaces and corporate offerings are simple to block and should be blocked, so that the company can choose which ones to allow.
  - a policy is needed on correct usage, and a project should exist to sanction the right ones. Ignoring this latter part will lead to rogue or maverick use of LLMs, so don’t ignore it.


- Over time, new LLM services will pop up to satisfy suppressed demand (like organizations that outright ban and never subscribe to sanctioned 制裁 services).
  - So list of detection will arise for blocking access to new and emergent 新兴的 or even shady LLM services. For that matter, expect cybercriminals to intentionally stand up such services with everything from real models to mechanical Turks behind the scenes.


- People will start using phones and personal systems to interact with LLMs if interaction is prohibited. Motivations for bypassing bans (e.g., performance, time savings, revenue generation, etc.) are there, so it will happen.


- LLM-directed traffic will morph over (not much) time to progressively look a lot like other traffic types. It will rapidly begin to look like new traffic types to get around blocking, behavioral detection, and pattern spotting. It will also migrate to new protocols.
  - QUIC is an obvious choice, but even traffic shape could look very different if resistance to the use of these services is high.


- Other services will act as proxies and connect via API/WS to GPT, meaning anything could be a gateway. And worse, many users may not know where they are in the “API supply chain,” especially in “human interaction” services like support services.
  - companies should always flag their own services as having humans or machines with a flag for end users, to encourage similar behaviors (an example where a standard is a great thing to champion).


- Part of the problem is keeping IP and specific information out of third-party hands, data loss protection (DLP) can help there. But there are also `traffic analysis attacks` and `data analysis attacks` (especially at scale) and information that can be inferred from LLM use.
  - The data never has to leave to imply something about what exists or is happening within a company’s walls (The big lesson of the Big Data era was that `Big Data stores can create PII and other data types`. This is even worse than that).
  - This means that the formerly carbon-based to carbon-based life form rules about secrecy now have to apply to carbon-to-silicon interactions too: Loose lips sink ships when talking to anything!







---

### Security Practices Review
- Each subprocessor is subject to a `comprehensive security evaluation to obtain documented evidence of its commitment to implementing robust measures` for safeguarding its systems and protecting customer data.

- vendor should looks for vendors to demonstrate mature information security practices at least as high as vendor should’s own program, such as SOC2, ISO 27001, ISO 27017, ISO 27018, third-party auditing, penetration testing, and responsiveness to industry security events.

---

### Contractual Protections 合同的

- Before transferring customer data to any subprocessor, vendor should requires contractual protections in `place permitting and placing explicit limits on the processing carried out on that data`. A **Data Processing Addendum (DPA)** provides vendor and its customers enforceable assurances, including that
  - the appropriate security practices will remain in place,
  - customer data will be processed only for specified purposes,
  - the subprocessor will enforce the same level of rigor on any partners it relies on.

Where available, vendor should also enters into a **Business Associate Agreement (BAA)** to provide access for customers that may work with healthcare data such as protected health information (PHI).

---

#### Data Processing Addendum (DPA)

为了保证消费者数据的安全，欧盟 (EU) 实施了严格的隐私和安全法，称为通用数据保护条例 `European Union General Data Protection Regulation (GDPR)`. GDPR 定义并强制执行欧盟公民对其个人数据的权利。它在使用该数据时实施了问责制、安全性和透明度标准。


如何保持 GDPR 合规性。
- 实施 `数据处理协议 (DPA)`。GDPR 数据处理协议规定了与数据处理活动相关的细节、规则、权利和义务。它有助于确保公司合规、保护数据并让消费者得到保护和满足。
- TBD...


**GDPR 下的 DPA**
- 数据处理协议 是 `数据控制器` 和将处理其数据的 `数据处理器` 之间签署的合同。它是完全符合 GDPR 要求的。

  - 数据控制者

    - 每个 DPA 协议都发生在数据控制者和数据处理者之间。数据控制者是决定如何以及为何处理个人数据的组织或个人。如果您的公司决定将数据发送给第三方以在其服务器上进行备份，那么您的公司就是数据控制者。

    - 数据控制者的决定性特征是决策权。数据控制者对数据收集的原因和个人数据的处理方式做出总体决定。

    - 在大多数情况下，公司或组织是数据控制者。数据处理器是与公司签订合同的独立实体。如果个人（例如独资经营者或个体经营者）做出有关收集和处理个人数据的决定，则该个人也可能是数据控制者。

  - 数据处理者

    - 数据处理者是为数据控制者处理数据的第三方。在上述场景中，如果您的公司决定将您的数据发送出去进行备份，那么提供备份服务的公司就是数据处理器。

    - 数据处理器可以采用多种形式。它可以是公司、个人或公共机构。相关标准是该个人或实体是否代表数据控制者处理数据。



- DPA 列出了将要发生的处理活动的性质、目的和持续时间。它还指定了要处理的个人数据的类型以及数据所属的个人类别。它定义了控制者将拥有的权利和义务。它可以指定必须采用的技术安全措施的使用，例如一定程度的加密。

- DPA 具有法律约束力，数据控制者和处理者必须遵守它，否则将面临严厉处罚。

- DPA 的主要好处是它确保了数据处理器的资格和可靠性。公司需要知道他们的数据掌握在良好的手中，并且是私密且安全的，不会被窥探。DPA 有助于提供这些保证。

- GDPR 及其 DPA 要求可能会对未来的业务运营产生重大影响。随着个人数据收集变得更加有限，关于数据收集和存储的沟通变得至关重要，以及第三方供应商关系需要更严格的合同，业务交易可能会发生变化。个别公司及其人力资源部门在调整其流程以符合 GDPR 要求时将感受到广泛的影响。

- GDPR 要求的好处是，随着人们对其数据的隐私和保护越来越有信心，信任可能会在企业中蓬勃发展。


**何时需要DPA**

- 根据 GDPR，每当个人或组织将个人数据提供给第三方服务提供商以进行协作服务时，DPA 文件都是强制性的。作为数据处理者的任何一方都必须与数据控制者签署 DPA。

  - 例如，在欧盟，托管网站的服务必须与网站所属的公司签署 DPA。处理个人数据以提供有针对性的消费者营销的公司还必须签署 DPA。

- 以下是其他几个需要 DPA 的常见业务服务和场景：
  - 电子邮件管理外包
  - 财务和工资核算的技术数据处理解决方案
  - 通过物理服务器或云中的数据备份服务
  - 通过外部服务提供商进行数据收集或数字化
  - 处理包含敏感数据的旧硬件

- 在某些情况下，GDPR 可能要求欧洲以外的公司提供 DPA。只要涉及欧盟数据，此要求就会发挥作用。例如，位于加拿大的公司在处理有关欧盟公民的数据时可能需要遵守 DPA 要求。


**什么时候不需要 DPA**

- 一些特定的场景不需要 DPA。它们具有使 DPA 保护变得不必要的内置保护。

  - 与有保密要求的专业团体合作： 在许多职业中，最佳实践是让服务提供商拥有针对特定行业的定制保密协议，涵盖 DPA 所需的所有安全措施和隐私要求。通常使用这些保密协议的一些专业包括法律、税务咨询和财务审计。许多医疗保健服务通常也有自己严格的保密保证。

  - 门户服务： 仅连接人员或实体的服务通常不受 DPA 要求的约束。这些专业的婚介服务是如此短暂，以至于 DPA 几乎没有什么好处。例如，招聘人员就属于这一类。他们只是将寻找工作的人与寻找有才华的新团队成员的公司联系起来。这种情况使得与招聘人员的 DPA 变得不必要。

  - 与收债机构合作： 收债机构可以访问个人财务信息和医疗信息。由于收债机构与原始债权人分开并为自己的利益收取债务，因此它们不受 DPA 要求的约束。如果他们代表原始债权人工作，则收款机构需要签署 DPA。

  - 来自多家公司的联合数据管理： 在某些情况下，公司作为一个团队来管理数据集合。当公司可以共同访问来自供应商、产品或销售线索的数据时，通常会发生这种情况。尽管这些公司可能是竞争对手，但它们出于相同的一般目的使用相同的数据。这种数据使用的规模通常意味着 DPA 不是强制性的。

  - 临床试验：大规模临床药物试验通常不使用 DPA，因为它们需要众多贡献者。医生、研究中心和赞助商都可以访问主题数据，并且他们都根据自己的需要进行不同的处理。收集的数据通常还用于整个临床试验的各种目的。在这些情况下，DPA 通常不适用。

**DPA 文件包括**

- GDPR 规定了根据 GDPR 的 DPA 规则，数据处理者必须承担哪些合同义务。
- 以下是一些必需的 DPA 条款：

1. 数据处理细节的彻底分解

   - DPA 应全面详细说明数据处理的各个方面将如何进行。
   - DPA 应包含有关以下主题的明确信息：

   - 要处理的个人数据类型
   - 数据的主题
   - 数据主体的类别
   - 处理的目的和性质
   - 数据处理的预期持续时间
   - 个人数据处理的法律依据
   - 在处理结束时返回或删除个人数据


2. 数据控制者和处理者的权利和责任

   - 在指定双方的权利和责任时，DPA 确保明确谁控制数据处理。

   - DPA 应明确声明数据处理者必须根据数据控制者的意愿和规范执行处理。它应该指定控制器，而不是处理器，保留对数据及其发生的事情的完全控制。

   - DPA 应指示数据处理者仅根据数据控制者的直接指示处理数据，仅在欧盟法律或成员国法律之一要求时才偏离这些指示。


3. 数据处理者所需的保密措施

   - DPA 应指定数据处理器应遵循的协议，以确保个人数据的机密性。

   - 例如，数据处理者必须要求永久雇员、临时雇员和分包商在开始处理个人数据之前签署保密协议。唯一不需要保密协议的情况是法定义务已经要求处理者确保保密。


4. 信息安全所需的技术和组织协议

   - DPA 应概述数据处理者必须实施的安全措施，包括适当时的以下措施：

   - 数据加密
   - 数据主体假名
   - 用于确保所有数据处理系统的数据机密性、可用性、弹性和安全性的协议
   - 在遭受攻击或破坏后恢复对个人数据的访问的流程
   - 用于测试和评估所有安全措施有效性的常规程序
   - 许多处理器可能希望获得正式认证或制定官方行为准则来证明其实施的协议。此类措施有助于确保其数据处理完全符合 GDPR。



5. 任何分包商合同的条款

   - DPA 还应概述数据处理器必须对其分包商施加的要求。

   - 例如，处理者必须确保遵守这些规则和最佳实践：

   - 仅在数据控制者明确同意和授权的情况下雇用分包商
   - 起草和签署合同，对分包商施加相同的数据安全要求，数据处理器本身必须遵守
   - 确保分包商遵守数据保护要求
   - 通知数据控制者涉及分包商的任何变化，并给控制者时间做出响应


6. 数据处理者的合作义务

   - DPA 应指定数据处理器必须何时以及如何与数据控制器合作。
   - 例如，数据处理器必须合作以帮助解决数据访问请求。处理者还必须合作保护数据主体的隐私和权利，特别是通过满足以下要求：

   - 确保个人数据安全
   - 及时通知当局和数据主体个人数据泄露
   - 根据需要执行数据保护影响评估 (DPIA)
   - 出现严重数据风险时咨询相关部门
   - 数据处理者还必须允许数据控制者在处理过程中进行合规性审计。在审核期间，处理者必须及时向控制者提供所有相关信息，以表明其已履行第 3 条规定的合规义务 28 GDPR 的规定。

   - 最佳做法也是处理者保留其处理活动的记录，以证明其符合 GDPR。



---


#### Business Associate Agreement (BAA)

- BAA是商业助理协议（或合同）。
- 合同按照HIPAA指南保护个人健康信息（PHI）。

- 商業夥伴協議，也被稱為商業夥伴協議，規定了每一方有關PHI的義務。

- HIPAA要求受影響的企業只與保證PHI完全安全的商業夥伴合作。關聯公司和BA必須以合同或其他協議的形式書面表達這些保證。除了受影響的公司外，HHS也可能審查她與BA和分包商的HIPAA合規性。

- 爲了遵守HIPAA法規，公司必須爲他所做的三個等級中的每一個簽訂一份商業夥伴協議（BAA）。這三個等級都要負責保護PHI，所以簽訂協議對雙方都是最好的。

  - 討論商業夥伴或分包商對 PHI 的許可和規定使用。
  - 業務夥伴/分包商明確表示，它將只在法律要求或本條款允許的範圍內，根據協議條款使用她的個人健康保險。
  - 要求商業夥伴或分包商採取合理的預防措施，以防止未經授權使用或披露健康保險。

- 一旦主體實體、商業夥伴和商業夥伴的分包商相互聯繫，必須確保第三方保護她收到的PHI。BA知道簽署的協議要求安全處理PHI。


**HIPAA BAA規則**

- **隱私政策**。
  - 醫療保健計劃、醫療保健交換所和醫療保健提供者被認爲是受隱私法規約束的企業。
  - 受保護實體將與第三方商業夥伴合作，以改善其業務，如果該商業夥伴能確保只將其PHI用於指定目的。

  - 商業夥伴必須保護個人健康信息不被濫用和未經授權的訪問，並協助有關公司遵守數據保護條例。
  - 患者有權根據本政策查看和編輯他們的信息。這必須以書面形式貫穿於商業夥伴協議。

- **網絡安全條款**。
  - 爲保護電子健康保險（PHI根據本條例以電子方式存儲或傳輸），相關組織及其業務夥伴必須採取適當的物理、技術和行政措施。以任何其他形式提交的信息，包括硬拷貝，都不包括在內。

- 一般規則。HITECH在2009年進行了調整，以確保商業夥伴必須遵守其HIPAA，但多線規則加強了這種偏見，並在2013年生效。一旦該規則生效，HIPAA要求其商業夥伴和供應商遵守其PHI，作爲相關實體的保護和指令。所涉及的公司不代表BAA的責任。








---

### Customer Data Rights

- customer data is always customer property.

- As training data plays a crucial role in LLMs, it is essential for customers and partners to `have a clear understanding of the allowable uses of customer data`.

- vendor should requires prospective LLM partners to agree that they will not use vendor customer data for training purposes without proper notice and consent.

- require customer data to be deleted promptly after processing is complete, typically in 30 days or less.

#### Data Retention Policy




---


### Transparency

- On the subprocessors page, we share who we work with, what services they provide, and how to learn more.

- ensure the quality of the partners chosen and the ecosystem that helps provide a service that is reliable, scalable, innovative, and cost-effective.

---

### Regulatory Compliance

- `Where LLMs get their training data and how they use it` are questions under intense consideration by regulators in the US, EU, and elsewhere.

- closely monitor these deliberations to align the service offerings to the evolving legal environment.

---

### Ethics

- As a unified business communications platform built on AI, vendor should has been thinking deeply about ethical practices in AI services

- We hold ourselves to the standard that we want to build products and services that are part of a world we want to live in, and that means considering not just the principles of Security & Safety addressed above, but also aspects like Fairness & Inclusiveness, User-focused Benefit, and Accountability.

- look for these same principles in the prospective partners.






---

## Hallucinations

### Hallucinations in Large Language Models

> Large Language Models (LLMs) are known to have `hallucinations`

hallucinations
- behavior in that the model speaks false knowledge as if it is accurate.

- when a model generates text, it can’t tell if the generation is accurate.
  - A large language model is a trained machine learning model that generates text based on the prompt you provided. The model’s training equipped it with some knowledge derived from the `training data` provided. It is difficult to tell what knowledge a model remembers or what it does not.

In the context of LLMs,
- “hallucination”: a phenomenon where the model generates text that is incorrect, nonsensical, or not real.
- Since LLMs are not databases or search engines, `they would not cite where their response is based on`.
- These models generate text as an extrapolation from the prompt you provided.
- The result of extrapolation is not necessarily supported by any training data, but is the most correlated from the prompt.

- For example
  - build a two-letter bigrams Markov model from some text: Extract a long piece of text, build a table of every pair of neighboring letters and tally the count.
  - “hallucinations in large language models” would produce “HA”, “AL”, “LL”, “LU”, etc. and there is one count of “LU” and two counts of “LA.”
  - when started with a prompt of “L”, you are twice as likely to produce “LA” than “LL” or “LS”.
  - with a prompt of “LA”, you have an equal probability of producing “AL”, “AT”, “AR”, or “AN”.
  - with a prompt of “LAT” and continue this process.
  - Eventually, this model invented a new word that didn’t exist.
  - This is a result of the statistical patterns. You may say customer Markov model hallucinated a spelling.

- Hallucination in LLMs is not much more complex than this, even if the model is much more sophisticated. From a high level, hallucination is caused by limited contextual understanding since the model is obligated to transform the prompt and the training data into an abstraction, in which some information may be lost. Moreover, noise in the training data may also provide a skewed statistical pattern that leads the model to respond in a way you do not expect.


#### Using Hallucinations

- You may consider hallucinations a feature in large language models.

- You want to see the models hallucinate if you want them to be creative.
  - For example, if you ask ChatGPT or other Large Language Models to give you a plot of a fantasy story, you want it not to copy from any existing one but to generate a new character, scene, and storyline. This is possible only if the models are not looking up data that they were trained on.

- you want hallucinations when looking for diversity
  - for example, asking for ideas. It is like asking the models to brainstorm for you. You want to have derivations from the existing ideas that you may find in the training data, but not exactly the same. Hallucinations can help you explore different possibilities.

Many language models have a “temperature” parameter.
- control the temperature in ChatGPT using the API instead of the web interface.
- This is a parameter of randomness. The higher temperature can introduce more hallucinations.


### Mitigating Hallucinations

- Language models are not search engines or databases.
- Hallucinations are unavoidable. What is annoying is that the `models generate text with mistakes that is hard to spot`.

- If the contaminated training data caused the hallucination, you can **clean up the data and retrain the model**.
  - However, most models are too large to train on customer own devices. Even fine-tuning an existing model may be impossible on commodity hardware.

- The best mitigation may be **human intervention in the result**
  - asking the model to regenerate if it went gravely wrong.

- The other solution to avoid hallucinations is **controlled generation**.
  - It means providing enough details and constraints in the prompt to the model.
  - Hence the model has limited freedom to hallucinate.
  - The reason for prompt engineering is to specify the role and scenario to the model to guide the generation, so that it does not hallucinate unbounded.

.
