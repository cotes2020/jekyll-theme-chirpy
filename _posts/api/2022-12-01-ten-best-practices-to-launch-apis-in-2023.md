---
title: 10 Best Practices to Launch APIs in 2023
description: "The guide can help launch successful APIs while gauging the potential issues and keeping things on track." 
tags: ["api", "api best practices", "design"]
category: ["architecture", "api"]
date: 2022-12-01
permalink: '/api/ten-best-practices-to-launch-apis-in-2023/'
counterlink: 'ten-best-practices-to-launch-apis-in-2023/'
image:
  path: https://raw.githubusercontent.com/Gaur4vGaur/traveller/master/images/api/2022-12-01-ten-best-practices-to-launch-apis-in-2023.jpg
  width: 800
  height: 200
---

## Introduction
Application programming interfaces or [APIs](https://en.wikipedia.org/wiki/API){:target="_blank"} are becoming ubiquitous in the digital world, driving Amazon cloud, Google products, and Facebook feature. APIs can enable big organizations to develop a technology platform that can seamlessly connect interdependent systems, data, and people. APIs allow organizations to develop reusable and interoperable systems, not only within the organization but also within the entire community. The article presents ten points to consider for building successful APIs.

Along with the indirect benefits of growing the business, APIs today are leading to direct monetization. As per a [report published by Custom Market Insights](https://www.custommarketinsights.com/report/api-management-market/){:target="_blank"}, the Global API management market size is estimated to reach 41.5 billion by the end of 2030. In 2021, an average organization [earned 27% of the revenue from APIs](https://blogs.mulesoft.com/digital-transformation/the-state-of-integration-2021/){:target="_blank"}. It is not a surprise that big firms such as Salesforce.com, Expedia.com, and ebay.com are earning a major chunk of their revenue from APIs, [as highlighted by HBR](https://hbr.org/2015/01/the-strategic-value-of-apis){:target="_blank"}.

In the past couple of years, since the beginning of the pandemic, it has become even more important to keep the APIs up and running. Thus, it is critical for IT teams to have a well-thought-through API implementation. Below are the ten best practices to build a better API and avoid common mistakes.

## 1. KYC (Know Your Consumers)
Start by understanding the needs of the consumers. One of the mistakes that organizations commit is to create the APIs with preconceived assumptions. These assumptions are usually formed by API usage within the organization, which does not hold well for external users. API Teams should spend some time understanding what exactly external users want. Internal APIs are much more flexible in build and design. However, external APIs are for the public. These APIs need to be stable and backward compatible. Any breaking changes to external APIs can disrupt many consumers.

It would also be practical to understand the domain knowledge of the users, understanding of the related topics, and familiarity with business terms. It assists to create the right API and fitting documentation for the consumers.

## 2. Build Trust With Consumers
APIs are contracts between an organization and external users. These must be reliable, efficient, and effective as per the specifications. Any deviation from the contract is likely to cause hitches for the consumers and will result in a loss of trust. A rigorous testing plan must be prepared to identify any gotchas in the code.

APIs must be tested for features, usability, reliability, and efficiency. Along with that, teams must also execute [performance testing](https://en.wikipedia.org/wiki/Software_performance_testing){:target="_blank"}, [discoverability testing](https://en.wikipedia.org/wiki/Discoverability){:target="_blank"}, and [chaos engineering](https://en.wikipedia.org/wiki/Chaos_engineering){:target="_blank"}. By executing suitable tests, API teams can detect issues early and meet the expectation of the consumers.

## 3. API Versioning
Versioning APIs is a requirement for organizations who wants to evolve their APIs. With advancing API features, breaking changes are inevitable. API versions guards against the interruptions that can be caused by breaking changes. Any breaking change would lead to a new version of the APIs, and the old version can be supported for a grace period. There is a multitude of resources available for API versioning. However, a few approaches are:

- URI versioning:
    + Version by Number, e.g., api.org-name.com/v1
    + Version by domain, e.g., apiv1.org-name.com
    + Version by query string, e.g., api.org-name/operation?version=1

- HTTP header version
    + Version by ‘Accept’ header e.g., Accept: application/vnd.ex+json;version=1.0
    + Version by customer header, e.g., Accept-version: v1

The most common preferred approach to version the APIs is by URI versioning by numbers, e.g., api.org-name.com/v1. Irrespective of whichever version scheme is chosen, it is important to have a version. 

## 4. Make It Secure
[Data security](https://en.wikipedia.org/wiki/Data_security){:target="_blank"} is the top priority for any organization; hence, it is a critical aspect of developing APIs. Exposing APIs increases the attack surface and must not be overlooked by organizations. [According to Gartner](https://www.govevents.com/details/48066/api-security-protect-your-apis-from-attacks-and-data-breaches/){:target="_blank"}, in the year 2022, APIs are the leading attack vector to get unauthorized access to data in enterprise web applications.

APIs must be thoroughly tested for authorization, authentication, undue data exposure, rate limits, security configurations, and other vulnerabilities.

Making the APIs secure is not a one-time job; rather, APIs must be closely monitored for any unknown login attempts, an unusual spike in traffic, or any other abnormal behavior. API teams should always seek advice from security experts before releasing anything to the public.

## 5. Document the Essentails
APIs are built for external developers, partners, and integrators, who need access to the systems. It is the responsibility of the organization to onboard its consumers onto the API platform. Good API documentation can lead to widespread adoption of the platform.

A well-documented API is one where developers can discover the APIs, verify if they can use them, determine the benefits of the APIs, and, finally, can integrate with the APIs. Detailed API guidance should:

- Provide a conceptual overview of APIs.
- API references, which include resources, endpoints, methods, parameters, sample requests, sample responses, and error codes.
- Rate limits and Terms of use.
- Testing strategy
- Timeouts, and
- How to get support.

There are a few ways to document the APIs using RAML, API Blueprint JSON Schema. However, [OpenAPI specification](https://swagger.io/specification/){:target="_blank"} is widely accepted for documenting the APIs.

## 6. Keeping Consumers Informed
Developers don't like surprises; they want to integrate with the APIs, automate everything and keep the applications running. Everyone is working on tight deadlines. No scrum master or developer wants a sprint where they must deprioritize a story because of a broken third-party API. Hence, any API changes should be announced as early as possible.

Consumers want to be well informed about any upcoming features, new versions, deprecations, or service maintenance. They would like to hear about the API roadmap, any call for actions, or changes in scope.

Along with the announcements, the community also expects to release change logs from APIs. Any existing or new consumers can refer to these changes logs to get to know about the history of changes.

## 7. Monitor Health
It is impossible to maintain APIs without monitoring. API metrics can help answer key questions about performance, availability, usage, and functional correctness. Monitoring can help to find out if APIs are performing as expected, gauge the impact of the new features, debug reported issues and identify any possible abuse.

Although logging and monitoring are important, it is a cross-cutting concern. Thus, it should be implemented asynchronously and in a way that does not impact API performance. Nowadays, all cloud providers offer log management tools and dashboards.

## 8. Tutorials
It is a good idea to include a ‘Getting Started’ section for the APIs. The section should guide the consumers through basic examples of requests and responses from the APIs. Code samples provided in the section must be easy to understand and well explained and should cater to a large community. It would help them to get familiarize themselves with the platform, authorization/authentication, and other basic constructs. The easier it is for developers to understand the API, the more likely they are to use it.

Guide the developers to the [API Sandbox](https://smartbear.com/learn/api-design/what-is-an-api-sandbox/){:target="_blank"} if there is one. Developers would be keen to try out the APIs with different variables and parameters before they start integrating with APIs.

## 9. Choose the Right Technology 
Choosing the right technology and patterns for the APIs can enhance the experience of integrators, such as choosing GraphQL over REST for querying complex data. Although [GraphQL](https://graphql.org/){:target="_blank"} addresses certain niches, it still has low adoption than REST. Moreover, finding developers to create GraphQL APIs could be a challenge.

Another choice could be an API management tool. An organization should pick a mature platform that comes with a default developer portal, authorization, security policies, and rate limiting rather than coding these tools. Choosing the right management tool can save both time and money.

Other choices include picking up the right language for the API platform, the appropriate framework, and developer/testing tools. Making the right choices would fast-track API developers and will result in a smoother experience for consumers.

## 10. Consistent Naming 
APIs have a lot of moving parts, such as endpoints, parameters, query strings, headers, and payloads. Choosing a sensible and consistent naming of APIs can drastically reduce the learning time for API integrators. An integrator should be able to guess the names of the resources from the context. APIs should use similar terms for similar resources, for example, employee_id and department_id.

Not following a consistent naming convention would leave a bitter taste in the integrators, and it will become hard for them to keep track of everything. A consistent naming convention is one where integrators should not have to read the documentation again to find out the name of a particular resource.


## Final Thoughts
These ten simple steps have helped me in the past to develop the APIs that are apt for business and close to the end users. The approach has helped me to gauge the potential issues, save time and resources, and assure users that APIs meet the minimum standards.

Releasing the APIs for the end customers could be tiring as it needs a lot of effort, especially if you are tracking everything in your head. Although the list is not exhaustive, hopefully, the guidelines above will remove some of the anxiety.


