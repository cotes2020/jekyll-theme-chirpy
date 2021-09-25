---
title: SecConcept - Secure Coding Review and Analysis
date: 2020-11-11 11:11:11 -0400
categories: [10SecConcept]
tags: [SecConcept]
toc: true
image:
---

[toc]

---

# Secure Coding Review and Analysis

- applications require a <font color=red> last look </font> to ensure that the application and its’ components, are free of security flaws.

secure code review 
- serves to <font color=red> detect all the inconsistencies </font> that weren’t found in other types of security testing
- to ensure the <font color=red> application’s logic and business code </font> is sound.
- Reviews can be done via both <font color=red> manual </font> and <font color=red> automated methods </font>


## benefits

- <font color=red> cut down on time and resources it would take if vulnerabilities were detected after release </font> 
  - The security bugs being looked for during a secure code review can cause countless breaches , resulted in billions of dollars in lost revenue, fines, and abandoned customers.

- <font color=red> focus on finding flaws </font> in areas 
  - Authentication, authorization, security configuration,
  - session management, logging,
  - data validation, error handling, and encryption.

- Code reviewers should be <font color=red> well-versed in the language </font> of the application they’re testing
  - knowledgeable on the secure coding practices and security controls that they need to be looking out for.

- need to understand <font color=red> the full context of the application </font>,
  - including its intended audience and use cases
  - Without that context, code may look secure at first glance, but easily be attacked.
  - Knowing the context by which an app is going to be used and how it will function is the only way to certify that the code adequately protects whatever you’ve relegating to it.

## 5 Tips to a Better Secure Code Review 
1. <font color=red> Produce code review checklists </font> to ensure consistency between reviews and by different developers
   1. all reviewers are working by the same <font color=red> comprehensive checklist </font>. reviewers can forget to certain checks without a well-designed checklist.
   2. enforce time constraints as well as <font color=red> mandatory breaks </font> for manual code reviewers. especially when looking at high value applications.

2. Ensure a <font color=red> positive security culture by not singling out developers </font>
   1. It can be easy, especially with reporting by some tools being able to compare results over time, to point the finger at developers who routinely make the same mistakes. It’s important when building a security culture to refrain from playing the blame game with developers; this only serves to deepen the gap between security and development. Use your findings to help guide your security education and awareness program, using those common mistakes as a jumping off point and relevant examples developers should be looking out for.
   2. Again, developers aren’t going to improve in security if they feel someone’s watching over their shoulder, ready to jump at every mistake made. Facilitate their security awareness in more positive ways and your relationship with the development team, but more importantly the organization in general, will reap the benefits.

3. <font color=red> Review code each time a meaningful change </font> in the code has been introduced
   1. If you have a secure SDLC in place, you understand the value of testing code on a regular basis. Secure code reviews don’t have to wait until just before release. For major applications, we suggest performing manual code reviews when new changes are introduced, saving time and human brainpower by having the app reviewed in chunks.

4. A <font color=red> mix of human review and tool use is best </font> to detect all flaws
   1. Tools aren’t (yet) armed with the mind of a human, and therefore can’t detect issues in the logic of code and are hard-pressed to correctly estimate the risk to the organization if such a flaw is left unfixed in a piece of code. Thus, as we discussed above, a mix of static analysis testing and manual review is the best combination to avoid missing blind spots in the code. Use your teams’ expertise to review more complicated code and valuable areas of the application and rely on automated tools to cover the rest.

5. <font color=red> Continuously monitor and track patterns </font> of insecure code
   1. By tracking repetitive issues you see between reports and applications, help inform future reviews by modifying your secure code review checklist, as well as your AppSec awareness training. Monitoring your code offers great insight into the patterns that could be the cause of certain flaws, and will help you when you’re updating your review guide.


## Top 10 Secure Coding Practices

- Validate input. 
  - Validate input from all untrusted data sources. 
  - Proper input validation can eliminate the vast majority of software vulnerabilities. 
  - Be suspicious of most external data sources, including command line arguments, network interfaces, environmental variables, and user controlled files [Seacord 05].

- Heed 注意 compiler warnings.
  - Compile code using the highest warning level available for your compiler and eliminate warnings by modifying the code [C MSC00-A, C++ MSC00-A]. 
  - Use static and dynamic analysis tools to detect and eliminate additional security flaws.

- Architect and design for security policies.
  - Create a software architecture and design your software to implement and enforce security policies. 
  - For example, if your system requires different privileges at different times, consider dividing the system into distinct intercommunicating subsystems, each with an appropriate privilege set.

- Keep it simple.
  - Keep the design as simple and small as possible [Saltzer 74, Saltzer 75]. 
  - Complex designs increase the likelihood that errors will be made in their implementation, configuration, and use. 
  - Additionally, the effort required to achieve an appropriate level of assurance increases dramatically as security mechanisms become more complex.

- Default deny.
  - make access decisions base on permission rather than exclusion. 
  - by default, access is denied and the protection scheme identifies conditions under which access is permitted [Saltzer 74, Saltzer 75].

- Adhere to the principle of least privilege. 
  - Every process should execute with the least set of privileges necessary to complete the job. 
  - Any elevated permission should only be accessed for the least amount of time required to complete the privileged task. 
  - This approach reduces the opportunities an attacker has to execute arbitrary code with elevated privileges [Saltzer 74, Saltzer 75].

- Sanitize data sent to other systems. 
  - Sanitize all data passed to complex subsystems [C STR02-A] such as <font color=red> command shells, relational databases, and commercial off-the-shelf (COTS) components </font>
  - Attackers may be able to invoke unused functionality in these components through the use of SQL, command, or other injection attacks. 
  - This is not necessarily an input validation problem because the complex subsystem being invoked does not understand the context in which the call is made. 
  - Because the calling process understands the context, it is responsible for sanitizing the data before invoking the subsystem.

- Practice defense in depth. 
  - Manage risk with multiple defensive strategies, so that if one layer of defense turns out to be inadequate, another layer of defense can prevent a security flaw from becoming an exploitable vulnerability and/or limit the consequences of a successful exploit. 
  - For example, combining secure programming techniques with secure runtime environments should reduce the likelihood that vulnerabilities remaining in the code at deployment time can be exploited in the operational environment [Seacord 05].

- Use effective quality assurance techniques. 
  - Good quality assurance techniques can be effective in identifying and eliminating vulnerabilities. Fuzz testing, penetration testing, and source code audits should all be incorporated as part of an effective quality assurance program. Independent security reviews can lead to more secure systems. External reviewers bring an independent perspective; for example, in identifying and correcting invalid assumptions [Seacord 05].

- Adopt a secure coding standard. 
  - Develop and/or apply a secure coding standard for your target development language and platform.