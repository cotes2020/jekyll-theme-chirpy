---
title: OUTRAY's Race Condition 
date: 2026-1-13 13:10:30 +0000
categories: [Security Research, CVES]
tags: [Security Research, CVES]
math: true
mermaid: true
media_subpath: /assets/posts/cves/outray
image:
  path: preview.png
---
### Summary
This vulnerability allows a user i.e a free plan user to get more than the desired subdomains due to lack of db transaction lock mechanisms in `https://github.com/akinloluwami/outray/blob/main/apps/web/src/routes/api/%24orgSlug/subdomains/index.ts`

### Details
- The affected code-:

```typescript
//Race condition
        const [subscription] = await db
          .select()
          .from(subscriptions)
          .where(eq(subscriptions.organizationId, organization.id));

        const currentPlan = subscription?.plan || "free";
        const planLimits = getPlanLimits(currentPlan as any);
        const subdomainLimit = planLimits.maxSubdomains;

        const existingSubdomains = await db
          .select()
          .from(subdomains)
          .where(eq(subdomains.organizationId, organization.id));

        if (existingSubdomains.length >= subdomainLimit) {
          return json(
            {
              error: `Subdomain limit reached. The ${currentPlan} plan allows ${subdomainLimit} subdomain${subdomainLimit > 1 ? "s" : ""}.`,
            },
            { status: 403 },
          );
        }

        const existing = await db
          .select()
          .from(subdomains)
          .where(eq(subdomains.subdomain, subdomain))
          .limit(1);

        if (existing.length > 0) {
          return json({ error: "Subdomain already taken" }, { status: 409 });
        }

        const [newSubdomain] = await db
          .insert(subdomains)
          .values({
            id: crypto.randomUUID(),
            subdomain,
            organizationId: organization.id,
            userId: session.user.id,
          })
          .returning();
```

- The first part of the code checks the user plan and determine his/her existing_domains without locking the transaction and allowing it to run.
```typescript
const existingSubdomains = await db
          .select()
          .from(subdomains)
          .where(eq(subdomains.organizationId, organization.id));
```

- The other part of the code checks if the desired domain is more than the limit.

```typescript
if (existingSubdomains.length >= subdomainLimit) {
          return json(
            {
              error: `Subdomain limit reached. The ${currentPlan} plan allows ${subdomainLimit} subdomain${subdomainLimit > 1 ? "s" : ""}.`,
            },
            { status: 403 },
          );
        }
```

- Finally, it inserts the subdomain also after the whole check without locking transactions.

```typescript
const [newSubdomain] = await db
          .insert(subdomains)
          .values({
            id: crypto.randomUUID(),
            subdomain,
            organizationId: organization.id,
            userId: session.user.id,
          })
          .returning();
```
- An attacker can exploit this by making parallel requests to the same endpoint and if the second request reads row `subdomains` before the `INSERT` statement  of request one is made.It allows the attacker to act on a not yet updated row which bypasses the checks and allow the attacker to get more subdomains.For example-:

```text
  Parallel request 1                               Parallel  Request  2    
     |                                                                     |
checks for                                                     Checks the not yet updated
available subdomain                                     row and bypasses the logic checks
and determines if it is more than limit
    |                                                                        |
Inserts subdomain and calls it a day           Also inserts the  subdomain
```
-  The attack focuses on exploiting the race window between reading and writing the db rows.

### PoC

- Intercept with Burp proxy,pass to `Repeater` and create multiple requests in a single batch with different subdomain names as seen below. Lastly, send the requests in `parallel`.

<img width="1844" height="855" alt="image" src="https://github.com/user-attachments/assets/f46d5993-31bd-4b96-902a-b2de5b0518bd" />

- Result-:

<img width="1905" height="977" alt="image" src="https://github.com/user-attachments/assets/4c877de2-4b55-46f4-9f1c-78590dfebefc" />


### Impact
The vulnerability provides an infiinite supply of domains to users bypassing the need for subscription

- [Original Advisory](https://github.com/outray-tunnel/outray/security/advisories/GHSA-45hj-9x76-wp9g)