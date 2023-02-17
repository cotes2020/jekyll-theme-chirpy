---
title: What does validate tokens mean in modern authentication?
date: 2021-06-25 00:00
categories: [identity]
tags: [identity,AAD]
---

# Introduction

An important rule when you deal with backend application like webapp or webapi is that **you have to validate each token you receive** but what does it mean? When we check the auth providers documentation like [Microsoft](https://docs.microsoft.com/en-us/azure/active-directory/develop/access-tokens#validating-tokens), [Auth0](https://auth0.com/docs/tokens/access-tokens/validate-access-tokens) or [Okta](https://developer.okta.com/docs/guides/validate-access-tokens/dotnet/overview/#retrieve-the-json-web-keys) , we can see that validating a token means decode the JWT, verify the digital signature and finally validate the claims before "processing" the request. The goal of this article will be to explain all those flows.

{% include note.html content="The focus of this article will be on AAD, but the same actions can be done on other providers." %}

# Decode the JWT

As we’ve seen in the previous articles, JWT is the acronym for JSON Web Token and we use a function to decode the received token on each request. It’s basically a Base64URL encoded composed of three parts:

* **Header** where we will find the:
    * **typ** (type) of token. In this case JWT.
    * **alg** (algorithm) used to sign the token. Today AAD is using RS256 (asymmetric key encryption).
    * **kid** (Key identifier) used to sign the token. This information is important to validate the digital signature (see below).
* **Payload** (see below for more information) where you will find the:
  *   Aud (audience) 
  *   Iss (Issuer)
  *   Exp (token expiration date)
  *   And a lot of other claims ...
*   **Signature** (see below for more information)

In summary, decoding a token is important, we will use it for multiple reasons like:

1.	To test if what you received the a valid JWT
2.	To help in the digital signature validation process
3.	To help to validate the token (various claims)
4.	To help in the authorization check process

If something does not pass, drop the request.

# Validate the signature

Why do we want to validate the digital signature? To answer this question, we will have to understand the basics of cryptography.

{% include note.html content="I’m far to be a guru in cryptography." %}

Let’s start by define what a digital signature is? According to [Wikipedia](https://en.wikipedia.org/wiki/Digital_signature) definition, a digital signature is used to guarantee that a known source generated the message (**non-repudiation**), and that the message was not altered in transit (**integrity**).

Mkay…, how is it done?

No, **to sign**, we use **asymmetric keys** that we **distribute with certificates**. Contrary to data encryption where we use a mix of symmetric and asymmetric keys, with signatures we only need asymmetric keys (RS256). **A private key that has to be kept secret** and a **public one that can be distributed to who need it**. Then, you can use **one of the keys** the **encrypt** the data and the other to **decrypt** it (can be done on both side).

That’s it? Almost!

The last concept that has to be understood is the [one way function](https://en.wikipedia.org/wiki/One-way_function). If we have data x it is easy to calculate f(x) but, on the other hand, knowing the value of f(x) it is quite difficult to calculate the value of x. We usually call it hashing, for example in Powershell we use Get-FileHash to get this value when we want to validate the integrity of a file.

And this is it. With one-way function and asymmetric keys, we **should be able to say with high confidence that the data we received came from a know source and has not been tampered**.

Thank you to jwt.io, I’ve discovered 2 Powershell modules to validate tokens. You can find the first one [here](https://github.com/DigitalAXPP/jwtPS) and the second [there](https://github.com/SP3269/posh-jwt/blob/master/JWT/JWT.psm1). I’ve created my AAD signature part more than a year ago, this is why I’ve decided to keep my code, but if jwt.io trust those modules, why not using them 😊.

So now that we have all the information regarding the digital signature, here how we can explain what happen when a frontend app tries to call a backend app:

![signature](/assets/img/2021-06-25/signature.jpg)

# Validate the token

Now that we know our token is valid, what do we have to do? 

You don’t need to verify all claims, but some of them are **“mandatory”** like:

* **aud** (the audience): the app/client ID for whom this token was issued. Verify it’s your app.
* **Iss** (the issuer): the identity service that created this token. For a single tenant, it should be your tenant, but for multi-tenant app it’s different. Imagine you want to allow only several tenants to consume your app, this is where you can drop the request.
* **Iat** (issue at), **nbf** (not before), **exp** (expired) are simple “dates” related to token validation. Make sure those values are valid compared to a get-date.

**And then it depends on you or your app**. For example, in my previous articles, I’ve enforced the token version in version 2 (ver) or I wanted to validate my request can come from only one clientId (azp). Another claim to validate can be the scp if your application’s authorization is based on scopes instead of app roles. My advice (no idea if it’s a good one or not), **validate what makes you comfortable to use this JWT**.

# Authorization check

Now that you know your token is good to be consumed, do you want to execute your api right away?  It depends… For example, in my latest article, I’ve decided to create a multitier app where the backend api was protected with app roles. In other word, an admin has to assign a role to a user to permit him to use the backend api. 

**Your API is in charge of validating the authorization, not AAD**. AAD makes sure a claim called roles exist in the access token, but the validation of this info has to be done by your API.

And that’s it! Now your API can run with high confidence!

# Conclusion
During this article, we’ve seen that the API oversees a lot of things! Of course, you can run your API right away without verification once you receive a HTTP call, but one day or another bad things will happen. Now to avoid bad surprises, make sure you validate the token you receive before doing the job. Start by validate the JWT, to make sure you won’t execute something without your consent (like SQL injection techniques). Then the JWT signature to make sure the token comes from AAD (non-repudiation) and has not been tampered by someone (integrity).Finally, in the case you implement authorization, make sure people, which are authenticated are allowed to execute the API. I hope it was useful, see you in the next one.
