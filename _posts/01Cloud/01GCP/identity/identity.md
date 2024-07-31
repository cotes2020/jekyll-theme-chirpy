



- [Patterns for authenticating corporate users in a hybrid environment](#patterns-for-authenticating-corporate-users-in-a-hybrid-environment)
  - [Introduction](#introduction)
  - [federating an external IdP with Google Cloud](#federating-an-external-idp-with-google-cloud)
    - [Federating `Active Directory` with `Cloud Identity` by using GCDS and AD FS](#federating-active-directory-with-cloud-identity-by-using-gcds-and-ad-fs)
      - [User experience](#user-experience)
      - [Advantages](#advantages)
      - [Best practices](#best-practices)
    - [Federating `Azure AD` with `Cloud Identity`](#federating-azure-ad-with-cloud-identity)
      - [User experience](#user-experience-1)
      - [Advantages](#advantages-1)
      - [Best practices](#best-practices-1)
  - [Patterns for extending an external IdP to Google Cloud](#patterns-for-extending-an-external-idp-to-google-cloud)


# Patterns for authenticating corporate users in a hybrid environment

- to extend your identity management solution to Google Cloud
- to enable your corporate users to authenticate and consume services in a hybrid computing environment.

---


## Introduction

When you extend your IT landscape to Google Cloud as part of a hybrid strategy, we recommend that you take `a consistent approach to managing identities across environments`.

when design and tailor the architecture to meet these constraints and requirements, rely on some common patterns fall into two categories:
* **Patterns for federating an external identity provider (IdP) with GCP**. 将外部身份提供商 (IdP) 与 GCP 联合
  * enable Google to become an IdP for your corporate users
  * so that Google identities are maintained automatically and your IdP remains the source of truth.
* **Patterns for extending an IdP to Google Cloud**. 将 IdP 扩展到 Google Cloud 的模式
  * let applications deployed on Google Cloud reuse your IdP—either
    * by `connecting to it directly`
    * or by `maintaining a replica of your IdP on Google Cloud`.


---

## federating an external IdP with Google Cloud

To enable access to the Cloud Console, the `gcloud` command-line tool, or any other resource that uses Google as IdP, a corporate user must have a Google identity.

Maintaining Google identities for each employee would be cumbersome when all employees already have an account in an IdP.
- By `federating user identities between your IdP and Google Cloud`,
- you can `automate the maintenance of Google accounts`
- and `tie their lifecycle to accounts that exist`.


Federation helps ensure the following:
* Your `IdP remains the single source of truth` for identity management.
* For all user accounts that your IdP manages, or a selected subset of those accounts, a Google Account is created automatically.
* If an account is disabled or deleted in your IdP, the corresponding Google Account is suspended or deleted.
* To `prevent passwords or other credentials from being copied, the act of authenticating a user is delegated to your IdP`.


----



### Federating `Active Directory` with `Cloud Identity` by using GCDS and AD FS

If you use Active Directory as IdP, you can `federate Active Directory with Cloud Identity` by using `Google Cloud Directory Sync (GCDS)` and `Active Directory Federation Services (AD FS)`:

* **Cloud Directory Sync**
  * a free Google-provided tool
  * implements the synchronization process.
  * Cloud Directory Sync communicates with `Google Identity Platform` over `Secure Sockets Layer (SSL)` and usually runs in the existing computing environment.
* **AD FS**
  * provided by Microsoft as part of Windows Server.
  * With AD FS, you can `use Active Directory for federated authentication`.
  * AD FS usually runs in the existing computing environment.


For a variation of this pattern, you can also use `Active Directory Lightweight Directory Services (AD LDS)` or a different LDAP directory with either AD FS or another SAML-compliant IdP.

#### User experience

1.  When you request the protected resource, you are redirected to the Google sign-on screen, which prompts you for your email address.
2.  If the email address is known to be associated with an account that has been synchronized from Active Directory, you are `redirected to AD FS`.
3.  Depending on the configuration of AD FS,
    1.  you might see a sign-on screen prompting for `your Active Directory username and password`.
    2.  Or AD FS might attempt to sign you in automatically based on your `Windows login (IWA).`
4.  When AD FS has authenticated you, you are redirected back to the protected resource.


#### Advantages

* The approach enables a single sign-on experience across on-premises applications and resources on Google Cloud.
* If you configured AD FS to require multi-factor authentication, that configuration automatically applies to Google Cloud.
* `do not need to synchronize passwords or other credentials` to Google.
* Because the Cloud Identity API is publicly accessible, there's no need to set up `hybrid connectivity` between your on-premises network and Google Cloud.


#### Best practices

* `Active Directory` and `Cloud Identity` use a different logical structure.
  * understand the way of mapping domains, identities, and groups suits your situation best.
* `Synchronize groups in addition to users`.
  * With this approach, you can set up IAM so that you can `use group memberships` in Active Directory to control who has access to resources in Google Cloud.
* Deploy and expose AD FS so that corporate users can access it, but don't expose it more than necessary.
  * Although corporate users must be able to access AD FS, there's no requirement for AD FS to be reachable from Google or from any application deployed on Google Cloud.
* Consider enabling Integrated Windows Authentication (IWA) in AD FS
  * to allow users to sign in automatically based on their Windows login.
* If AD FS becomes unavailable,
  * users might not be able to use the Cloud Console or any other resource that uses Google as IdP.
  * So ensure that AD FS and the domain controllers AD FS relies on are deployed and sized to meet your availability objectives.
* If you use Google Cloud to help ensure business continuity,
  * relying on an on-premises AD FS might undermine the intent of using Google Cloud as an independent copy of your deployment.
  * In this case, consider deploying replicas of all relevant systems on Google Cloud:
    * Replicate your Active Directory to Google Cloud and deploy GCDS to run on Google Cloud.
    * Run dedicated AD FS servers on Google Cloud. These servers use the Active Directory domain controllers running on Google Cloud.
    * Configure Cloud Identity to use the AD FS servers deployed on Google Cloud for single sign-on.



---

### Federating `Azure AD` with `Cloud Identity`

If you are a Microsoft Office 365 or Azure customer, you might have connected your on-premises Active Directory to Azure AD.
- If all user accounts that potentially need access to Google Cloud are already being synchronized to Azure AD, you can reuse this integration by federating Cloud Identity with Azure AD, as the following diagram shows.


![AD](https://i.imgur.com/kvMwISp.png)

#### User experience

1.  When you request the protected resource, you are redirected to the Google sign-on screen, which prompts you for your email address.
2.  If the email address is associated with an account that has been synchronized from Azure AD, you are redirected to Azure AD.
3.  Depending on how your on-premises Active Directory is connected to Azure AD, Azure AD might prompt you for a username and password. Or it might redirect you to an on-premises AD FS.
4.  After successfully authenticating with Azure AD, you are redirected back to the protected resource.



#### Advantages

* You don't need to install any additional software on-premises.
* The approach enables a single sign-on experience across Office 365, Azure, and resources on Google Cloud.
* If you configured Azure AD to require multi-factor (MFA) authentication, MFA automatically applies to Google Cloud.
* If your on-premises Active Directory uses multiple domains or forests and you have set up a custom Azure AD Connect configuration to map this structure to an Azure AD tenant, you can take advantage of this integration work.
* You don't need to synchronize passwords or other credentials to Google.
* Because the Cloud Identity API is publicly accessible, there's no need to set up hybrid connectivity between your on-premises network and Google Cloud or between Azure and Google Cloud.
* You can surface the Cloud Console as a tile in the Office 365 portal.



#### Best practices

* Because Azure AD and Cloud Identity use a different logical structure, make sure you understand the differences. Assess which way of mapping domains, identities, and groups suits your situation best. For more detailed information, see federating Google Cloud with Azure AD.
* Synchronize groups in addition to users. With this approach, you can set up IAM so that you can use group memberships in Azure AD to control who has access to resources in Google Cloud.
* If you use Google Cloud to help ensure business continuity, relying on Azure AD for authentication might undermine the intent of using Google Cloud as an independent copy of your deployment.



---


## Patterns for extending an external IdP to Google Cloud

Some of the applications you plan to deploy on Google Cloud might require the use of authentication protocols not offered by Cloud Identity. To support these workloads, you must allow these applications to use your IdP from within Google Cloud.

The following sections describe common patterns for allowing your IdP to be used by workloads deployed on Google Cloud.


```
### Exposing an on-premises AD FS to Google Cloud

If an application requires the use of WS-Trust or WS-Federation, or relies on AD FS-specific features or claims when using OpenID Connect, you can allow the application to directly use AD FS for authentication.

![Application directly using AD&nbsp;FS for authentication](/solutions/images/authenticating-corporate-users-auth-using-ad-fs.svg)

By using AD FS, an application can authenticate a user. However, because authentication is not based on a Google identity, the application won't be able to perform any API calls [on the user's behalf](/docs/authentication/end-user). Instead, any calls to the Google Cloud API must be authenticated [using a service account](/docs/authentication/production).

#### User experience

1.  When you request the protected resource, you are redirected to the ADFS sign-on screen, which prompts you for your email address. If AD FS isn't publicly exposed over the internet, accessing AD FS might require you to be connected to your company network or corporate VPN.
2.  Depending on the configuration of AD FS, you might see a sign-on screen prompting for your Active Directory username and password. Or AD FS might attempt to sign you in automatically based on your Windows login.
3.  When AD FS has authenticated you, you are redirected back to the protected resource.

#### Advantages

* You can use authentication protocols that aren't supported by Cloud Identity, including WS-Trust and WS-Federation.
* If the application has been developed and tested against AD FS, you can avoid risks that might arise from switching the application to use Cloud Identity.
* There's no need to set up [hybrid connectivity](/hybrid-connectivity) between your on-premises network and Google Cloud.

#### Best practices

* Deploy and expose AD FS so that corporate users can access it, but don't expose it more than necessary. Although corporate users must be able to access AD FS, there's no requirement for AD FS to be reachable from Google or from any application deployed on Google Cloud.
* If AD FS becomes unavailable, users might not be able to use the application anymore. Ensure that AD FS and the domain controllers it relies on are deployed and sized to meet your availability objectives.
* Consider refactoring applications that rely on WS-Trust and WS-Federation to use SAML or OpenID Connect instead.
* If the application relies on group information being exposed as claims in `IdTokens` issued by AD FS, consider retrieving group information from a different source such as the [Directory API](https://developers.google.com/admin-sdk/directory/v1/reference/members/get). Querying the Directory API is a privileged operation that requires using a [service account](/compute/docs/access/service-accounts) that is enabled for [Google Workspace domain-wide delegation](https://developers.google.com/admin-sdk/reports/v1/guides/delegation).

### Exposing an on-premises LDAP directory to Google Cloud

Some of your applications might require users to provide their username and password and use these credentials to attempt an LDAP bind operation. If you cannot [modify these applications](/solutions/authenticating-corporate-users-in-a-hybrid-environment#ldap) to use other means such as SAML to perform authentication, you can grant them access to an on-premises LDAP directory.

![Granting users access to an on-premises LDAP directory](/solutions/images/authenticating-corporate-users-on-prem-ldap.svg)

#### Advantages

* You don't need to change your application.

#### Best practices

* Use [Cloud VPN] or [Cloud Interconnect] to establish hybrid connectivity between Google Cloud and your on-premises network so that you don't need to expose the LDAP directory over the internet.
* Verify that the latency introduced by querying an on-premises LDAP directory doesn't negatively impact user experience.
* Ensure that the communication between the application and the LDAP directory is encrypted. You can achieve this encryption by using [Cloud VPN] or by using [Cloud Interconnect] with LDAP/S.
* If the LDAP directory or the private connectivity between Google Cloud and your on-premises network becomes unavailable, users might not be able to use an LDAP-based application anymore. Therefore, ensure that the respective servers are deployed and sized to meet your availability objectives, and consider using [redundant VPN tunnels](/network-connectivity/docs/vpn/concepts/redundant-vpns) or [interconnects](/network-connectivity/docs/interconnect/tutorials/production-level-overview).
* If you use Google Cloud to ensure business continuity, relying on an on-premises LDAP directory might undermine the intent of using Google Cloud as an independent copy of your existing deployment. In this case, consider [replicating the LDAP directory](#replicate_an_on-premises_ldap_directory_to_gcp) to Google Cloud instead.
* If you use Active Directory, consider [running a replica on Google Cloud instead](/solutions/authenticating-corporate-users-in-a-hybrid-environment#heading=h.ps5qjiahyxoj), particularly if you plan to domain-join Windows machines running on Google Cloud to Active Directory.

### Replicating an on-premises LDAP directory to Google Cloud

Replicating an on-premises LDAP directory to Google Cloud is similar to the pattern of [Exposing an on-premises LDAP directory to Google Cloud]. For applications that use LDAP to verify usernames and passwords, the intent of this approach is to be able to run those applications on Google Cloud. Instead of allowing such applications to query your on-premises LDAP directory, you can maintain a replica of the on-premises directory on Google Cloud.

![Maintaining a replica of the on-premises directory on Google Cloud]

#### Advantages

* You don't need to change your application.
* The availability of LDAP-based applications running on Google Cloud doesn't depend on the availability of the on-premises directory or connectivity to the on-premises network. This pattern is well-suited for [business continuity hybrid scenarios].

#### Best practices

* Use [Cloud VPN] or [Cloud Interconnect] to establish hybrid connectivity between Google Cloud and your on-premises network so that you don't need to expose the LDAP directory over the internet.
* Ensure that the replication between the on-premises LDAP directory is conducted over a secure channel.
* Deploy multiple LDAP directory replicas across multiple [zones or regions] to meet your availability objectives. You can use an [internal load balancer] to distribute LDAP connections among multiple replicas deployed in the same region.
* Use a separate Google Cloud project with a [Shared VPC] to deploy LDAP replicas and grant access to this project on a least-privilege basis.

### Extending an on-premises Active Directory to Google Cloud

Some of the workloads that you plan to deploy on Google Cloud might depend on Active Directory Domain Services, for example:

* Windows machines that need to be domain-joined
* Applications that use Kerberos or NTLM for authentication
* Applications that use Active Directory as an LDAP directory to verify usernames and passwords

To support such workloads, you can extend your on-premises Active Directory forest to Google Cloud—for example, by deploying a resource forest to Google Cloud and connecting it to your on-premises Active Directory forest, as in the following diagram.

![connecting a resource forest to your on-premises Active Directory forest](/solutions/images/patterns-for-using-active-directory-in-hybrid-4-synchronized-forests.svg)

For more detail about this approach and other ways to deploy Active Directory in a hybrid environment, see [Patterns for using Active Directory in a hybrid environment](/solutions/patterns-for-using-active-directory-in-a-hybrid-environment).

![Extending your on-premises Active Directory forest to Google Cloud by deploying additional domain controllers on Google Cloud](/solutions/images/authenticating-corporate-users-deploy-additional-controllers.svg)

#### Advantages

* Your workloads can take full advantage of Active Directory, including the ability to join Windows machines to the Active Directory domain.
* The availability of Active Directory-based applications running on Google Cloud doesn't depend on the availability of on-premises resources or connectivity to the on-premises network. The pattern is well-suited for [business continuity hybrid scenarios](/solutions/hybrid-and-multi-cloud-architecture-patterns#business_continuity_hybridmulti-cloud).

#### Best practices

* Use [Cloud VPN] or [Cloud Interconnect] to establish hybrid connectivity between Google Cloud and your on-premises network.
* To minimize communication between Google Cloud and your on-premises network, create a separate Active Directory site for Google Cloud deployments. You can use either a single site per Shared VPC or, to minimize inter-region communication, one site per Shared VPC and region.
* Create a separate Active Directory domain dedicated to resources deployed on Google Cloud and add the domain to the existing forest. Using a separate domain helps reduce replication overhead and partition sizes.
* To increase availability, [deploy at least two domain controllers](/solutions/deploy-fault-tolerant-active-directory-environment), spread over multiple zones. If you use multiple regions, consider deploying domain controllers in each region.
* Use a separate Google Cloud project with a [Shared VPC] to deploy domain controllers and grant access to this project on a least-privilege basis. By [generating a password](/compute/docs/instances/windows/creating-passwords-for-windows-instances) or accessing the [serial console](/compute/docs/instances/interacting-with-serial-console) of domain controller instances, rogue project members might otherwise be able to compromise the domain.
* Consider deploying an AD FS server farm and GCDS on Google Cloud. This approach lets you [federate Active Directory with Cloud Identity](/solutions/federating-gcp-with-azure-active-directory) without depending on the availability of resources or connectivity to the on-premises network.

```
