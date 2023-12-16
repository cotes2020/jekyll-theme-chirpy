
- [Professional Cloud Security Engineer](#professional-cloud-security-engineer)
  - [Learning Objectives](#learning-objectives)
  - [Concept](#concept)
    - [identity and access management](#identity-and-access-management)
    - [Boundary Security](#boundary-security)
    - [Data Protection](#data-protection)
  - [Question](#question)
    - [IAM](#iam)
    - [Network](#network)
    - [Data Protection](#data-protection-1)


# Professional Cloud Security Engineer

link:
- https://www.coursera.org/learn/preparing-for-your-professional-cloud-security-engineer-journey/home/week/1

---

## Learning Objectives
- Understand the role of a Professional Cloud Security Engineer
  - enables organizations to `design and implement secure workloads and infrastructure` on Google Cloud.
  - understanding of `security best practices and industry security requirements, individual designs, develops, and manages a secure solution` by leveraging Google Cloud security technologies. The
- Learn about the sample scenario used throughout the course.
- Understand the benefits of becoming Google Cloud Certified.
- Understand the course structure and types of learning assessments.

---

## Concept


### identity and access management


Planning Cymbal Bank’s cloud identity and access management

![Screenshot 2023-06-25 at 23.44.37](/assets/img/Screenshot%202023-06-25%20at%2023.44.37.png)

![Screenshot 2023-06-25 at 23.44.52](/assets/img/Screenshot%202023-06-25%20at%2023.44.52_wc48t0cji.png)

![Screenshot 2023-06-25 at 23.45.52](/assets/img/Screenshot%202023-06-25%20at%2023.45.52.png)

![Screenshot 2023-06-25 at 23.46.50](/assets/img/Screenshot%202023-06-25%20at%2023.46.50.png)

![Screenshot 2023-06-25 at 23.47.36](/assets/img/Screenshot%202023-06-25%20at%2023.47.36.png)

![Screenshot 2023-06-25 at 23.47.53](/assets/img/Screenshot%202023-06-25%20at%2023.47.53.png)

![Screenshot 2023-06-25 at 23.48.10](/assets/img/Screenshot%202023-06-25%20at%2023.48.10.png)


### Boundary Security


![Screenshot 2023-07-01 at 23.29.58](/assets/img/Screenshot%202023-07-01%20at%2023.29.58.png)

![Screenshot 2023-07-01 at 23.30.27](/assets/img/Screenshot%202023-07-01%20at%2023.30.27.png)

![Screenshot 2023-07-01 at 23.31.02](/assets/img/Screenshot%202023-07-01%20at%2023.31.02.png)

![Screenshot 2023-07-01 at 23.31.19](/assets/img/Screenshot%202023-07-01%20at%2023.31.19.png)

![Screenshot 2023-07-01 at 23.32.08](/assets/img/Screenshot%202023-07-01%20at%2023.32.08.png)

![Screenshot 2023-07-01 at 23.32.18](/assets/img/Screenshot%202023-07-01%20at%2023.32.18.png)

![Screenshot 2023-07-01 at 23.32.37](/assets/img/Screenshot%202023-07-01%20at%2023.32.37.png)

![Screenshot 2023-07-01 at 23.32.53](/assets/img/Screenshot%202023-07-01%20at%2023.32.53.png)

![Screenshot 2023-07-01 at 23.33.56](/assets/img/Screenshot%202023-07-01%20at%2023.33.56.png)


### Data Protection

![Screenshot 2023-07-02 at 14.23.23](/assets/img/Screenshot%202023-07-02%20at%2014.23.23.png)

![Screenshot 2023-07-02 at 14.23.44](/assets/img/Screenshot%202023-07-02%20at%2014.23.44.png)

![Screenshot 2023-07-02 at 14.24.46](/assets/img/Screenshot%202023-07-02%20at%2014.24.46.png)

![Screenshot 2023-07-02 at 14.25.43](/assets/img/Screenshot%202023-07-02%20at%2014.25.43.png)

![Screenshot 2023-07-02 at 14.26.07](/assets/img/Screenshot%202023-07-02%20at%2014.26.07.png)

![Screenshot 2023-07-02 at 14.26.42](/assets/img/Screenshot%202023-07-02%20at%2014.26.42.png)

![Screenshot 2023-07-02 at 14.27.03](/assets/img/Screenshot%202023-07-02%20at%2014.27.03.png)




---

## Question

### IAM

- Cymbal Bank has acquired a non-banking financial company (NBFC). This NBFC uses Active Directory as their central directory on an on-premises Windows Server. You have been tasked with **migrating all the NBFC users and employee information to Cloud Identity**. What should you do?
  - <font color=blue> No </font>: Run Microsoft System Center Configuration Manager (SCCM) on a Compute Engine instance. <font color=blue> Leave the channel unencrypted </font> because you are in a secure Google Cloud environment. Deploy Google Cloud Directory Sync on the Compute Engine instance. Connect to the on-premises Windows Server environment from the instance, and migrate users to Cloud Identity.
  - <font color=blue> No </font>: Run Configuration Manager on a Compute Engine instance. <font color=blue> Copy </font> the resulting configuration file from this machine onto a new Compute Engine instance to keep the production environment separate from the staging environment. <font color=blue> Leave the channel unencrypted </font> because you are in a secure Google Cloud environment. Deploy Google Cloud Directory Sync on this new instance. Connect to the on-premises Windows Server environment from the new instance, and migrate users to Cloud Identity.
  - <font color=red> YES </font>: Use **<font color=red> Cloud VPN to connect the on-premises network to your Google Cloud environment </font>**. Select an on-premises domain-joined Windows Server. **<font color=red> On the domain-joined Windows Server, run Configuration Manager and Google Cloud Directory Sync </font>**. Use **<font color=red> Cloud VPN’s encrypted channel to transfer users </font>** from the on-premises Active Directory to Cloud Identity.
  - <font color=blue> No </font>: Select an on-premises domain-joined Windows Server. Run Configuration Manager on the domain-joined Windows Server, and `copy` the resulting configuration file to a Compute Engine instance. Run Google Cloud Directory Sync on the Compute Engine instance over the internet, and use Cloud VPN to sync users from the on-premises Active Directory to Cloud Identity.

- Cymbal Bank has certain default permissions and access for their analyst, finance, and teller teams. These teams are organized into groups that have a set of role-based IAM permissions assigned to them. After a recent acquisition of a small bank, you find that `the small bank directly assigns permissions to their employees in IAM.` You have been tasked with applying Cymbal Bank’s organizational structure to the small bank. Employees will need access to Google Cloud services. What should you do?
  - <font color=blue> Nox2</font>: Leave all user permissions <font color=blue> as-is in the small bank’s IAM </font>. <font color=red> Use the Directory API in the Google Workspace Admin SDK </font> to create Google Groups. Use a Python script to allocate users to the Google Groups.
  - <font color=red> YES </font>: <font color=red> Reset all user permissions in the small bank’s IAM </font>. Use Cloud Identity to <font color=red> create dynamic groups for each of the bank’s teams </font>. <font color=red> Use the dynamic groups’ metadata field for team typ </font>e to allocate users to their appropriate group with a Python script.
    - <font color=red> Use Dynamic Groups to create groups based on Identity attributes </font>, such as department, and place the users in a flat hierarchy.
    - <font color=red> Dynamic group metadata </font> helps build the structure to identify the users.
  - <font color=blue> Nox2</font>: <font color=red> Reset all user permissions in the small bank’s IAM </font>. Use Cloud Identity to create the required Google Groups. <font color=blue> Upgrade the Google Groups to Security Groups </font>. Use a Python script to allocate users to the groups.
  - <font color=blue> Nox2</font>: <font color=red> Reset all user permissions in the small bank’s IAM </font>. Use the Directory API in the Google Workspace Admin SDK to create Google Groups. Use a Python script to allocate users to the groups.

- Cymbal Bank leverages Google Cloud storage services, an on-premises Apache Spark Cluster, and a web application hosted on a third-party cloud. The Spark cluster and web application require limited access to Cloud Storage buckets and a Cloud SQL instance for only a few hours per day. You have been tasked with sharing credentials while **minimizing the risk that the credentials will be compromised**. What should you do?
  - <font color=blue> No </font>: Create a service account with appropriate permissions. Authenticate the Spark Cluster and the web application as direct requests and <font color=blue> share the service account key </font>.
  - <font color=red> YES </font>: Create a service account with appropriate permissions. Have the Spark Cluster and the web application authenticate as delegated requests, and <font color=red> share the short-lived service account credential as a JWT </font>.
  - <font color=blue> No </font>: Create a service account with appropriate permissions. Authenticate the Spark Cluster and the web application as a delegated request, and <font color=blue> share the service account key </font>.
  - <font color=blue> No </font>: Create a service account with appropriate permissions. Have the Spark Cluster and the web application authenticate as a direct request, and <font color=blue> share the short-lived service account credentials as XML tokens </font>.

- Cymbal Bank recently discovered service account key misuse in one of the teams during a security audit. As a precaution, going forward you **do not want any team** in your organization to generate new external service account keys. You also want to **restrict every new service account’s usage to its associated Project**. What should you do?
  - <font color=red> YES </font>: Navigate to Organizational policies in the Google Cloud Console. <font color=red> Select your organization </font>. Select iam.disableServiceAccountKeyCreation. Customize the applied to property, and set Enforcement to ‘On’. Click Save. Repeat the process for iam.disableCrossProjectServiceAccountUsage.
  - <font color=blue> No </font>: Run the gcloud resource-manager org-policies enable-enforce command with the constraints iam.disableServiceAccountKeyCreation, and iam.disableCrossProjectServiceAccountUsage <font color=blue> and the Project IDs </font> you want the constraints to apply to.
  - <font color=blue> No </font>: Navigate to Organizational policies in the Google Cloud Console. Select your organization. Select iam.disableServiceAccountKeyCreation. Under Policy Enforcement, select Merge with parent. Click Save. Repeat the process for <font color=blue> iam.disableCrossProjectServiceAccountLienRemoval </font>.
  - <font color=blue> No </font>: Run the gcloud resource-manager org-policies <font color=blue> allow </font> command with the boolean constraints iam.disableServiceAccountKeyCreation and iam.disableCrossProjectServiceAccountUsage with Organization ID.

- Cymbal Bank publishes its APIs through Apigee. Cymbal Bank has recently acquired ABC Corp, which uses a third-party identity provider. You have been tasked with **connecting ABC Corp’s identity provider to Apigee for single sign-on (SSO)**. You need to set up SSO so that Google is the service provider. You also want to **monitor and log high-risk activities**. Which two choices would you select to enable SSO?
  - <font color=red> YES </font>: Use openssl to generate public and private keys. <font color=red> Store the public key in an X.509 certificate </font>, and encrypt using RSA or DSA for SAML. Sign in to the <font color=red> Google Admin console, and under Security, upload the certificate </font>.
  - <font color=blue> No </font>: Use openssl to generate a private key. <font color=blue> Store the private key in an X.509 certificate </font>, and encrypt using AES or DES for SAML. Sign in to the Google Workspace Admin Console and upload the certificate.
  - <font color=blue> No </font>: Use openssl to generate public and private keys. <font color=blue> Store the private key in an X.509 certificate </font>, and encrypt using AES or DES for SAML. Sign in to the Google Admin console, and under Security, upload the certificate.
  - <font color=red> YES </font>: Review Network mapping results, and assign <font color=red> SSO profiles </font> to required users.
  - <font color=blue> No </font>: Review Network mapping results, and assign SAML profiles to required users.

- You are an administrator for Cymbal Bank’s Mobile Development Team. You want to control `how long different users can access the Google Cloud console`, the Cloud SDK, and any applications that require user authorization for Google Cloud scopes without having to reauthenticate. More specifically, you want `users with elevated privileges (project owners and billing administrators) to reauthenticate more frequently` than regular users at the organization level. What should you do?
  - <font color=blue> No </font>: Open all Google Cloud projects that belong to Cymbal Bank’s Mobile Development team. <font color=blue> Find each project’s </font> Google Cloud session control setting, and configure a reauthentication policy that requires reauthentication. Choose the reauthentication frequency from the drop-down list.
  - <font color=red> YES </font>: In the <font color=red> Admin console </font>, select Google Cloud <font color=red> session control and set a reauthentication policy </font> that requires reauthentication. Choose the reauthentication frequency from the drop-down list.
  - <font color=blue> No </font>: Create a custom role for project owners and billing administrators at the organization level in the <font color=blue> Google Cloud console </font>. Add the <font color=blue> reauthenticationRequired </font> permission to this role. `Assign this role to each` project owner and billing administrator.
  - <font color=blue> Nox2</font>: Create a custom role for project owners and billing administrators at the organization level in the <font color=blue> Google Cloud console </font>. Add the <font color=blue> reauthenticationRequired </font> permission to this role. Create a Google Group that contains all billing administrators and project owners. Apply the custom role to the group.

- Cymbal Bank’s organizational hierarchy divides the Organization into departments. The Engineering Department has a ‘product team’ folder. This folder contains folders for each of the bank’s products. Each product folder contains one Google Cloud Project, but more may be added. Each project contains an App Engine deployment. Cymbal Bank has hired a new technical product manager and a new web developer. The **technical product manager must be able to interact with and manage all services in projects that roll up to the Engineering Department folder**. The **web developer needs read-only** access to App Engine configurations and settings for a specific product. How should you provision the new employees’ roles into your hierarchy following principles of least privilege?
  - <font color=blue> No </font>: Assign the Project Editor role <font color=blue> in each individual project </font> to the technical product manager. Assign the Project Editor role<font color=blue> in each individual project </font> to the web developer.
  - <font color=blue> No </font>: Assign the Project Owner role <font color=blue> in each individual project </font> to the technical product manager. Assign the App Engine Deployer role <font color=blue> in each individual project </font> to the web developer.
  - <font color=red> YES </font>: Assign the <font color=red> Project Editor role at the Engineering Department folder level </font> to the technical product manager. Assign the <font color=red> App Engine Deployer role at the specific product’s folder </font> level to the web developer.
  - <font color=blue> No </font>: Assign the Project Editor role at the Engineering Department folder level to the technical product manager. Create a Custom Role in the product folder that the web developer needs access to. Add the appengine.versions.<font color=blue> create </font> and appengine.versions.<font color=blue> delete </font> permissions to that role, and assign it to the web developer.

- Cymbal Bank’s organizational hierarchy divides the Organization into departments. The Engineering Department has a ‘product team’ folder. This folder contains folders for each of the bank’s products. One folder titled “analytics” contains a Google Cloud Project that contains an App Engine deployment and a Cloud SQL instance.  A `team needs specific access to this project`. The team `lead needs full administrative access` to App Engine and Cloud SQL. A `developer must be able to configure and manage` all aspects of App Engine deployments. There is also a `code reviewer who may periodically review the deployed App Engine source code without making any changes`. What types of permissions would you provide to each of these users?
  - <font color=blue> No </font>: Create custom roles for all three user types at the “analytics” folder level. For the team lead, provide all appengine.* and cloudsql.* permissions. For the developer, provide appengine.applications.* and appengine.instances.* permissions. For the code reviewer, provide the appengine.instances.* permissions.
  - <font color=red> YES </font>: Assign the basic <font color=red> App Engine Admin </font> and <font color=red> Cloud SQL Admin </font> roles to the team lead. Assign the ‘App Engine Admin’ role to the developer. Assign the <font color=red> App Engine Code Viewer </font> role to the code reviewer. Assign all these permissions at the analytics project level.
  - <font color=blue> No </font>: Create custom roles for all three user types at the project level. For the team lead, provide all appengine.* and cloudsql.* permissions. For the developer, provide appengine.applications.* and appengine.instances.* permissions. For the code reviewer, provide the appengine.instances.* permissions.
  - <font color=blue> No </font>: Assign the basic ‘Editor’ role to the team lead. Create a custom role for the developer. Provide all appengine.* permissions to the developer. Provide the predefined ‘App Engine Code Viewer’ role to the code reviewer. Assign all these permissions at the “analytics” folder level.

- Cymbal Bank is divided into **separate departments**. Each department is **divided into teams**. Each team **works on a distinct product** that requires Google Cloud resources for development. How would you **design a Google Cloud organization hierarchy** to best match Cymbal Bank’s organization structure and needs?
  - <font color=blue> No </font>: Create an Organization node. Under the Organization node, create Department folders. Under each Department, create Product folders. Under each Product, create Teams folders. In the Teams folder, add Projects.
  - <font color=blue> No </font>: Create an Organization node. Under the Organization node, create Department folders. Under each Department, create Product folders. Add Projects to the Product folders.
  - <font color=blue> No </font>: Create an Organization node. Under the Organization node, create Department folders. Under each Department, create Teams folders. Add Projects to the Teams folders.
  - <font color=red> YES </font>: Create an Organization node. Under the Organization node, create Department folders. Under each Department, create a Teams folder. <font color=red> Under each Team, create Product folders. Add Projects to the Product folders </font>.

- Cymbal Bank has a team of developers and administrators working on different sets of Google Cloud resources. The Bank’s `administrators should be able to access the serial ports on Compute Engine Instances and create service accounts`. `Developers should only be able to access serial ports`. How would you design the organization hierarchy to provide the required access?
  - <font color=blue> Nox2</font>: Deny Serial Port Access and Service Account Creation at the Organization level.
    - Create an ‘admin’ folder and set enforced: false for constraints/compute.disableSerialPortAccess.
    - Create a new ‘dev’ folder inside the ‘admin’ folder, and set enforced: false for constraints/iam.disableServiceAccountCreation.
    - Give developers access to the ‘dev’ folder, and administrators access to the ‘admin’ folder.
  - <font color=red> YES </font>: <font color=red> Deny Serial Port Access and Service Account Creation at the organization level. </font>
    - Create a ‘dev’ folder and set <font color=red> enforced: false for constraints/compute.disableSerialPortAccess </font>.
     - <font color=red> Create a new ‘admin’ folder inside the ‘dev’ folder, </font> and set <font color=red> enforced: false for constraints/iam.disableServiceAccountCreation </font>.
     - <font color=red> Give developers access to the ‘dev’ folder, and administrators access to the ‘admin’ folder. </font>
  - <font color=blue> Nox2</font>: Deny Serial Port Access and Service Account Creation at the organization level.
    - Create a ‘dev’ folder and set enforced: true for constraints/compute.disableSerialPortAccess and enforced: true for constraints/iam.disableServiceAccountCreation.
    - Create a new ‘admin’ folder inside the ‘dev’ folder, and set enforced: false for constraints/iam.disableServiceAccountCreation.
    - Give developers access to the ‘dev’ folder, and administrators access to the ‘admin’ folder.
  - <font color=blue> Nox2</font>: Allow Serial Port Access and Service Account Creation at the organization level.
    - Create a ‘dev’ folder and set enforced: true for constraints/iam.disableServiceAccountCreation.
     - <font color=blue> Create another ‘admin’ folder that inherits from the parent inside the organization node. </font>
    - Give developers access to the ‘dev’ folder, and administrators access to the ‘admin’ folder.




- Which tool will Cymbal Bank use to `synchronize their identities from their on-premise identity management system to Google Cloud`?
  - Active Directory
  - Service Accounts
  - <font color=red> Google Cloud Directory Sync </font>
  - Cloud Identity

- Which feature of Google Cloud will Cymbal Bank use to `control the source locations and times that authorized identities will be able to access resources`?
  - <font color=red> IAM Conditions </font>
  - <font color=blue> IAM Roles </font>: necessary to authorize identities to access resources, but <font color=blue> can’t be used alone </font> to control when or from where the authorized identities can access the resources.
  - <font color=blue> Service Accounts </font>: Service Accounts are service identities in Google Cloud, and can’t be used to control when or from where authorized identities can access resources.
  - Identity-aware Proxy


### Network





> <font color=blue> Rate-based-ban would be helpful to disable the incoming services for a time period </font>.
> Error <font color=blue> 403 is incorrect; it indicates invalid authorization </font>

- Cymbal Bank has published an API that `internal teams will use` through the HTTPS load balancer. You need to limit the API usage to `200 calls` every hour. Any exceeding usage should `inform the users that servers are busy`. Which gcloud command would you run to throttle the load balancing for the given specification?

  - gcloud compute security-policies rules create priority
    --security-policy sec-policy
    --src-ip-ranges=<font color=red> source-range </font>
    --action=<font color=red> throttle </font>
    --rate-limit-threshold-count=200
    --rate-limit-threshold-interval-sec=3600
    --conform-action=<font color=red> allow </font>
    --exceed-action=deny-429
    --enforce-on-key=HTTP-HEADER

  - gcloud compute security-policies rules create priority
    --security-policy sec-policy
    --src-ip-ranges=source-range
    --action=<font color=blue> rate-based-ban </font>
    --rate-limit-threshold-count=200
    --rate-limit-threshold-interval-sec=3600
    --conform-action=deny
    --exceed-action=deny-403
    --enforce-on-key=HTTP-HEADER

  - gcloud compute security-policies rules create priority
    --security-policy sec-policy
    --src-ip-ranges=source-range
    --action=<font color=blue> rate-based-ban </font>
    --rate-limit-threshold-count=200
    --rate-limit-threshold-interval-sec=3600
    --conform-action=allow
    --exceed-action=deny-500
    --enforce-on-key=IP

  - gcloud compute security-policies rules create priority
    --security-policy sec-policy
    --src-ip-ranges=source-range
    --action=throttle
    --rate-limit-threshold-count=200
    --rate-limit-threshold-interval-sec=60
    --conform-action=<font color=blue> deny </font>
    --exceed-action=deny-404
    --enforce-on-key=HTTP-HEADER



- Cymbal Bank is releasing a new loan management application using a Compute Engine managed instance group. External `users will connect to the application using a domain name or IP address protected with TLS 1.2`. A load balancer already hosts this application and preserves the source IP address. You are tasked with `setting up the SSL certificate for this load balancer`. What should you do?

  - Create a <font color=red> Google-managed SSL certificate </font>.
    Attach a global <font color=red> static </font> external IP address to the external HTTPS load balancer.
    Validate that an existing <font color=red> URL map will route the incoming service to your managed instance group backend </font>.
    Load your certificate and create an <font color=red> HTTPS proxy </font> routing to your URL map.
    <font color=red> Create a global forwarding rule that routes incoming requests to the proxy </font>.

  - Create a Google-managed SSL certificate.
    Attach a global <font color=blue> dynamic </font> external IP address to the internal HTTPS load balancer.
    Validate that an existing URL map will route the incoming service to your managed instance group backend.
    Load your certificate and create an HTTPS proxy routing to your URL map.
    Create a global forwarding rule that routes incoming requests to the proxy.

  - Import a <font color=blue> self-managed SSL certificate </font>.
    Attach a global static external IP address to the SSL Proxy load balancer.
    Validate that an existing URL map will route the incoming service to your managed instance group backend.
    Load your certificate and create an <font color=blue> SSL proxy </font> routing to your URL map.
    Create a global forwarding rule that routes incoming requests to the proxy.

  - Import a <font color=blue> self-managed SSL certificate </font>.
    Attach a global static external IP address to the TCP Proxy load balancer.
    Validate that an existing URL map will route the incoming service to your managed instance group backend.
    Load your certificate and create a <font color=blue> TCP proxy </font> routing to your URL map.
    Create a global forwarding rule that routes incoming requests to the proxy.



> IAP TCP forwarding establishes an encrypted tunnel that supports both SSH and RDP requests.

- Your organization has a website running on Compute Engine. This instance only has a private IP address. You need to `provide SSH access to an on-premises developer who will debug the website from the authorized on-premises location only`. How do you enable this?

  - Use SOCKS proxy over SSH. Set up an SSH tunnel to one of the hosts in the network. Create the SOCKS proxy on the client side.

  - Set up Cloud VPN. Set up an unencrypted tunnel to one of the hosts in the network. Create outbound or egress firewall rules. Use the private IP address to log in using a gcloud ssh command.

  - Use the default VPC’s firewall. Open port 22 for TCP protocol using the Google Cloud Console.

  - Use <font color=red> Identity-Aware Proxy (IAP) </font>. Set up IAP TCP forwarding by creating ingress firewall rules on port 22 for TCP using the gcloud command.





- You have recently joined Cymbal Bank as a cloud engineer. You `created a custom VPC network, selecting to use the automatic subnet creation mode and nothing else`. The default network still exists in your project. You create a new Linux VM instance and select the custom VPC as the network interface. You try to `SSH into your instance, but you are getting a “connection failed” error`. What answer best explains why you cannot SSH into the instance?

  - You should have deleted the default network. When you have multiple VPCs in your project, Compute Engine can’t allow you to connect because overlapping IP ranges prevent the API from establishing a root connection.

  - You should have used custom subnet creation mode. Since the default VPC still exists, automatic mode created subnets in the same regions, which led to overlapping IP addresses.

  - You did not set up any firewall rules on your custom VPC network. While <font color=red> the default VPC comes with a predefined firewall rule that allows SSH traffic, these need to be added to any custom VPCs </font>.

  - You should have used the default network when setting up your instance. While custom networks support instance creation, they should only be used for internal communication.


- Cymbal Bank needs to `connect its employee MongoDB database to a new human resources web application on the same network`. Both the database and the application are autoscaled with the help of Instance templates. As the Security Administrator and Project Editor, you have been tasked with `allowing the application to read port 27017 on the database`. What should you do?

  - <font color=red> Create service accounts for the application and database </font>. Create a firewall rule using:
    gcloud compute firewall-rules create ALLOW_MONGO_DB
    --network network-name
    --allow <font color=red> TCP:27017 </font>
    <font color=red> --source-service-accounts web-application-service-account </font>
    <font color=red> --target-service-accounts database-service-account </font>

  - Create service accounts for the application and database. Create a firewall rule using:
    gcloud compute firewall-rules create ALLOW_MONGO_DB
    --network network-name
    --allow ICMP:27017
    --source-service-accounts web-application-service-account
    --target-service-accounts database-service-account

  - Create user accounts for the application and database. Create a firewall rule using:
    gcloud compute firewall-rules create ALLOW_MONGO_DB
    --network network-name
    --deny UDP:27017
    --source-service-accounts web-application-user-account
    --target-service-accounts database-admin-user-account

  - Create a user account for the database admin and a service account for the application. Create a firewall rule using:
    gcloud compute firewall-rules create ALLOW_MONGO_DB
    --network network-name
    --allow TCP:27017
    --source-service-accounts web-application-service-account
    --target-service-accounts database-admin-user-account


- Cymbal Bank has designed an application to detect credit card fraud that will analyze sensitive information. The application that’s running on a Compute Engine instance is hosted in a new subnet on an existing VPC. Multiple teams who have access to other VMs in the same VPC must access the VM. You want to `configure the access so that unauthorized VMs or users from the internet can’t access the fraud detection VM`. What should you do?

  - Use target filtering. Create two tags called ‘app’ and ‘data’. Assign the ‘app’ tag to the Compute Engine instance hosting the Fraud Detection App (source), and assign the ‘data’ tag to the other Compute Engine instances (target). Create a firewall rule to allow all ingress communication on this tag.

  - Use <font color=blue> target filtering </font>. Create a tag called ‘app’, and assign the tag to both the source and the target. Create a firewall rule to allow all ingress communication on this tag.

  - Use subnet isolation. Create a service account for the fraud detection VM. Create one service account for all the teams’ Compute Engine instances that will access the fraud detection VM.
    Create a new firewall rule using:
    gcloud compute firewall-rules create ACCESS_FRAUD_ENGINE
    --network <network name>
    --allow TCP:80
    --source-service-accounts <one service account for all teams>
    --target-service-accounts <fraud detection engine’s service account>

  - Use <font color=red> subnet isolation </font>. Create a service account for the fraud detection engine. <font color=red> Create service accounts for each of the teams’ Compute Engine instances </font> that will access the engine. Add a firewall rule using:
    gcloud compute firewall-rules create ACCESS_FRAUD_ENGINE
    --network <network name>
    --allow TCP:80
    <font color=red> --source-service-accounts <list of service accounts> </font>
    --target-service-accounts <fraud detection engine’s service account>


> App Engine uses a fixed set of NAT and health check IP address ranges that must be permitted into the VPC.
> Because the charges must be incurred by the credit analysis team, you need to create the connector on the client side.

- The data from Cymbal Bank’s loan applicants resides in a shared VPC. A credit analysis team uses a CRM tool hosted in the App Engine standard environment. You need to `provide credit analysts with access to this data. You want the charges to be incurred by the credit analysis team`. What should you do?


    - Add egress firewall rules to allow SSH and/or RDP ports for the App Engine standard environment in the Shared VPC network.
      - Create a client-side connector in the Service Project using the IP range of the target VPC. Verify that the connector is in a READY state.
      - Create an egress rule on the Shared VPC network to allow the connector using Network Tags or IP ranges.

    - Add egress firewall rules to allow TCP and UDP ports for the App Engine standard environment in the Shared VPC network.
      - Create either a client-side connector in the Service Project or a server-side connector in the Host Project using the IP Range or Project ID of the target VPC. Verify that the connector is in a READY state.
      - Create an egress rule on the Shared VPC network to allow the connector using Network Tags or IP ranges.

    - Add ingress firewall rules to allow NAT and Health Check ranges for the App Engine standard environment in the Shared VPC network.
      - <font color=blue> Create a server-side connector </font> in the Host Project using the Shared VPC Project ID. Verify that the connector is in a READY state.
      - Create an ingress rule on the Shared VPC network to allow the connector using Network Tags or IP ranges.

    - <font color=red> Add ingress firewall rules </font> to allow NAT and Health Check ranges for the App Engine standard environment in the Shared VPC network.
      - <font color=red> Create a client-side connector </font> in the Service Project using the Shared VPC Project ID. Verify that the connector is in a READY state.
      - <font color=red> Create an ingress rule on the Shared VPC network to allow the connector using Network Tags or IP ranges </font>.




- Cymbal Bank’s Customer Details `API runs on a Compute Engine instance with only an internal IP address`. Cymbal Bank’s new branch is co-located outside the Google Cloud points-of-presence (PoPs) and requires a `low-latency way for its on-premises apps to consume the API without exposing the requests to the public internet`. Which solution would you recommend?

  - Use Carrier Peering. Use a service provider to access their enterprise grade infrastructure to connect to the Google Cloud environment.

  - Use a Content Delivery Network (CDN). Establish direct peering with one of Google’s nearby edge-enabled PoPs.

  - Use Dedicated Interconnect. Establish direct peering with one of Google’s nearby edge-enabled PoPs.

  - Use <font color=red> Partner Interconnect </font>. Use a service provider to access their enterprise grade infrastructure to connect to the Google Cloud environment.


- An external audit agency needs to perform a one-time review of Cymbal Bank’s Google Cloud usage. The auditors `should be able to access a Default VPC` containing BigQuery, Cloud Storage, and Compute Engine instances where all the usage information is stored. You have been tasked with `enabling the access from their on-premises environment, which already has a configured VPN`. What should you do?

  - Use Dedicated Interconnect. Configure a VLAN in the auditor's on-premises environment. Use Cloud DNS to create DNS zones and records for restricted.googleapis.com and private.googleapis.com. Set up on-premises routing with Cloud Router. Add custom static routes in the VPC to connect individually to BigQuery, Cloud Storage, and Compute Engine instances.

  - <font color=red> Use a Cloud VPN tunnel </font>.
    - Use Cloud DNS to create DNS zones and records for *.googleapis.com.
    - Set up on-premises routing with Cloud Router. Use Cloud Router custom route advertisements to announce routes for Google Cloud destinations.

  - Use a Cloud VPN tunnel.
    - Use your DNS provider to create DNS zones and records for private.googleapis.com. Connect the DNS provider to your on-premises network. Broadcast the request from the on-premises environment. Use a software-defined firewall to manage incoming and outgoing requests.

  - Use Partner Interconnect. Configure an encrypted tunnel in the auditor's on-premises environment. Use Cloud DNS to create DNS zones and A records for private.googleapis.com.



> Cloud NAT gateways help provide internet access (outbound) without requiring a public IP address.
> Cloud DNS is required for domain name resolution; it cannot decide upon internet access.

- An ecommerce portal uses Google Kubernetes Engine to deploy its recommendation engine in Docker containers. This `cluster instance does not have an external IP address`. You need to `provide internet access to the pods in the Kubernetes cluster`. What configuration would you add?

  - <font color=blue> Cloud DNS </font>, subnet primary IP address range for nodes, and subnet secondary IP address range for pods and services in the cluster

  - Cloud VPN, subnet secondary IP address range for nodes, and subnet secondary IP address range for pods and services in the cluster

  - Nginx load balancer, subnet secondary IP address range for nodes, and subnet secondary IP address range for pods and services in the cluster

  - Cloud NAT gateway, subnet primary IP address range for nodes, and subnet secondary IP address range for pods and services in the cluster


> Identity-Aware Proxy (IAP) provides authentication and authorization for services deployed to Google Cloud.

- Which tool will Cymbal Bank use `to enforce authentication and authorization for services deployed to Google Cloud`?
  - Identity-Aware proxy
  - HTTP(S) load balancer
  - Firewall rules
  - Google Cloud Armor


> Cloud NAT is primarily intended for enabling resources with only internal IP addresses to make requests to the Internet.

- How will Cymbal Bank `enable resources with only internal IP addresses to make requests to the Internet`?
  - Dedicated Interconnect
  - Google private access
  - Shared VPC
  - Cloud NAT



### Data Protection
