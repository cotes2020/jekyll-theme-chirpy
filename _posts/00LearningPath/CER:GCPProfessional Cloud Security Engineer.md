
- [Professional Cloud Security Engineer](#professional-cloud-security-engineer)
  - [Learning Objectives](#learning-objectives)
  - [Question](#question)


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

1. Planning Cymbal Bank’s cloud identity and access management

![Screenshot 2023-06-25 at 23.44.37](/assets/img/Screenshot%202023-06-25%20at%2023.44.37.png)

![Screenshot 2023-06-25 at 23.44.52](/assets/img/Screenshot%202023-06-25%20at%2023.44.52_wc48t0cji.png)

![Screenshot 2023-06-25 at 23.45.52](/assets/img/Screenshot%202023-06-25%20at%2023.45.52.png)

![Screenshot 2023-06-25 at 23.46.50](/assets/img/Screenshot%202023-06-25%20at%2023.46.50.png)

![Screenshot 2023-06-25 at 23.47.36](/assets/img/Screenshot%202023-06-25%20at%2023.47.36.png)

![Screenshot 2023-06-25 at 23.47.53](/assets/img/Screenshot%202023-06-25%20at%2023.47.53.png)

![Screenshot 2023-06-25 at 23.48.10](/assets/img/Screenshot%202023-06-25%20at%2023.48.10.png)

---

## Question

1. Cymbal Bank has acquired a non-banking financial company (NBFC). This NBFC uses Active Directory as their central directory on an on-premises Windows Server. You have been tasked with **migrating all the NBFC users and employee information to Cloud Identity**. What should you do?
   - [No] Run Microsoft System Center Configuration Manager (SCCM) on a Compute Engine instance. <font color=blue> Leave the channel unencrypted </font> because you are in a secure Google Cloud environment. Deploy Google Cloud Directory Sync on the Compute Engine instance. Connect to the on-premises Windows Server environment from the instance, and migrate users to Cloud Identity.
   - [No] Run Configuration Manager on a Compute Engine instance. <font color=blue> Copy </font> the resulting configuration file from this machine onto a new Compute Engine instance to keep the production environment separate from the staging environment. <font color=blue> Leave the channel unencrypted </font> because you are in a secure Google Cloud environment. Deploy Google Cloud Directory Sync on this new instance. Connect to the on-premises Windows Server environment from the new instance, and migrate users to Cloud Identity.
   - [x] Use **<font color=red> Cloud VPN to connect the on-premises network to your Google Cloud environment </font>**. Select an on-premises domain-joined Windows Server. **<font color=red> On the domain-joined Windows Server, run Configuration Manager and Google Cloud Directory Sync </font>**. Use **<font color=red> Cloud VPN’s encrypted channel to transfer users </font>** from the on-premises Active Directory to Cloud Identity.
   - [No] Select an on-premises domain-joined Windows Server. Run Configuration Manager on the domain-joined Windows Server, and `copy` the resulting configuration file to a Compute Engine instance. Run Google Cloud Directory Sync on the Compute Engine instance over the internet, and use Cloud VPN to sync users from the on-premises Active Directory to Cloud Identity.

2. Cymbal Bank has certain default permissions and access for their analyst, finance, and teller teams. These teams are organized into groups that have a set of role-based IAM permissions assigned to them. After a recent acquisition of a small bank, you find that `the small bank directly assigns permissions to their employees in IAM.` You have been tasked with applying Cymbal Bank’s organizational structure to the small bank. Employees will need access to Google Cloud services. What should you do?
   - [**No**] Leave all user permissions <font color=blue> as-is in the small bank’s IAM </font>. <font color=red> Use the Directory API in the Google Workspace Admin SDK </font> to create Google Groups. Use a Python script to allocate users to the Google Groups.
   - [x] <font color=red> Reset all user permissions in the small bank’s IAM </font>. Use Cloud Identity to <font color=red> create dynamic groups for each of the bank’s teams </font>. <font color=red> Use the dynamic groups’ metadata field for team typ </font>e to allocate users to their appropriate group with a Python script.
     - <font color=red> Use Dynamic Groups to create groups based on Identity attributes </font>, such as department, and place the users in a flat hierarchy.
     - <font color=red> Dynamic group metadata </font> helps build the structure to identify the users.
   - [**No**] <font color=red> Reset all user permissions in the small bank’s IAM </font>. Use Cloud Identity to create the required Google Groups. <font color=blue> Upgrade the Google Groups to Security Groups </font>. Use a Python script to allocate users to the groups.
   - [**No**] <font color=red> Reset all user permissions in the small bank’s IAM </font>. Use the Directory API in the Google Workspace Admin SDK to create Google Groups. Use a Python script to allocate users to the groups.

3. Cymbal Bank leverages Google Cloud storage services, an on-premises Apache Spark Cluster, and a web application hosted on a third-party cloud. The Spark cluster and web application require limited access to Cloud Storage buckets and a Cloud SQL instance for only a few hours per day. You have been tasked with sharing credentials while **minimizing the risk that the credentials will be compromised**. What should you do?
   - [No] Create a service account with appropriate permissions. Authenticate the Spark Cluster and the web application as direct requests and <font color=blue> share the service account key </font>.
   - [x] Create a service account with appropriate permissions. Have the Spark Cluster and the web application authenticate as delegated requests, and <font color=red> share the short-lived service account credential as a JWT </font>.
   - [No] Create a service account with appropriate permissions. Authenticate the Spark Cluster and the web application as a delegated request, and <font color=blue> share the service account key </font>.
   - [No] Create a service account with appropriate permissions. Have the Spark Cluster and the web application authenticate as a direct request, and <font color=blue> share the short-lived service account credentials as XML tokens </font>.

4. Cymbal Bank recently discovered service account key misuse in one of the teams during a security audit. As a precaution, going forward you **do not want any team** in your organization to generate new external service account keys. You also want to **restrict every new service account’s usage to its associated Project**. What should you do?
   - [x] Navigate to Organizational policies in the Google Cloud Console. <font color=red> Select your organization </font>. Select iam.disableServiceAccountKeyCreation. Customize the applied to property, and set Enforcement to ‘On’. Click Save. Repeat the process for iam.disableCrossProjectServiceAccountUsage.
   - [No] Run the gcloud resource-manager org-policies enable-enforce command with the constraints iam.disableServiceAccountKeyCreation, and iam.disableCrossProjectServiceAccountUsage <font color=blue> and the Project IDs </font> you want the constraints to apply to.
   - [No] Navigate to Organizational policies in the Google Cloud Console. Select your organization. Select iam.disableServiceAccountKeyCreation. Under Policy Enforcement, select Merge with parent. Click Save. Repeat the process for <font color=blue> iam.disableCrossProjectServiceAccountLienRemoval </font>.
   - [No] Run the gcloud resource-manager org-policies <font color=blue> allow </font> command with the boolean constraints iam.disableServiceAccountKeyCreation and iam.disableCrossProjectServiceAccountUsage with Organization ID.

5. Cymbal Bank publishes its APIs through Apigee. Cymbal Bank has recently acquired ABC Corp, which uses a third-party identity provider. You have been tasked with **connecting ABC Corp’s identity provider to Apigee for single sign-on (SSO)**. You need to set up SSO so that Google is the service provider. You also want to **monitor and log high-risk activities**. Which two choices would you select to enable SSO?
   - [x] Use openssl to generate public and private keys. <font color=red> Store the public key in an X.509 certificate </font>, and encrypt using RSA or DSA for SAML. Sign in to the <font color=red> Google Admin console, and under Security, upload the certificate </font>.
   - [No] Use openssl to generate a private key. <font color=blue> Store the private key in an X.509 certificate </font>, and encrypt using AES or DES for SAML. Sign in to the Google Workspace Admin Console and upload the certificate.
   - [No] Use openssl to generate public and private keys. <font color=blue> Store the private key in an X.509 certificate </font>, and encrypt using AES or DES for SAML. Sign in to the Google Admin console, and under Security, upload the certificate.
   - [x] Review Network mapping results, and assign <font color=red> SSO profiles </font> to required users.
   - [No] Review Network mapping results, and assign SAML profiles to required users.

6. You are an administrator for Cymbal Bank’s Mobile Development Team. You want to control `how long different users can access the Google Cloud console`, the Cloud SDK, and any applications that require user authorization for Google Cloud scopes without having to reauthenticate. More specifically, you want `users with elevated privileges (project owners and billing administrators) to reauthenticate more frequently` than regular users at the organization level. What should you do?
   - [No] Open all Google Cloud projects that belong to Cymbal Bank’s Mobile Development team. <font color=blue> Find each project’s </font> Google Cloud session control setting, and configure a reauthentication policy that requires reauthentication. Choose the reauthentication frequency from the drop-down list.
   - [x] In the <font color=red> Admin console </font>, select Google Cloud <font color=red> session control and set a reauthentication policy </font> that requires reauthentication. Choose the reauthentication frequency from the drop-down list.
   - [No] Create a custom role for project owners and billing administrators at the organization level in the <font color=blue> Google Cloud console </font>. Add the <font color=blue> reauthenticationRequired </font> permission to this role. `Assign this role to each` project owner and billing administrator.
   - [**No**] Create a custom role for project owners and billing administrators at the organization level in the <font color=blue> Google Cloud console </font>. Add the <font color=blue> reauthenticationRequired </font> permission to this role. Create a Google Group that contains all billing administrators and project owners. Apply the custom role to the group.

7. Cymbal Bank’s organizational hierarchy divides the Organization into departments. The Engineering Department has a ‘product team’ folder. This folder contains folders for each of the bank’s products. Each product folder contains one Google Cloud Project, but more may be added. Each project contains an App Engine deployment. Cymbal Bank has hired a new technical product manager and a new web developer. The **technical product manager must be able to interact with and manage all services in projects that roll up to the Engineering Department folder**. The **web developer needs read-only** access to App Engine configurations and settings for a specific product. How should you provision the new employees’ roles into your hierarchy following principles of least privilege?
   - [No] Assign the Project Editor role <font color=blue> in each individual project </font> to the technical product manager. Assign the Project Editor role<font color=blue> in each individual project </font> to the web developer.
   - [No] Assign the Project Owner role <font color=blue> in each individual project </font> to the technical product manager. Assign the App Engine Deployer role <font color=blue> in each individual project </font> to the web developer.
   - [x] Assign the <font color=red> Project Editor role at the Engineering Department folder level </font> to the technical product manager. Assign the <font color=red> App Engine Deployer role at the specific product’s folder </font> level to the web developer.
   - [No] Assign the Project Editor role at the Engineering Department folder level to the technical product manager. Create a Custom Role in the product folder that the web developer needs access to. Add the appengine.versions.<font color=blue> create </font> and appengine.versions.<font color=blue> delete </font> permissions to that role, and assign it to the web developer.

8. Cymbal Bank’s organizational hierarchy divides the Organization into departments. The Engineering Department has a ‘product team’ folder. This folder contains folders for each of the bank’s products. One folder titled “analytics” contains a Google Cloud Project that contains an App Engine deployment and a Cloud SQL instance.  A `team needs specific access to this project`. The team `lead needs full administrative access` to App Engine and Cloud SQL. A `developer must be able to configure and manage` all aspects of App Engine deployments. There is also a `code reviewer who may periodically review the deployed App Engine source code without making any changes`. What types of permissions would you provide to each of these users?
   - [No] Create custom roles for all three user types at the “analytics” folder level. For the team lead, provide all appengine.* and cloudsql.* permissions. For the developer, provide appengine.applications.* and appengine.instances.* permissions. For the code reviewer, provide the appengine.instances.* permissions.
   - [x] Assign the basic <font color=red> App Engine Admin </font> and <font color=red> Cloud SQL Admin </font> roles to the team lead. Assign the ‘App Engine Admin’ role to the developer. Assign the <font color=red> App Engine Code Viewer </font> role to the code reviewer. Assign all these permissions at the analytics project level.
   - [No] Create custom roles for all three user types at the project level. For the team lead, provide all appengine.* and cloudsql.* permissions. For the developer, provide appengine.applications.* and appengine.instances.* permissions. For the code reviewer, provide the appengine.instances.* permissions.
   - [No] Assign the basic ‘Editor’ role to the team lead. Create a custom role for the developer. Provide all appengine.* permissions to the developer. Provide the predefined ‘App Engine Code Viewer’ role to the code reviewer. Assign all these permissions at the “analytics” folder level.

9. Cymbal Bank is divided into **separate departments**. Each department is **divided into teams**. Each team **works on a distinct product** that requires Google Cloud resources for development. How would you **design a Google Cloud organization hierarchy** to best match Cymbal Bank’s organization structure and needs?
   - [No] Create an Organization node. Under the Organization node, create Department folders. Under each Department, create Product folders. Under each Product, create Teams folders. In the Teams folder, add Projects.
   - [No] Create an Organization node. Under the Organization node, create Department folders. Under each Department, create Product folders. Add Projects to the Product folders.
   - [No] Create an Organization node. Under the Organization node, create Department folders. Under each Department, create Teams folders. Add Projects to the Teams folders.
   - [x] Create an Organization node. Under the Organization node, create Department folders. Under each Department, create a Teams folder. <font color=red> Under each Team, create Product folders. Add Projects to the Product folders </font>.

10. Cymbal Bank has a team of developers and administrators working on different sets of Google Cloud resources. The Bank’s `administrators should be able to access the serial ports on Compute Engine Instances and create service accounts`. `Developers should only be able to access serial ports`. How would you design the organization hierarchy to provide the required access?
   - [**No**] <font color=red> Deny Serial Port Access and Service Account Creation at the Organization level. </font>
     - Create an `‘admin’ folder and set enforced: false for constraints/compute.disableSerialPortAccess`.
     - Create a new ‘dev’ folder inside the ‘admin’ folder, and set enforced: false for constraints/iam.disableServiceAccountCreation.
     - <font color=red> Give developers access to the ‘dev’ folder, and administrators access to the ‘admin’ folder. </font>
   - [x] <font color=red> Deny Serial Port Access and Service Account Creation at the organization level. </font>
     - Create a ‘dev’ folder and set <font color=red> enforced: false for constraints/compute.disableSerialPortAccess </font>.
     - <font color=red> Create a new ‘admin’ folder inside the ‘dev’ folder, </font> and set <font color=red> enforced: false for constraints/iam.disableServiceAccountCreation </font>.
     - <font color=red> Give developers access to the ‘dev’ folder, and administrators access to the ‘admin’ folder. </font>
   - [**No**] <font color=red> Deny Serial Port Access and Service Account Creation at the organization level. </font>
     - Create a ‘dev’ folder and set enforced: true for constraints/compute.disableSerialPortAccess and enforced: true for constraints/iam.disableServiceAccountCreation.
     - <font color=red> Create a new ‘admin’ folder inside the ‘dev’ folder, </font> and set enforced: false for constraints/iam.disableServiceAccountCreation.
     - <font color=red> Give developers access to the ‘dev’ folder, and administrators access to the ‘admin’ folder. </font>
   - [**No**] Allow Serial Port Access and Service Account Creation at the organization level.
     - Create a ‘dev’ folder and set enforced: true for constraints/iam.disableServiceAccountCreation.
     - <font color=blue> Create another ‘admin’ folder that inherits from the parent inside the organization node. </font>
     - <font color=red> Give developers access to the ‘dev’ folder, and administrators access to the ‘admin’ folder. </font>


















.
