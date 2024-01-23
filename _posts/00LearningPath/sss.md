






- Cymbal Bank has hired a data analyst team to analyze scanned copies of loan applications. Because this is an external team, Cymbal Bank `does not want to share the name, gender, phone number, or credit card numbers` listed in the scanned copies. You have been tasked with `hiding this PII information while minimizing latency`. What should you do?

    - Use the Cloud Vision API to perform text extraction from scanned images. Redact the text using the Cloud Natural Language API with regular expressions.

    - Use the Cloud Data Loss Prevention (DLP) API to make redact image requests. Provide your project ID, built-in infoTypes, and the scanned copies when you make the requests.

    - Use the Cloud Vision API to perform optical code recognition (OCR) from scanned images. Redact the text using the Cloud Data Loss Prevention (DLP) API with regular expressions.

    - Use the Cloud Vision API to perform optical code recognition (OCR) from scanned images. Redact the text using the Cloud Natural Language API with regular expressions.


- Cymbal Bank needs to statistically predict the days customers delay the payments for loan repayments and credit card repayments. Cymbal Bank `does not want to share the exact dates` a customer has defaulted or made a payment with data analysts. Additionally, you need to `hide the customer name and the customer type, which could be corporate or retail`. How do you `provide the appropriate information to the data analysts`?

  - Generalize all dates to year and month with date shifting. Use a predefined infoType for customer name. Use a custom infoType for customer type with a custom dictionary.

  - Generalize all dates to year and month with date shifting. Use a predefined infoType for customer name. Use a custom infoType for customer type with regular expression.

  - Generalize all dates to year and month with bucketing. Use the built-in infoType for customer name. Use a custom infoType for customer type with regular expression.

  - Generalize all dates to year and month with bucketing. Use the built-in infoType for customer name. Use a custom infoType for customer type with a custom dictionary.



- Cymbal Bank stores customer information in a BigQuery table called ‘Information,’ which belongs to the dataset ‘Customers.’ Various departments of Cymbal Bank, including loan, credit card, and trading, access the information table. Although the data source remains the same, `each department needs to read and analyze separate customers and customer-attributes`. You want a `cost-effective way to configure departmental access to BigQuery to provide optimal performance`. What should you do?

    - Create <font color=blue> separate datasets </font> for each department. Create authorized functions in each dataset to perform required aggregations. Write transformed data to new tables for each department separately. Provide the bigquery.dataViewer role to each department’s required users.

    - Create an authorized dataset in BigQuery’s Explorer panel. Write Customers’ table metadata into a JSON file, and edit the file to add each department’s Project ID and Dataset ID. Provide the bigquery.user role to each department’s required users.

    - Create separate datasets for each department. Create views for each dataset separately. Authorize these views to access the source dataset. Share the datasets with departments. Provide the bigquery.dataViewer role to each department’s required users.

    - Secure data with classification. Open the Data Catalog Taxonomies page in the Google Cloud Console. Create policy tags for required columns and rows. Provide the bigquery.user role to each department’s required users. Provide policy tags access to each department separately.



- Your colleague at Cymbal Bank is a cloud security engineer.
  - She sketches out the following solution to manage her team’s access to application security keys, `What (if any) step does not follow Google Cloud’s best practices for secret management`?
    - 1 - Create 2 projects
    - Project A: Cloud Storage to store secrets
    - Project B: Cloud KMS to manage encryption keys
    - 2 - Store each secret individually in Cloud Storage
    - 3 - Rotate secrets and encryption keys regularly
    - 4 - Protect each bucket by using encryption with Cloud KMS

  - It is not recommended to use Cloud KMS keys to encrypt buckets. Default management is safer and more reliable.

  - Your colleague should cluster her secrets together in Cloud Storage. That way that can be easily accessed by applications.

  - Your colleague should have created one project for the Cloud Storage bucket and one to store the KMS encryption keys. Two projects create an unnecessary burden of IAM management.

  - Your colleague’s proposal follows Google Cloud’s best practices.



- Cymbal Bank has a Cloud SQL instance that must be shared with an external agency. The agency’s developers will be `assigned roles and permissions through a Google Group in Identity and Access Management (IAM)`. The external agency is on an annual contract and will `require a connection string, username, and password to connect to the database`. How would you `configure the group’s access`?

  - Use Cloud Key Management Service. Use the destination IP address and Port attributes to provide access for developers at the external agency. <font color=blue> Remove the IAM access after one year </font> and rotate the shared keys. Add cloudkms.cryptoKeyEncryptorDecryptor role for the group that contains the external developers.

  - Use Secret Manager for the connection string and username, and use Cloud Key Management Service for the password. Use tags to set the expiry period to the timestamp one year from now. Add secretmanager.secretVersionManager and secretmanager.secretAccessor roles for the group that contains external developers.

  - Use Secret Manager. Use the resource attribute to set a key-value pair with key as duration and values as expiry period one year from now. Add secretmanager.viewer role for the group that contains external developers.

  - Use Secret Manager. Use the duration attribute to set the expiry period to one year. Add the secretmanager.secretAccessor role for the group that contains external developers.




- Cymbal Bank wants to deploy an n-tier web application. The frontend must be supported by an App Engine deployment, an API with a Compute Engine instance, and Cloud SQL for a MySQL database. This `application is only supported during working hours, App Engine is disabled, and Compute Engine is stopped`. How would you `enable the infrastructure to access the database`?

  - Use Project metadata to read the current machine’s IP address, and use a startup script to add access to Cloud SQL. Store Cloud SQL’s connection string in Cloud Key Management Service, and store the password in Secret Manager. Store the Username in Project metadata.

  - Use VM metadata to read the current machine’s IP address, and use a gcloud command to add access to Cloud SQL. Store Cloud SQL’s connection string and password in Cloud Key Management Service. Store the Username in Project metadata.

  - Use Project metadata to read the current machine’s IP address and use a gcloud command to add access to Cloud SQL. Store Cloud SQL’s connection string and username in Cloud Key Management Service, and store the password in Secret Manager.

  - Use VM metadata to read the current machine’s IP address and use a startup script to add access to Cloud SQL. Store Cloud SQL’s connection string, username, and password in Secret Manager.






- Cymbal Bank calculates employee incentives on a monthly basis for the sales department and on a quarterly basis for the marketing department. The incentives are released with the next month’s salary. Employee’s performance documents are stored as spreadsheets, which are retained for at least one year for audit. You want to configure the most cost-effective storage for this scenario. What should you do?

  - Import the spreadsheets to Cloud SQL, and create separate tables for Sales and Marketing. For Table Expiration, set 365 days for both tables. Use stored procedures to calculate incentives. Use App Engine cron jobs to run stored procedures monthly for Sales and quarterly for Marketing.

  - Upload the spreadsheets to Cloud Storage. Select the Nearline storage class for the sales department and Coldline storage for the marketing department. Use object lifecycle management rules to set the storage class to Archival after 365 days. Process the data on BigQuery using jobs that run monthly for Sales and quarterly for Marketing.

  - Import the spreadsheets to BigQuery, and create separate tables for Sales and Marketing. Set table expiry rules to 365 days for both tables. Create jobs scheduled to run every quarter for Marketing and every month for Sales.

  - Import the spreadsheets into Cloud Storage and create NoSQL tables. Use App Engine cron jobs to run monthly for Sales and quarterly for Marketing. Use a separate job to delete the data after 1 year.

- Cymbal Bank uses Google Kubernetes Engine (GKE) to deploy its Docker containers. You want to encrypt the boot disk for a cluster running a custom image so that the key rotation is controlled by the Bank. GKE clusters will also generate up to 1024 randomized characters that will be used with the keys with Docker containers. What steps would you take to apply the encryption settings with a dedicated hardware security layer?

  - Create a new GKE cluster with customer-managed encryption and HSM enabled. Deploy the containers to this cluster. Delete the old GKE cluster. Use Cloud HSM to generate random bytes and provide an additional layer of security.

  - Create a new key ring using Cloud Key Management Service. Extract this key to a certificate. Use the kubectl command to update the Kubernetes configuration. Validate using MAC digital signatures, and use a startup script to generate random bytes.

  - Create a new key ring using Cloud Key Management Service. Extract this key to a certificate. Use the Google Cloud Console to update the Kubernetes configuration. Validate using MAC digital signatures, and use a startup script to generate random bytes.

  - In the Google Cloud console, navigate to Google Kubernetes Engine. Select your cluster and the boot node inside the cluster. Enable customer-managed encryption. Use Cloud HSM to generate random bytes and provide an additional layer of security.

- You have recently joined Cymbal Bank as a cloud security engineer. You want to encrypt a connection from a user on the internet to a VM in your development project. This is at the layer 3/4 (network/transport) level and you want to set up user configurable encryption for the in transit network traffic. What architecture choice best suits this use case?

  - Set up an IPsec tunnel. This will allow you to create L3/L4 encryption between a user and a VM instance in her project.

  - Set up transport layer security (TLS). This will encrypt data sent to the Google Front End, and in turn, your VM. This is not setup by default.

  - Set up a managed SSL certificate by configuring a load balancer. By default, this will encrypt at the L3/L4 layer.

  - Set up app layer transport security (ALTS). This is a mutual authentication and transport encryption system developed by Google. This is configured for L3/L4 network connections.

- Cymbal Bank needs to migrate existing loan processing applications to Google Cloud. These applications transform confidential financial information. All the data should be encrypted at all stages, including sharing between sockets and RAM. An integrity test should also be performed every time these instances boot. You need to use Cymbal Bank’s encryption keys to configure the Compute Engine instances. What should you do?

  - Create a Confidential VM instance with Customer-Managed Encryption Keys. In Cloud Logging, collect all logs for earlyBootReportEvent.

  - Create a Shielded VM instance with Customer-Managed Encryption Keys. In Cloud Logging, collect all logs for sevLaunchAttestationReportEvent.

  - Create a Confidential VM instance with Customer-Supplied Encryption Keys. In Cloud Logging, collect all logs for sevLaunchAttestationReportEvent.

  - Create a Shielded VM instance with Customer-Supplied Encryption Keys. In Cloud Logging, collect all logs for earlyBootReportEvent.
  Coursera Honor Code  Learn more
