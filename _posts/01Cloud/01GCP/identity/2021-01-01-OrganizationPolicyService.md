---
title: GCP - Organization Policy Service
date: 2021-01-01 11:11:11 -0400
categories: [01GCP, Identity]
tags: [GCP]
toc: true
image:
---

- [Organization Policy Service](#organization-policy-service)
  - [Benefits](#benefits)
  - [Common use cases](#common-use-cases)
  - [Differences from Identity and Access Management](#differences-from-identity-and-access-management)
  - [Key Concepts](#key-concepts)
    - [Organization policy](#organization-policy)
    - [Constraints](#constraints)
      - [boolean constraints](#boolean-constraints)
      - [list constraints](#list-constraints)
      - [Predefined Constraints](#predefined-constraints)
        - [Constraints supported by multiple Google Cloud services](#constraints-supported-by-multiple-google-cloud-services)
          - [Allowed Worker Pools (Cloud Build)](#allowed-worker-pools-cloud-build)
          - [Disable Enabling Identity-Aware Proxy (IAP) on regional resources](#disable-enabling-identity-aware-proxy-iap-on-regional-resources)
          - [Google Cloud Platform - Resource Location Restriction](#google-cloud-platform---resource-location-restriction)
          - [Restrict allowed Google Cloud APIs and services](#restrict-allowed-google-cloud-apis-and-services)
          - [Restrict Resource Service Usage](#restrict-resource-service-usage)
          - [Restrict which projects may supply KMS CryptoKeys for CMEK](#restrict-which-projects-may-supply-kms-cryptokeys-for-cmek)
          - [Restrict which services may create resources without CMEK](#restrict-which-services-may-create-resources-without-cmek)
      - [Custom constraints](#custom-constraints)
    - [Inheritance](#inheritance)
    - [Violations](#violations)
  - [command](#command)

---

# Organization Policy Service

- give centralized and programmatic control over the organization's cloud resources.

- organization policy administrator can configure constraints across the entire resource hierarchy.

---

## Benefits

- Centralize control to configure restrictions on how the organization’s resources can be used.
- Define and establish guardrails for the development teams to stay within compliance boundaries.
- Help project owners and their teams move quickly without worry of breaking compliance.

---

## Common use cases

Organization policies are made up of constraints that allow you to:

- Limit resource sharing based on domain.
- Limit the usage of Identity and Access Management service accounts.
- Restrict the physical location of newly created resources.
- There are many more constraints that give you fine-grained control of the organization's resources. For more information, see the [list of all Organization Policy Service constraints](https://cloud.google.com/resource-manager/docs/organization-policy/org-policy-constraints).

---

## Differences from Identity and Access Management

Identity and Access Management
- focuses on who
- lets the administrator authorize who can take action on specific resources based on permissions.

Organization Policy
- focuses on what
- lets the administrator `set restrictions on specific resources` to determine how they can be configured.

---

## Key Concepts

### Organization policy

![Screenshot 2023-08-11 at 14.03.22](/assets/img/Screenshot%202023-08-11%20at%2014.03.22.png)

- An organization policy is a configuration of restrictions.

- organization policy administrator define an organization policy, and set that organization policy on organizations, folders, and projects in order to enforce the restrictions on that resource and its descendants.

- to define an organization policy, choose a `constraint`, which is a particular type of restriction against either a Google Cloud service or a group of Google Cloud services. Configure that constraint with the desired restrictions.

- Descendants of the targeted resource hierarchy node inherit the organization policy. By applying an organization policy to the root organization node, you are able to effectively drive enforcement of that organization policy and configuration of restrictions across the organization.

### Constraints

> https://cloud.google.com/resource-manager/docs/organization-policy/understanding-constraints

- A constraint is a particular type of `restriction` against a Google Cloud service or a list of Google Cloud services.

- Think of the constraint as `a blueprint that defines what behaviors are controlled`.
  - This blueprint is then applied to a resource hierarchy node as an organization policy, which implements the rules defined in the constraint.
  - The Google Cloud service mapped to that constraint and associated with that resource hierarchy node will then enforce the restrictions configured within the organization policy.

- A constraint has a type, either list or boolean.

  - `List` constraints evaluate the constraint with a list of allowed or denied values, such as an allowlist of IP addresses that can connect to a virtual machine.

  - `Boolean` constraints are either enforced or not enforced for a given resource, and govern a specific behavior, such as whether external service accounts can be created.


- Each Google Cloud service evaluates constraint types and values to determine what should be restricted.

```yaml

# List
# Restrict configuration of external IPs to a list of instances

resource: "organizations/ORGANIZATION_ID"
policy: {
  constraint: "constraints/compute.vmExternalIpAccess"
  listPolicy: {
    allowedValues: [
      projects/PROJECT_NAME/zones/ZONE_ID/instances/INSTANCE_NAME,
      projects/PROJECT_NAME/zones/ZONE_ID/instances/INSTANCE_NAME
    ]
  }
}

# Boolean
# Disable service account creation

resource: "organizations/ORGANIZATION_ID"
policy: {
  constraint: "constraints/iam.disableServiceAccountCreation"
  booleanPolicy: {
    enforced: true
  }
}
```


#### boolean constraints

Under Enforcement, select an enforcement option:

- To enable enforcement of this constraint, select On.
- To disable enforcement of this constraint, select Off.


#### list constraints

Under Policy enforcement, select an enforcement option:

- To merge and evaluate the organization policies together, select Merge with parent. For more information about inheritance and the resource hierarchy, see Understanding Hierarchy Evaluation.
- To override the inherited policies completely, select Replace.

Under Policy type, select whether this organization policy will specify `allowed_values` or `denied_values`:

- To specify that the listed values will be the only allowed values, and all other values will be denied, select `Allow`.
- To specify that the listed values will be explicitly denied, and all other values will be allowed, select `Deny`.

Under Policy values, select whether this organization policy will apply to all values or a list of specific values:

- To apply the above policy type to every possible value, select `All`.
- To list explicit values, select Custom. In the Policy value text box that appears, enter a value and then press Enter. You can add multiple entries in this way. Click the New Policy Value button for each additional value.
- Specific values accepted by the policy depend on the service to which the policy applies. For a list of constraints and the values they accept, see Organization policy constraints.

---

#### Predefined Constraints

list: https://cloud.google.com/resource-manager/docs/organization-policy/org-policy-constraints

---

##### Constraints supported by multiple Google Cloud services

Constraints Description	Supported Prefixes

###### Allowed Worker Pools (Cloud Build)
- By default, Cloud Build can use any Worker Pool.
- This list constraint defines the `set of allowed Cloud Build Worker Pools` for performing Builds using Cloud Build.
- When this constraint is enforced, builds will be `required to build in a Worker Pool that matches one of the allowed values.`
  - The allowed list of Worker Pools must be of the form:
  - under:organizations/ORGANIZATION_ID
  - under:folders/FOLDER_ID
  - under:projects/PROJECT_ID
  - projects/PROJECT_ID/locations/REGION/workerPools/WORKER_POOL_ID
- **Constraints**: `constraints/cloudbuild.allowedWorkerPools`
- **Supported Prefixes**: `"is:", "under:"`


###### Disable Enabling Identity-Aware Proxy (IAP) on regional resources
- By default, enabling IAP on regional resources is allowed.
- This boolean constraint, when enforced, disables turning on Identity-Aware Proxy on regional resources. Enabling IAP on global resources is not restricted by this constraint.
- **Constraints**: `constraints/iap.requireRegionalIapWebDisabled`
- **Supported Prefixes**: `"is:"`


###### Google Cloud Platform - Resource Location Restriction
- By default, resources can be created in any location.
- This list constraint `defines the set of locations where location-based Google Cloud resources can be created.`
- Policies for this constraint can specify multi-regions such as asia and europe, regions such as us-east1 or europe-west1 as allowed or denied locations.
  - Allowing or denying a multi-region does not imply that all included sub-locations should also be allowed or denied.
  - For example, if the policy denies the us multi-region (which refers to multi-region resources, like some storage services), resources can still be created in the regional location us-east1. On the other hand, the `in:us-locations` group contains all locations within the us region, and can be used to block every region.
- We recommend using value groups to define the policy.
  - You can specify value groups, collections of locations that are curated by Google to provide a simple way to define the resource locations.
  - To use value groups in the organization policy, prefix the entries with the string `in:`, followed by the value group.
  - For example, to create resources that will only be physically located within the US, set `in:us-locations` in the list of allowed values.
  - If the suggested_value field is used in a location policy, it should be a region.
  - If the value specified is a region, a UI for a zonal resource may pre-populate any zone in that region.
- **Constraints**: `constraints/gcp.resourceLocations`
- **Supported Prefixes**: `"is:", "in:"`

```yaml
constraint: constraints/gcp.resourceLocations
listPolicy:
    deniedValues:
    - in:us-east1-locations
    - in:northamerica-northeast1-locations
```


###### Restrict allowed Google Cloud APIs and services
- By default, all services are allowed.
- This list constraint `restricts the set of services and their APIs that can be enabled on this resource`.
- The denied list of services must come from the list below.
- Explicitly enabling APIs via this constraint is not currently supported.
- Specifying an API not in this list will result in an error.
  - compute.googleapis.com
  - deploymentmanager.googleapis.com
  - dns.googleapis.com
- Enforcement of this constraint is not retroactive. If a service is already enabled on a resource when this constraint is enforced, it will remain enabled.
- **Constraints**: `constraints/serviceuser.services`
- **Supported Prefixes**: `"is:"`


###### Restrict Resource Service Usage
- By default, all Google Cloud resource services are allowed.
- This constraint defines `the set of Google Cloud resource services that can be used within an organization, folder, or project`, such as compute.googleapis.com and storage.googleapis.com.
- For more information, see https://cloud.google.com/resource-manager/help/organization-policy/restricting-resources.
- **Constraints**: `constraints/gcp.restrictServiceUsage`
- **Supported Prefixes**: `"is:"`

- Administrators can use this constraint to define hierarchical restrictions on allowed Google Cloud resource services within a resource container, such as an organization, a folder, or a project. For example,
  - allow storage.googleapis.com within project X,
  - deny compute.googleapis.com within folder Y.

- constraint can be used in two mutually exclusive ways:
  - Denylist: resources of any service that isn't denied are allowed.
  - Allowlist: resources of any service that isn't allowed are denied.

```yaml
constraint: constraints/gcp.restrictServiceUsage
  list_policy:
  denied_values:
    - file.googleapis.com
    - bigquery.googleapis.com
    - storage.googleapis.com
```


###### Restrict which projects may supply KMS CryptoKeys for CMEK

- This list constraint defines which projects may be used to supply Customer-Managed Encryption Keys (CMEK) when creating resources.
- Setting this constraint to Allow (i.e. only allow CMEK keys from these projects) `ensures that CMEK keys from other projects cannot be used to protect newly created resources`.
- Values for this constraint must be specified in the form of
  - under:organizations/ORGANIZATION_ID,
  - under:folders/FOLDER_ID,
  - or projects/PROJECT_ID.

- Supported services that enforce this constraint are:
  - aiplatform.googleapis.com
  - artifactregistry.googleapis.com
  - bigquery.googleapis.com
  - bigtable.googleapis.com
  - cloudfunctions.googleapis.com
  - composer.googleapis.com
  - compute.googleapis.com
  - container.googleapis.com
  - dataflow.googleapis.com
  - dataproc.googleapis.com
  - documentai.googleapis.com
  - logging.googleapis.com
  - pubsub.googleapis.com
  - run.googleapis.com
  - secretmanager.googleapis.com
  - spanner.googleapis.com
  - sqladmin.googleapis.com
  - storage.googleapis.com

- Enforcement of this constraint may grow over time to include additional services. Use caution when applying this constraint to projects, folders, or organizations where a mix of supported and unsupported services are used.

- Setting this constraint to `Deny or Deny All` is not permitted.

- Enforcement of this constraint is not retroactive. Existing CMEK Google Cloud resources with KMS CryptoKeys from disallowed projects must be reconfigured or recreated manually to ensure enforcement.

- **Constraints**: `constraints/gcp.restrictCmekCryptoKeyProjects`
- **Supported Prefixes**: `"is:", "under:"`


###### Restrict which services may create resources without CMEK

- This list constraint defines which services require Customer-Managed Encryption Keys (CMEK).

- Setting this constraint to `Deny (i.e. deny resource creation without CMEK), requires newly created resources in the specified services must be protected by a CMEK key`.

- Supported services that can be set in this constraint are:
  - aiplatform.googleapis.com
  - artifactregistry.googleapis.com
  - bigquery.googleapis.com
  - bigtable.googleapis.com
  - cloudfunctions.googleapis.com
  - composer.googleapis.com
  - compute.googleapis.com
  - container.googleapis.com
  - dataflow.googleapis.com
  - dataproc.googleapis.com
  - documentai.googleapis.com
  - logging.googleapis.com
  - pubsub.googleapis.com
  - run.googleapis.com
  - secretmanager.googleapis.com
  - spanner.googleapis.com
  - sqladmin.googleapis.com
  - storage.googleapis.com

- Setting this constraint to `Deny All` is not permitted.
- Setting this constraint to `Allow` is not permitted.

- Enforcement of this constraint is not retroactive. Existing non-CMEK Google Cloud resources must be reconfigured or recreated manually to ensure enforcement.


- **Constraints**: `constraints/gcp.restrictNonCmekServices`
- **Supported Prefixes**: `"is:"`





---

#### Custom constraints

> https://cloud.google.com/resource-manager/docs/organization-policy/creating-managing-custom-constraints

- Custom constraints can allow or restrict resource creation and updates in the same way that predefined constraints do, but `allow administrators to configure conditions` based on request parameters and other metadata.

- You can create custom constraints that restrict operations on certain service resources, such as Dataproc NodePool resources.

- list of service resources that support custom constraints: https://cloud.google.com/resource-manager/docs/organization-policy/custom-constraint-supported-services

- Required roles

  - ask for Organization policy administrator (`roles/orgpolicy.policyAdmin`) IAM role on the organization.

  - This predefined role contains the permissions required to manage organization policies.
    - orgpolicy.constraints.list
    - orgpolicy.policies.create
    - orgpolicy.policies.delete
    - orgpolicy.policies.list
    - orgpolicy.policies.update
    - orgpolicy.policy.get
    - orgpolicy.policy.set

- A custom constraint is created in a YAML file which specifies the resources, methods, conditions, and actions that are subject to the constraint. These are specific to the service on which you're enforcing the organization policy.

- The conditions for your custom constraint are defined using `Common Expression Language (CEL)`.


```yaml
name: organizations/ORGANIZATION_ID/customConstraints/CONSTRAINT_NAME
resourceTypes:
- RESOURCE_NAME
- container.googleapis.com/NodePool
methodTypes:
- METHOD1
- METHOD2
- a list of RESTful methods for which to enforce the constraint.
- Can be CREATE or CREATE and UPDATE
condition: ("resource.management.autoUpgrade == false")
actionType: (ALLOW or DENY)
displayName: DISPLAY_NAME
description: DESCRIPTION

name: organizations/1234567890123/customConstraints/custom.disableGkeAutoUpgrade
resourceTypes:
- container.googleapis.com/NodePool
methodTypes:
- CREATE
- UPDATE
condition: "resource.management.autoUpgrade == false"
actionType: ALLOW
displayName: Disable GKE auto upgrade
description: Only allow GKE NodePool resource to be created or updated if AutoUpgrade is not enabled where this custom constraint is enforced.
```






---

### Inheritance

> https://cloud.google.com/resource-manager/docs/organization-policy/understanding-hierarchy

- When an organization policy is set on a resource hierarchy node, all descendants of that node inherit the organization policy by default. If you set an organization policy at the root organization node, then the configuration of restrictions defined by that policy will be passed down through all descendant folders, projects, and service resources.

- A user with the Organization Policy Administrator role can set descendant resource hierarchy nodes with another organization policy that either overwrites the inheritance, or merges them based on the rules of hierarchy evaluation. This provides precise control for how the organization policies apply throughout the organization, and where you want exceptions made.

---

### Violations

- A violation is when a Google Cloud service acts or is in a state that is counter to the organization policy restriction configuration within the scope of its resource hierarchy. Google Cloud services will enforce constraints to prevent violations, but the application of new organization policies is usually not retroactive. If an organization policy constraint is retroactively enforced, it will be labeled as such on the Organization Policy Constraints page.

- If a new organization policy sets a restriction on an action or state that a service is already in, the policy is considered to be in violation, but the service will not stop its original behavior. You will need to address this violation manually. This prevents the risk of a new organization policy completely shutting down the business continuity.

---

## command

```bash
gcloud org-policies describe LIST_CONSTRAINT \
  --organization=ORGANIZATION_ID

gcloud org-policies set-policy /tmp/policy.yaml




```






.
