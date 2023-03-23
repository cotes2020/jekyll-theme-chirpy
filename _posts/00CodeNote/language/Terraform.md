






- [Terraform](#terraform)
  - [Basic](#basic)
  - [use case](#use-case)
  - [Deploy](#deploy)
    - [install](#install)
    - [alias](#alias)
  - [Variables and Outputs](#variables-and-outputs)
    - [Input Variables](#input-variables)
      - [Arguments](#arguments)
    - [Output Values](#output-values)
    - [Modules](#modules)
    - [局部参数（Local Values）](#局部参数local-values)
- [Template](#template)
  - [AWS](#aws)
    - [provision an EC2 instance](#provision-an-ec2-instance)
  - [GCP](#gcp)

---


# Terraform


---

## Basic

- HashiCorp Terraform is an infrastructure as code tool
- lets you define both cloud and on-prem resources in human-readable configuration files that you can version, reuse, and share.
- use a consistent workflow to safely and efficiently provision and manage your infrastructure throughout its lifecycle.

![intro-terraform-workflow](https://i.imgur.com/fJFVY3m.jpg)

- Terraform plugins called providers let Terraform interact with cloud platforms and other services via their application programming interfaces (APIs).


- Terraform's configuration language is **declarative**
  - it describes the desired end-state for your infrastructure, in contrast to procedural programming languages that require step-by-step instructions to perform tasks.
  - Terraform providers automatically calculate dependencies between resources to create or destroy them in the correct order.


![assets](https://i.imgur.com/nnjRp1E.png)




---

## use case

- **Standardize your deployment workflow**
  - Terraform's state allows you to track resource changes throughout your deployments.
  - compose resources from different providers into reusable Terraform configurations called **modules**, and manage them with a consistent language and workflow.



- The human-readable configuration language helps you write infrastructure code quickly.


- You can commit your configurations to version control to safely collaborate on infrastructure.

- **Multi-Cloud Deployment**
  - Terraform can manage infrastructure on multiple cloud platforms.
  - Provisioning infrastructure across multiple clouds increases fault-tolerance, allowing for more graceful recovery from cloud provider outages.
  - multi-cloud deployments add complexity because each provider has its own interfaces, tools, and workflows. Terraform lets you use the same workflow to manage multiple providers and handle cross-cloud dependencies. This simplifies management and orchestration for large-scale, multi-cloud infrastructures.


- **Application Infrastructure Deployment, Scaling, and Monitoring Tools**
  - efficiently deploy, release, scale, and monitor infrastructure for multi-tier applications.
  - N-tier application architecture lets you scale application components independently and provides a separation of concerns.
  - An application could consist of a pool of web servers that use a database tier, with additional tiers for API servers, caching servers, and routing meshes.
  - Terraform allows to manage the resources in each tier together, and automatically handles dependencies between tiers.
  - For example, Terraform will deploy a database tier before provisioning the web servers that depend on it.



- **Self-Service Clusters**
  - centralized operations team may get many repetitive infrastructure requests.
  - use Terraform to build a "self-serve" infrastructure model that lets product teams manage their own infrastructure independently.
  - create and use Terraform modules that codify the standards for deploying and managing services in your organization, allowing teams to efficiently deploy services in compliance with your organization’s practices. Terraform Cloud can also integrate with ticketing systems like ServiceNow to automatically generate new infrastructure requests.


- **Policy Compliance and Management**
  - Terraform help enforce policies on the types of resources teams can provision and use.
  - Ticket-based review processes are a bottleneck that can slow down development.
  - Instead, use Sentinel, a policy-as-code framework, to automatically enforce compliance and governance policies before Terraform makes infrastructure changes.
  - Sentinel is available with the Terraform Cloud team and governance tier.

- **PaaS Application Setup**
  - `Platform as a Service (PaaS) vendors` like Heroku allow you to create web applications and attach add-ons, such as databases or email providers.
    - Heroku can elastically scale the number of dynos or workers, but most non-trivial applications need many add-ons and external services.
  - use Terraform to codify the setup required for a Heroku application, configure a DNSimple to set a CNAME, and set up Cloudflare as a Content Delivery Network (CDN) for the app. Terraform can quickly and consistently do all of this without a web interface.



- **Software Defined Networking**
  - Terraform can interact with Software Defined Networks (SDNs) to automatically configure the network according to the needs of the applications running in it.
  - move from a ticket-based workflow to an automated one, reducing deployment times.
  - For example,
    - when a service registers with HashiCorp Consul, Consul-Terraform-Sync can automatically generate Terraform configuration to expose appropriate ports and adjust network settings for any SDN that has an associated Terraform provider.
    - Network Infrastructure Automation (NIA) allows you to safely approve the changes that your applications require without having to manually translate tickets from developers into the changes you think their applications need.


- **Kubernetes**
  - Kubernetes is an open-source workload scheduler for containerized applications.
  - Terraform lets you both deploy a Kubernetes cluster and manage its resources (e.g., pods, deployments, services, etc.).
  - can also use the Kubernetes Operator for Terraform to manage cloud and on-prem infrastructure through a Kubernetes Custom Resource Definition (CRD) and Terraform Cloud.


- **Parallel Environments**
  - You may have staging or QA environments that you use to test new applications before releasing them in production.
  - As the production environment grows larger and more complex, it can be increasingly difficult to maintain an up-to-date environment for each stage of the development process.
  - Terraform lets you rapidly spin up and decommission infrastructure for development, test, QA, and production.
  - Using Terraform to create disposable environments as needed is more cost-efficient than maintaining each one indefinitely.

- **Software Demos**
  - use Terraform to create, provision, and bootstrap a demo on various cloud providers.
  - This lets end users easily try the software on their own infrastructure and even enables them to adjust parameters like cluster size to more rigorously test tools at any scale.


---


## Deploy

To deploy infrastructure with Terraform:

- `Scope` - Identify the infrastructure for your project.
- `Author` - Write the configuration for your infrastructure.
- `Initialize` - Install the plugins Terraform needs to manage the infrastructure.
- `Plan` - Preview the changes Terraform will make to match your configuration.
- `Apply` - Make the planned changes.


---


### install

```bash
$ brew tap hashicorp/tap

$ brew install hashicorp/tap/terraform

$ brew update

$ brew upgrade hashicorp/tap/terraform

$ terraform init
$ terraform apply
$ terraform destroy
```

---



### alias

- A provider block without an alias argument is the default configuration for that provider. Resources that don't set the provider meta-argument will use the default provider configuration that matches the first word of the resource type name.



**Multiple Provider Configurations**

```bash
# The default provider configuration; resources that begin with `aws_` will use
# it as the default, and it can be referenced as `aws`.
provider "aws" {
  region = "us-east-1"
}

# Additional provider configuration for west coast region; resources can
# reference this as `aws.west`.
provider "aws" {
  alias  = "west"
  region = "us-west-2"
}

```


- To declare a configuration alias within a module in order to receive an alternate provider configuration from the parent module, add the configuration_aliases argument to that provider's required_providers entry. The following example declares both the mycloud and mycloud.alternate provider configuration names within the containing module:

```bash
terraform {
  required_providers {
    mycloud = {
      source  = "mycorp/mycloud"
      version = "~> 1.0"
      configuration_aliases = [ mycloud.alternate ]
    }
  }
}
```




**Selecting Alternate Provider Configurations**


```bash

# To use an alternate provider configuration for a resource or data source,
resource "aws_instance" "foo" {
  provider = aws.west
  # ...
}

```
To select alternate provider configurations for a child module, use its providers meta-argument to specify which provider configurations should be mapped to which local provider names inside the module:
```bash
module "aws_vpc" {
  source = "./aws_vpc"
  providers = {
    aws = aws.west
  }
}
```


---

## Variables and Outputs


- Input variables are like function arguments.
- Output values are like function return values.
- Local values are like a function's temporary local variables.




---

### Input Variables

- customize aspects of Terraform modules without altering the module's own source code.
- allows you to share modules across different Terraform configurations, making module composable and reusable.

- declare variables in the `root module` of your configuration, you can set their values using CLI options and environment variables.
- declare them in `child modules`, the calling module should pass values in the `module` block.

```bash
variable "image_id" {
  type = string
}

variable "availability_zone_names" {
  type    = list(string)
  default = ["us-west-1a"]
}

variable "docker_ports" {
  type = list(object({
    internal = number
    external = number
    protocol = string
  }))
  default = [
    {
      internal = 8300
      external = 8300
      protocol = "tcp"
    }
  ]
}


```

---


#### Arguments

Terraform CLI defines the following optional arguments for variable declarations:


`default`

- A default value which then makes the variable optional.

- the default value will be used if no value is set when calling the module or running Terraform.

- The default argument requires a literal value and cannot reference other objects in the configuration.

`type`

- This argument specifies what value types are accepted for the variable.

`description`

- This specifies the input variable's documentation.

`validation`

- A block to define validation rules, usually in addition to `type` constraints.
- module author can specify arbitrary custom validation rules for a particular variable using a validation block nested within the corresponding variable block:


```bash
variable "image_id" {
  type        = string
  description = "The id of the machine image (AMI) to use for the server."

  validation {
    condition     = length(var.image_id) > 4 && substr(var.image_id, 0, 4) == "ami-"
    error_message = "The image_id value must be a valid AMI id, starting with \"ami-\"."
  }

  validation {
    # regex(...) fails if it cannot find a match
    condition     = can(regex("^ami-", var.image_id))
    error_message = "The image_id value must be a valid AMI id, starting with \"ami-\"."
  }
}
```



`sensitive`

- Limits Terraform UI output when the variable is used in configuration.
- Setting a variable as sensitive prevents Terraform from showing its value in the plan or apply output, when you use that variable elsewhere in your configuration.
- Terraform will still record sensitive values in the state, and so anyone who can access the state data will have access to the sensitive values in cleartext.




`nullable`

- Specify if the variable can be null within the module.

---



### Output Values

- 在很多时候，我们需要将在云环境上创建的实例某些属性暴露给用户，比如创建的虚拟机需要暴露网卡地址，创建的web server需要暴露访问地址
- Output Values 可以将实例的属性或者组合的属性暴露给用户，最大程度降低用户访问实例资源的可能。
- Output可以有很多设置
  - 比如sensitive可以保证输出参数在控制台不可见，但是在模块内部可见。
  - depends_on可以显示指定出参的依赖资源



举个栗子：我们需要创建一个mysql数据库，用户实际上不需要关心mysql创建的虚拟机是什么，但是用户需要知道mysql虚拟的IP、访问端口等，因此我们通过定义Output来将mysql的相关信息返回

```bash
output "mysql_ip" {
  value = aws_instance.server.private_ip
}

output "mysql_port" {
  value = var.port
}
# 第一个参数mysql_ip是创建的虚拟机实例的IP地址，
# 第二个参数mysql_port是用户输入或者默认的端口地址
```



---

### Modules

- 模块可以包含多个Terraform配置文件（模板）
- 主要是为了方便资源重用、复杂场景模块化需要。
- 在同一个工作目录内定义一系列.tf文件，来整合一个复杂场景，充分利用一些基础资源等等。



Modules are containers for multiple resources that are used together.
- A module consists of a collection of `.tf and/or .tf.json` files kept together in a directory.

- Modules are the main way to package and reuse resource configurations with Terraform.


**The Root Module**

Every Terraform configuration has at least one module, known as its root module, which consists of the resources defined in the .tf files in the main working directory.


**Child Modules**

A Terraform module (usually the root module of a configuration) can call other modules to include their resources into the configuration. A module that has been called by another module is often referred to as a child module.

Child modules can be called multiple times within the same configuration, and multiple configurations can use the same child module.

`calling` module:

```bash
module "servers" {
  source = "./app-cluster"

  servers = 5
}
```


**Published Modules**

In addition to modules from the local filesystem, Terraform can load modules from a public or private registry. This makes it possible to publish modules for others to use, and to use modules that others have published.


---





### 局部参数（Local Values）

- 局部参数可以在某一个模块内定义
- 主要用途是方便在同一个模块中多次使用
  - locals 通常在 module 內用於重複使用，
  - 常常用來將 variable 的值拿來運算或者特殊處理。


- 一般来说是常量。

- Once a local value is declared, you can reference it in expressions as `local.<NAME>`

```bash
# A set of related local values can be declared together
locals {
  service_name = "forum"
  owner        = "Community Team"
}

locals {
  # Ids for multiple sets of EC2 instances, merged together
  instance_ids = concat(aws_instance.blue.*.id, aws_instance.green.*.id)
}

locals {
  # Common tags to be assigned to all resources
  common_tags = {
    Service = local.service_name
    Owner   = local.owner
  }
}
```



1. 多個 variable 一次餵給 resource。

```bash
# Define the common tags for all resources
locals {
  common_tags = {
    Component   = "awesome-app"
    Environment = "production"
  }
}

# Create a resource that blends the common tags with instance-specific tags.
resource "aws_instance" "example" {
  # ...
  tags = local.common_tags
}

resource "aws_instance" "server" {
  ami           = "ami-123456"
  instance_type = "t2.micro"
  tags = "${
    merge(
      local.common_tags,
      map(
        "Name", "awesome-app-server",
        "Role", "server"
      )
    )
  }"
}
```


2. locals 內可以包含 locals 進行多次處理後才給 resource。

```bash
# Ids for multiple sets of EC2 instances, merged together
locals {
  instance_ids = "${concat(aws_instance.blue.*.id, aws_instance.green.*.id)}"
}

# A computed default name prefix
locals {
  default_name_prefix = "${var.project_name}-web"
  name_prefix         = "${var.name_prefix != "" ? var.name_prefix : local.default_name_prefix}"
}

# Local values can be interpolated elsewhere using the "local." prefix.
resource "aws_s3_bucket" "files" {
  bucket = "${local.name_prefix}-files"
  # ...
}
```











---


# Template


## AWS


### provision an EC2 instance

```bash
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 3.27"
    }
  }
  required_version = ">= 0.14.9"
}

# Configure the AWS Provider
provider "aws" {
  profile = "default"
  region  = "us-west-2"
}

resource "aws_instance" "app_server" {
  ami           = "ami-830c94e3"
  instance_type = "t2.micro"

  tags = {
    Name = "ExampleAppServerInstance"
  }
}
# Create a VPC
resource "aws_vpc" "example" {
  cidr_block = "10.0.0.0/16"
}



$ terraform plan

# automatically updates configurations in the current directory for readability and consistency.
$ terraform fmt

# make sure your configu[[[''\'']]]ation is syntactically valid and internally consistent
$ terraform validate
```


---

## GCP


```bash

provider "google" {
  project = "acme-app"
  region  = "us-central1"
}


```











.
