---
title: GCP - Google Cloud Deployment Manager
date: 2021-01-01 11:11:11 -0400
categories: [01GCP, IaC]
tags: [GCP]
toc: true
image:
---

- [Google Cloud Deployment Manager](#google-cloud-deployment-manager)
  - [overall](#overall)
  - [command](#command)
  - [configuration](#configuration)
    - [Dependencies `metadata.dependsOn`](#dependencies-metadatadependson)
    - [Outputs `outputs`](#outputs-outputs)
      - [Declaring output](#declaring-output)
      - [Using outputs from templates](#using-outputs-from-templates)
      - [Output description](#output-description)
      - [output finalValue](#output-finalvalue)
      - [Avoid circular dependencies](#avoid-circular-dependencies)


# Google Cloud Deployment Manager


## overall


## command

```bash
gcloud deployment-manager deployments create my-deployment --config vm.yaml

gcloud deployment-manager deployments describe my-deployment

gcloud deployment-manager deployments delete my-deployment
```


## configuration

Output values can be:
- A static string
- A reference to a property
- A template property
- An environment variable



```yaml
# Copyright 2016 Google Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Put all your resources under `resources:`. For each resource, you need:
# - The type of resource. In this example, the type is a Compute VM instance.
# - An internal name for the resource.
# - The properties for the resource. In this example, for VM instances, you add
#   the machine type, a boot disk, network information, and so on.
#
# For a list of supported resources,
# see https://cloud.google.com/deployment-manager/docs/configuration/supported-resource-types.
resources:
- type: compute.v1.instance
  name: quickstart-deployment-vm
  properties:
    # The properties of the resource depend on the type of resource. For a list
    # of properties, see the API reference for the resource.
    zone: us-central1-f
    # Replace [MY_PROJECT] with your project ID
    machineType: https://www.googleapis.com/compute/v1/projects/[MY_PROJECT]/zones/us-central1-f/machineTypes/f1-micro
    disks:
    - deviceName: boot
      type: PERSISTENT
      boot: true
      autoDelete: true
      initializeParams:
        # See a full list of image families at https://cloud.google.com/compute/docs/images#os-compute-support
        # The format of the sourceImage URL is: https://www.googleapis.com/compute/v1/projects/[IMAGE_PROJECT]/global/images/family/[FAMILY_NAME]
        sourceImage: https://www.googleapis.com/compute/v1/projects/debian-cloud/global/images/family/debian-11
    # Replace [MY_PROJECT] with your project ID
    networkInterfaces:
    - network: https://www.googleapis.com/compute/v1/projects/[MY_PROJECT]/global/networks/default
      # Access Config required to give the instance a public IP address
      accessConfigs:
      - name: External NAT
        type: ONE_TO_ONE_NAT
```
---

### Dependencies `metadata.dependsOn`


- To add a dependency to a resource, add a metadata section that contains a dependsOn section. Then, in the dependsOn section, specify one or more dependencies.
- In the same deployment, you must define the persistent disks that are dependencies.


```yaml
# to make a-special-vm dependent on the creation of two persistent disks
# - add the metadata and dependsOn sections for a-special-vm.
# - add the dependencies for each persistent disk.

resources:
- name: a-special-vm
  type: compute.v1.instances
  properties:
    ...
  metadata:
    dependsOn:
    - persistent-disk-a
    - persistent-disk-b

- name: persistent-disk-a
  type: compute.v1.disks
  properties:
    ...

- name: persistent-disk-b
  type: compute.v1.disks
  properties:
    ...
```

In this deployment, Deployment Manager creates persistent-disk-a and persistent-disk-b before creating a-special-vm.

> Warning: Avoid creating dependency loops. For example, if you specify that resource A depends on resource B, and resource B depends on resource A, a dependency loop is created, and the deployment fails. Additionally, if you use references in your deployment, implicit dependencies are created, which might also cause dependency loops.



---

### Outputs `outputs`


#### Declaring output


```yaml
resources:
- name: my-first-vm
  type: compute.v1.instance
  properties:
    zone: us-central1-a
    machineType: zones/us-central1-a/machineTypes/{{ properties['machineType'] }}
    disks:
    - deviceName: boot
      type: PERSISTENT
      boot: true
      autoDelete: true
      initializeParams:
        sourceImage: projects/debian-cloud/global/images/family/debian-11
    networkInterfaces:
    - network: global/networks/default
      accessConfigs:
      - name: External NAT
        type: ONE_TO_ONE_NAT

outputs:
- name: databaseIp
  value: $(ref.my-first-vm.networkInterfaces[0].accessConfigs[0].natIP)
- name: machineType
  value: {{ properties['machineType'] }}
- name: databasePort
  value: 88

# The outputs section declares two properties: databaseIp and databasePort.
# databaseIp uses a reference that resolves to the network IP address of the master resource,
# databasePort is a static value.
```


#### Using outputs from templates

- In another template, import mongodb.jinja, use the template as a type, and call the outputs.
- To call an output, use: `$(ref.RESOURCE.OUTPUT)`

```yaml
imports:
- path: template_with_outputs.jinja
  name: template.jinja

resources:
- name: my-first-vm
  type: template.jinja
  properties:
    machineType: n1-standard-1

outputs:
- name: databaseIp
  value: $(ref.my-first-vm.databaseIp)
- name: machineType
  value: $(ref.my-first-vm.machineType)
- name: databasePort
  value: $(ref.my-first-vm.databasePort)
```

#### Output description

- Describing outputs in schemas
- For templates that have accompanying schemas, you can describe output properties in further details. Deployment Manager does not enforce or validate any information in the outputs section but it is potentially helpful to use this section to provide more information about relevant outputs, for the benefit of users using your templates.

```yaml
...
outputs:
  databaseIp:
    description: Reference to ip address of your new cluster
    type: string
  databasePort:
    description: Port to talk on
    type: integer
```

#### output finalValue

- view the final output values in the configuration `layout` of the deployment.
- Final output values are indicated by the `finalValue` property.
- All output values are included in this field, including output values from nested templates.

```yaml
layout: |
  resources:
  - name: vm_template
    outputs:
    - finalValue: 104.197.69.69
      name: databaseIp
      value: $(ref.vm-test.networkInterfaces[0].accessConfigs[0].natIP)
    properties:
      zone: us-central1-a
    resources:
    - name: datadisk-example-instance
      type: compute.v1.disk
    - name: vm-test
      type: compute.v1.instance
    type: vm_template.jinja
name: manifest-1455057116997
```



#### Avoid circular dependencies

- Be careful when creating templates where two or more resources rely on outputs from each other.
- Deployment Manager does not prevent this structure but if the outputs caused a circular dependency, the deployment won't deploy successfully.


> For example, the following snippet is accepted by Deployment Manager but if the contents of the templates causes a circular dependency, the deployment would fail:

```yaml
resources:
- name: frontend
  type: frontend.jinja
  properties:
    ip: $(ref.backend.ip)
- name: backend
  type: backend.jinja
  properties:
    ip: $(ref.frontend.ip)

# assume both frontend.jinja and backend.jinja
resources:
- name: {{ env['name'] }}
  type: compute.v1.instance
  properties:
    zone: us-central1-f
    ...
    networkInterfaces:
    - network: global/networks/default
      accessConfigs:
      - name: External NAT
        type: ONE_TO_ONE_NAT
    metadata:
      items:
      - key: startup-script
        value: |
          #!/bin/bash
          export IP={{ properties["ip"] }}
      ...
outputs:
- name: ip
  value: $(ref.{{ env['name'] }}.networkInterfaces[0].accessConfigs[0].natIP)
```

- both resources used the IP output property from the opposing resource
- But neither IP values can be populated because both properties rely on the existence of the other resource, creating a circular dependency.

```yaml
resources:
- name: frontend
  type: compute.v1.instance
  properties:
    zone: us-central1-f
    ...
    networkInterfaces:
    - network: global/networks/default
      accessConfigs:
      - name: External NAT
        type: ONE_TO_ONE_NAT
    metadata:
      items:
      - key: startup-script
        value: |
          #!/bin/bash
          export IP=$(ref.backend.networkInterfaces[0].accessConfigs[0].natIP)
- name: backend
  type: compute.v1.instance
  properties:
    zone: us-central1-f
    ...
    networkInterfaces:
    - network: global/networks/default
      accessConfigs:
      - name: External NAT
        type: ONE_TO_ONE_NAT
    metadata:
      items:
      - key: startup-script
        value: |
          #!/bin/bash
          export IP=$(ref.frontend.networkInterfaces[0].accessConfigs[0].natIP)
```


- Deployment Manager returns an error if you try to run configuration:

```bash
code: u'CONDITION_NOT_MET'
message: u'A dependency cycle was found amongst backend, frontend.'>]>
```


- However, this template would work if:
  - `frontend.jinja` created two virtual machine instances, vm-1 and vm-2.
  - `backend.jinja` created vm-3 and vm-4.
  - `vm-1` exposed it's external IP as an output and `vm-4` used that output.
  - `vm-3` exposed an external IP as an output, `vm-2` used that output.





.
