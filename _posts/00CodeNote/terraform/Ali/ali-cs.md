---
title: Terraform Example - ali-cs
# author: Grace JyL
date: 2021-10-12 11:11:11 -0400
description:
excerpt_separator:
categories: [00CodeNote, Terraform]
tags: [Terraform]
# math: true
# pin: true
toc: true
# image: /assets/img/note/tls-ssl-handshake.png
---

# Terraform Example - ali-cs

Example Usage
https://registry.terraform.io/providers/aliyun/alicloud/latest/docs/resources/cs_managed_kubernetes#example-usage


```t
# If there is not specifying vpc_id, the module will launch a new vpc
resource "alicloud_vpc" "vpc" {
  count      = var.vpc_id == "" ? 1 : 0
  cidr_block = var.vpc_cidr
}

# According to the vswitch cidr blocks to launch several vswitches
resource "alicloud_vswitch" "vswitches" {
  count      = length(var.vswitch_ids) > 0 ? 0 : length(var.vswitch_cidrs)
  vpc_id     = var.vpc_id == "" ? join("", alicloud_vpc.vpc.*.id) : var.vpc_id
  cidr_block = element(var.vswitch_cidrs, count.index)
  zone_id    = element(var.availability_zone, count.index)
}


# According to the vswitch cidr blocks to launch several vswitches
resource "alicloud_vswitch" "terway_vswitches" {
  count      = length(var.terway_vswitch_ids) > 0 ? 0 : length(var.terway_vswitch_cirds)
  vpc_id     = var.vpc_id == "" ? join("", alicloud_vpc.vpc.*.id) : var.vpc_id
  cidr_block = element(var.terway_vswitch_cirds, count.index)
  zone_id    = element(var.availability_zone, count.index)
}

resource "alicloud_cs_managed_kubernetes" "this" {
  # ++++++++++++++ Global params ++++++++++++++
  name               = "---Optional"
  worker_vswitch_ids = "---Optional/['vsw-id1', 'vsw-id1', 'vsw-id2']"
  name_prefix        = "---Optional/Terraform-Creation"
  timezone           = "---Optional"
  resource_group_id  = "---Optional"
  version            = "---Optional" # version can not be defined in variables.tf.
  # runtime            = "Deprecated"
  # runtdime = {
  #   name    = "containerd"
  #   version = "1.5.13"
  # }
  # enable_ssh                   = "Deprecated/false/true"
  # rds_instances                = "Deprecated"
  security_group_id            = "---Optional"
  is_enterprise_security_group = "---Optional/false/true"
  proxy_mode                   = "---Optional/ipvs/iptables"
  cluster_domain               = "---Optional/cluster.local/other"
  custom_san                   = "---Optional"
  user_ca                      = "---Optional"
  deletion_protection          = "---Optional"
  enable_rrsa                  = "---Optional/false/true"
  install_cloud_monitor        = "---Optional/true/false"
  # exclude_autoscaler_nodes     = "Deprecated/false/true"
  service_account_issuer = "---Optional"
  api_audiences          = "---Optional"
  tags                   = "---Optional/nil/{}"
  # tags = {
  #   "key1" = "value1"
  #   "key2" = "value2"
  #   "name" = "tf"
  # }
  cluster_spec            = "---Optional/ack.standard/ack.pro.small"
  encryption_provider_key = "---Optional"
  maintenance_window      = "---Optional"
  # maintenance_window {
  #   enable           = true
  #   maintenance_time = "01:00:00Z"
  #   duration         = "1-24h"
  #   weekly_period    = "Monday,Friday"
  # }
  load_balancer_spec           = "slb.s1.small/other"
  control_plane_log_ttl        = "---Optional/30/other"
  control_plane_log_components = "---Optional/apiserver/kcm/scheduler[LIST]"
  control_plane_log_project    = "---Optional"
  retain_resources             = "---Optional"
  addons                       = "---Optional"
  # dynamic "addons" {
  #   for_each = var.cluster_addons
  #   content {
  #     name     = lookup(addons.value, "name", var.cluster_addons)
  #     config   = lookup(addons.value, "config", var.cluster_addons)
  #     disabled = lookup(addons.value, "disabled", var.cluster_addons)
  #   }
  # }


  #++++++++++++++ Network params ++++++++++++++
  pod_cidr             = "---Optional"
  pod_vswitch_ids      = "---Optional"
  new_nat_gateway      = "---Optional/true/false"
  service_cidr         = "---Optional"
  node_cidr_mask       = "---Optional/24-28"
  slb_internet_enabled = "---Optional/true/false"




  # ++++++++++++++ Worker params ++++++++++++++
  # worker_number                   = "Deprecated/3-50"
  # worker_instance_types           = "Deprecated"
  # password                        = "Deprecated"
  # key_name                        = "Deprecated"
  # kms_encrypted_password          = "Deprecated"
  # kms_encryption_context          = "Deprecated"
  # worker_instance_charge_type     = "Deprecated"
  # worker_period                   = "Deprecated"
  # worker_period_unit              = "Deprecated"
  # worker_auto_renew               = "Deprecated"
  # worker_auto_renew_period        = "Deprecated/1, 2, 3, 6, 12"
  # worker_disk_category            = "Deprecated"
  # worker_disk_size                = "Deprecated"
  # worker_data_disks               = "Deprecated"
  # category                        = "Deprecated"
  # size                            = "Deprecated"
  # encrypted                       = "Deprecated"
  # performance_level               = "Deprecated"
  # auto_snapshot_policy_id         = "Deprecated"
  # node_name_mode                  = "Deprecated"
  # node_port_range                 = "Deprecated"
  # os_type                         = "Deprecated"
  # platform                        = "Deprecated"
  # image_id                        = "Deprecated"
  # cpu_policy                      = "Deprecated"
  # user_data                       = "Deprecated"
  # taints                          = "Deprecated"
  # taints {
  #   key = "key-a"
  #   value = "value-a"
  #   effect = "NoSchedule"
  # }
  # worker_disk_performance_level   = "Deprecated"
  # worker_disk_snapshot_policy_id  = "Deprecated"
  # duplicate_install_cloud_monitor = "Deprecated"


  # ++++++++++++++ Computed params ++++++++++++++
  kube_config       = "---Optional/'~/.kube/config'"
  client_cert       = "---Optional/'~/.kube/client-cert.pem'"
  client_key        = "---Optional/'~/.kube/client-key.pem'"
  cluster_ca_cert   = "---Optional/'~/.kube/cluster-ca-cert.pem'"
  availability_zone = "---Optional"



  # ++++++++++++++ Removed params ++++++++++++++
  worker_instance_type      = "Deprecated"
  vswitch_ids               = "---Optional"
  force_update              = "---Optional/false"
  log_config                = "---Optional"
  type                      = "---Optional/SLS"
  project                   = "---Optional"
  cluster_network_type      = "---Optional/flannel/terway"
  worker_data_disk_category = "---Optional"
  worker_data_disk_size     = "---Optional"
  worker_numbers            = "---Optional"
}
```
