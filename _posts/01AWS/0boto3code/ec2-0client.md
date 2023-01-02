
[toc]


---

# EC2 - client

Table of Contents

- EC2
  - Client
  - Paginators
  - Waiters
  - Service Resource
  - ClassicAddress
  - DhcpOptions
  - Image
  - Instance
  - InternetGateway
  - KeyPair
  - KeyPairInfo
  - NetworkAcl
  - NetworkInterface
  - NetworkInterfaceAssociation
  - PlacementGroup
  - Route e)
  - RouteTable
  - RouteTableAssociation
  - SecurityGroup
  - Snapshot
  - Subnet
  - Tag
  - Volume
  - Vpc
  - VpcPeeringConnection
  - VpcAddress


---


# EC2 - client

_class_ EC2.Client

- A low-level client representing Amazon Elastic Compute Cloud (EC2)
- Amazon Elastic Compute Cloud (Amazon EC2) provides secure and resizable computing capacity in the AWS cloud.
- Using Amazon EC2 eliminates the need to invest in hardware up front, so you can develop and deploy applications faster.


```py
import boto3
ec2client = boto3.client('ec2')

# available methods:
- accept_reserved_instances_exchange_quote()
- accept_transit_gateway_multicast_domain_associations()
- accept_transit_gateway_peering_attachment()
- accept_transit_gateway_vpc_attachment()
- accept_vpc_endpoint_connections()
- accept_vpc_peering_connection()
- advertise_byoip_cidr()
- allocate_address()
- allocate_hosts()
- apply_security_groups_to_client_vpn_target_network()
- assign_ipv6_addresses()
- assign_private_ip_addresses()
- associate_address()
- associate_client_vpn_target_network()
- associate_dhcp_options()
- associate_enclave_certificate_iam_role()
- associate_iam_instance_profile()
- associate_route_table()
- associate_subnet_cidr_block()
- associate_transit_gateway_multicast_domain()
- associate_transit_gateway_route_table()
- associate_vpc_cidr_block()
- attach_classic_link_vpc()
- attach_internet_gateway()
- attach_network_interface()
- attach_volume()
- attach_vpn_gateway()
- authorize_client_vpn_ingress()
- authorize_security_group_egress()
- authorize_security_group_ingress()
- bundle_instance()
- can_paginate()
- cancel_bundle_task()
- cancel_capacity_reservation()
- cancel_conversion_task()
- cancel_export_task()
- cancel_import_task()
- cancel_reserved_instances_listing()
- cancel_spot_fleet_requests()
- cancel_spot_instance_requests()
- confirm_product_instance()
- copy_fpga_image()
- copy_image()
- copy_snapshot()
- create_capacity_reservation()
- create_carrier_gateway()
- create_client_vpn_endpoint()
- create_client_vpn_route()
- create_customer_gateway()
- create_default_subnet()
- create_default_vpc()
- create_dhcp_options()
- create_egress_only_internet_gateway()
- create_fleet()
- create_flow_logs()
- create_fpga_image()
- create_image()
- create_instance_export_task()
- create_internet_gateway()
- create_key_pair()
- create_launch_template()
- create_launch_template_version()
- create_local_gateway_route()
- create_local_gateway_route_table_vpc_association()
- create_managed_prefix_list()
- create_nat_gateway()
- create_network_acl()
- create_network_acl_entry()
- create_network_insights_path()
- create_network_interface()
- create_network_interface_permission()
- create_placement_group()
- create_reserved_instances_listing()
- create_route()
- create_route_table()
- create_security_group()
- create_snapshot()
- create_snapshots()
- create_spot_datafeed_subscription()
- create_subnet()
- create_tags()
- create_traffic_mirror_filter()
- create_traffic_mirror_filter_rule()
- create_traffic_mirror_session()
- create_traffic_mirror_target()
- create_transit_gateway()
- create_transit_gateway_connect()
- create_transit_gateway_connect_peer()
- create_transit_gateway_multicast_domain()
- create_transit_gateway_peering_attachment()
- create_transit_gateway_prefix_list_reference()
- create_transit_gateway_route()
- create_transit_gateway_route_table()
- create_transit_gateway_vpc_attachment()
- create_volume()
- create_vpc()
- create_vpc_endpoint()
- create_vpc_endpoint_connection_notification()
- create_vpc_endpoint_service_configuration()
- create_vpc_peering_connection()
- create_vpn_connection()
- create_vpn_connection_route()
- create_vpn_gateway()
- delete_carrier_gateway()
- delete_client_vpn_endpoint()
- delete_client_vpn_route()
- delete_customer_gateway()
- delete_dhcp_options()
- delete_egress_only_internet_gateway()
- delete_fleets()
- delete_flow_logs()
- delete_fpga_image()
- delete_internet_gateway()
- delete_key_pair()
- delete_launch_template()
- delete_launch_template_versions()
- delete_local_gateway_route()
- delete_local_gateway_route_table_vpc_association()
- delete_managed_prefix_list()
- delete_nat_gateway()
- delete_network_acl()
- delete_network_acl_entry()
- delete_network_insights_analysis()
- delete_network_insights_path()
- delete_network_interface()
- delete_network_interface_permission()
- delete_placement_group()
- delete_queued_reserved_instances()
- delete_route()
- delete_route_table()
- delete_security_group()
- delete_snapshot()
- delete_spot_datafeed_subscription()
- delete_subnet()
- delete_tags()
- delete_traffic_mirror_filter()
- delete_traffic_mirror_filter_rule()
- delete_traffic_mirror_session()
- delete_traffic_mirror_target()
- delete_transit_gateway()
- delete_transit_gateway_connect()
- delete_transit_gateway_connect_peer()
- delete_transit_gateway_multicast_domain()
- delete_transit_gateway_peering_attachment()
- delete_transit_gateway_prefix_list_reference()
- delete_transit_gateway_route()
- delete_transit_gateway_route_table()
- delete_transit_gateway_vpc_attachment()
- delete_volume()
- delete_vpc()
- delete_vpc_endpoint_connection_notifications()
- delete_vpc_endpoint_service_configurations()
- delete_vpc_endpoints()
- delete_vpc_peering_connection()
- delete_vpn_connection()
- delete_vpn_connection_route()
- delete_vpn_gateway()
- deprovision_byoip_cidr()
- deregister_image()
- deregister_instance_event_notification_attributes()
- deregister_transit_gateway_multicast_group_members()
- deregister_transit_gateway_multicast_group_sources()
- describe_account_attributes()
- describe_addresses()
- describe_aggregate_id_format()
- describe_availability_zones()
- describe_bundle_tasks()
- describe_byoip_cidrs()
- describe_capacity_reservations()
- describe_carrier_gateways()
- describe_classic_link_instances()
- describe_client_vpn_authorization_rules()
- describe_client_vpn_connections()
- describe_client_vpn_endpoints()
- describe_client_vpn_routes()
- describe_client_vpn_target_networks()
- describe_coip_pools()
- describe_conversion_tasks()
- describe_customer_gateways()
- describe_dhcp_options()
- describe_egress_only_internet_gateways()
- describe_elastic_gpus()
- describe_export_image_tasks()
- describe_export_tasks()
- describe_fast_snapshot_restores()
- describe_fleet_history()
- describe_fleet_instances()
- describe_fleets()
- describe_flow_logs()
- describe_fpga_image_attribute()
- describe_fpga_images()
- describe_host_reservation_offerings()
- describe_host_reservations()
- describe_hosts()
- describe_iam_instance_profile_associations()
- describe_id_format()
- describe_identity_id_format()
- describe_image_attribute()
- describe_images()
- describe_import_image_tasks()
- describe_import_snapshot_tasks()
- describe_instance_attribute()
- describe_instance_credit_specifications()
- describe_instance_event_notification_attributes()
- describe_instance_status()
- describe_instance_type_offerings()
- describe_instance_types()
- describe_instances()
- describe_internet_gateways()
- describe_ipv6_pools()
- describe_key_pairs()
- describe_launch_template_versions()
- describe_launch_templates()
- describe_local_gateway_route_table_virtual_interface_group_associations()
- describe_local_gateway_route_table_vpc_associations()
- describe_local_gateway_route_tables()
- describe_local_gateway_virtual_interface_groups()
- describe_local_gateway_virtual_interfaces()
- describe_local_gateways()
- describe_managed_prefix_lists()
- describe_moving_addresses()
- describe_nat_gateways()
- describe_network_acls()
- describe_network_insights_analyses()
- describe_network_insights_paths()
- describe_network_interface_attribute()
- describe_network_interface_permissions()
- describe_network_interfaces()
- describe_placement_groups()
- describe_prefix_lists()
- describe_principal_id_format()
- describe_public_ipv4_pools()
- describe_regions()
- describe_reserved_instances()
- describe_reserved_instances_listings()
- describe_reserved_instances_modifications()
- describe_reserved_instances_offerings()
- describe_route_tables()
- describe_scheduled_instance_availability()
- describe_scheduled_instances()
- describe_security_group_references()
- describe_security_groups()
- describe_snapshot_attribute()
- describe_snapshots()
- describe_spot_datafeed_subscription()
- describe_spot_fleet_instances()
- describe_spot_fleet_request_history()
- describe_spot_fleet_requests()
- describe_spot_instance_requests()
- describe_spot_price_history()
- describe_stale_security_groups()
- describe_subnets()
- describe_tags()
- describe_traffic_mirror_filters()
- describe_traffic_mirror_sessions()
- describe_traffic_mirror_targets()
- describe_transit_gateway_attachments()
- describe_transit_gateway_connect_peers()
- describe_transit_gateway_connects()
- describe_transit_gateway_multicast_domains()
- describe_transit_gateway_peering_attachments()
- describe_transit_gateway_route_tables()
- describe_transit_gateway_vpc_attachments()
- describe_transit_gateways()
- describe_volume_attribute()
- describe_volume_status()
- describe_volumes()
- describe_volumes_modifications()
- describe_vpc_attribute()
- describe_vpc_classic_link()
- describe_vpc_classic_link_dns_support()
- describe_vpc_endpoint_connection_notifications()
- describe_vpc_endpoint_connections()
- describe_vpc_endpoint_service_configurations()
- describe_vpc_endpoint_service_permissions()
- describe_vpc_endpoint_services()
- describe_vpc_endpoints()
- describe_vpc_peering_connections()
- describe_vpcs()
- describe_vpn_connections()
- describe_vpn_gateways()
- detach_classic_link_vpc()
- detach_internet_gateway()
- detach_network_interface()
- detach_volume()
- detach_vpn_gateway()
- disable_ebs_encryption_by_default()
- disable_fast_snapshot_restores()
- disable_transit_gateway_route_table_propagation()
- disable_vgw_route_propagation()
- disable_vpc_classic_link()
- disable_vpc_classic_link_dns_support()
- disassociate_address()
- disassociate_client_vpn_target_network()
- disassociate_enclave_certificate_iam_role()
- disassociate_iam_instance_profile()
- disassociate_route_table()
- disassociate_subnet_cidr_block()
- disassociate_transit_gateway_multicast_domain()
- disassociate_transit_gateway_route_table()
- disassociate_vpc_cidr_block()
- enable_ebs_encryption_by_default()
- enable_fast_snapshot_restores()
- enable_transit_gateway_route_table_propagation()
- enable_vgw_route_propagation()
- enable_volume_io()
- enable_vpc_classic_link()
- enable_vpc_classic_link_dns_support()
- export_client_vpn_client_certificate_revocation_list()
- export_client_vpn_client_configuration()
- export_image()
- export_transit_gateway_routes()
- generate_presigned_url()
- get_associated_enclave_certificate_iam_roles()
- get_associated_ipv6_pool_cidrs()
- get_capacity_reservation_usage()
- get_coip_pool_usage()
- get_console_output()
- get_console_screenshot()
- get_default_credit_specification()
- get_ebs_default_kms_key_id()
- get_ebs_encryption_by_default()
- get_groups_for_capacity_reservation()
- get_host_reservation_purchase_preview()
- get_launch_template_data()
- get_managed_prefix_list_associations()
- get_managed_prefix_list_entries()
- get_paginator()
- get_password_data()
- get_reserved_instances_exchange_quote()
- get_transit_gateway_attachment_propagations()
- get_transit_gateway_multicast_domain_associations()
- get_transit_gateway_prefix_list_references()
- get_transit_gateway_route_table_associations()
- get_transit_gateway_route_table_propagations()
- get_waiter()
- import_client_vpn_client_certificate_revocation_list()
- import_image()
- import_instance()
- import_key_pair()
- import_snapshot()
- import_volume()
- modify_availability_zone_group()
- modify_capacity_reservation()
- modify_client_vpn_endpoint()
- modify_default_credit_specification()
- modify_ebs_default_kms_key_id()
- modify_fleet()
- modify_fpga_image_attribute()
- modify_hosts()
- modify_id_format()
- modify_identity_id_format()
- modify_image_attribute()
- modify_instance_attribute()
- modify_instance_capacity_reservation_attributes()
- modify_instance_credit_specification()
- modify_instance_event_start_time()
- modify_instance_metadata_options()
- modify_instance_placement()
- modify_launch_template()
- modify_managed_prefix_list()
- modify_network_interface_attribute()
- modify_reserved_instances()
- modify_snapshot_attribute()
- modify_spot_fleet_request()
- modify_subnet_attribute()
- modify_traffic_mirror_filter_network_services()
- modify_traffic_mirror_filter_rule()
- modify_traffic_mirror_session()
- modify_transit_gateway()
- modify_transit_gateway_prefix_list_reference()
- modify_transit_gateway_vpc_attachment()
- modify_volume()
- modify_volume_attribute()
- modify_vpc_attribute()
- modify_vpc_endpoint()
- modify_vpc_endpoint_connection_notification()
- modify_vpc_endpoint_service_configuration()
- modify_vpc_endpoint_service_permissions()
- modify_vpc_peering_connection_options()
- modify_vpc_tenancy()
- modify_vpn_connection()
- modify_vpn_connection_options()
- modify_vpn_tunnel_certificate()
- modify_vpn_tunnel_options()
- monitor_instances()
- move_address_to_vpc()
- provision_byoip_cidr()
- purchase_host_reservation()
- purchase_reserved_instances_offering()
- purchase_scheduled_instances()
- reboot_instances()
- register_image()
- register_instance_event_notification_attributes()
- register_transit_gateway_multicast_group_members()
- register_transit_gateway_multicast_group_sources()
- reject_transit_gateway_multicast_domain_associations()
- reject_transit_gateway_peering_attachment()
- reject_transit_gateway_vpc_attachment()
- reject_vpc_endpoint_connections()
- reject_vpc_peering_connection()
- release_address()
- release_hosts()
- replace_iam_instance_profile_association()
- replace_network_acl_association()
- replace_network_acl_entry()
- replace_route()
- replace_route_table_association()
- replace_transit_gateway_route()
- report_instance_status()
- request_spot_fleet()
- request_spot_instances()
- reset_ebs_default_kms_key_id()
- reset_fpga_image_attribute()
- reset_image_attribute()
- reset_instance_attribute()
- reset_network_interface_attribute()
- reset_snapshot_attribute()
- restore_address_to_classic()
- restore_managed_prefix_list_version()
- revoke_client_vpn_ingress()
- revoke_security_group_egress()
- revoke_security_group_ingress()
- run_instances()
- run_scheduled_instances()
- search_local_gateway_routes()
- search_transit_gateway_multicast_groups()
- search_transit_gateway_routes()
- send_diagnostic_interrupt()
- start_instances()
- start_network_insights_analysis()
- start_vpc_endpoint_service_private_dns_verification()
- stop_instances()
- terminate_client_vpn_connections()
- terminate_instances()
- unassign_ipv6_addresses()
- unassign_private_ip_addresses()
- unmonitor_instances()
- update_security_group_rule_descriptions_egress()
- update_security_group_rule_descriptions_ingress()
- withdraw_byoip_cidr()
```



## instances

### accept_reserved_instances_exchange_quote(kwargs_)

Accepts the Convertible Reserved Instance exchange quote described in the GetReservedInstancesExchangeQuote call.

**Request Syntax**

```py
response = ec2client.accept_reserved_instances_exchange_quote(
    DryRun=True|False,
    # (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response.
    # If you have the required permissions, the error response is DryRunOperation .
    # Otherwise, it is UnauthorizedOperation .
    ReservedInstanceIds=['string'],
    # (_list_) -- [REQUIRED] The IDs of the Convertible Reserved Instances to exchange for another Convertible Reserved Instance of the same or higher value.
    TargetConfigurations=[
    # (_list_) -- The configuration of the target Convertible Reserved Instance to exchange for your current Convertible Reserved Instances.
        {
            'InstanceCount': 123,
            # (integer) -- The number of instances the convertible Reserved Instance offering can be applied to. This parameter is reserved and cannot be specified in a request
            'OfferingId': 'string'
            # (string) -- [REQUIRED] The Convertible Reserved Instance offering ID.
        },
    ]
)
```

Return
- Return type: dict
- **Response Syntax**

```py
# (dict) -- The result of the exchange and whether it was successful .
{
    'ExchangeId': 'string'
    # (string) -- The ID of the successful exchange.
}
```


---


## transit_gateway

### accept_transit_gateway_multicast_domain_associations(kwargs_)

Accepts a request to associate subnets with a transit gateway multicast domain.

**Request Syntax**

```py
response = ec2client.accept_transit_gateway_multicast_domain_associations(
    TransitGatewayMulticastDomainId='string',
    # (_string_) -- The ID of the transit gateway multicast domain.
    TransitGatewayAttachmentId='string',
    # (_string_) -- The ID of the transit gateway attachment.
    SubnetIds=[
        'string',
    ],
    # The IDs of the subnets to associate with the transit gateway multicast domain.
    DryRun=True|False
    # (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response.
    # If you have the required permissions, the error response is DryRunOperation .
    # Otherwise, it is UnauthorizedOperation .
)
```

Return
- Return type: dict
- **Response Syntax**

```py
{
    'Associations': {
        # Describes the multicast domain associations.
        'TransitGatewayMulticastDomainId': 'string',
        'TransitGatewayAttachmentId': 'string',
        'ResourceId': 'string',
        'ResourceType': 'vpc'|'vpn'|'direct-connect-gateway'|'connect'|'peering'|'tgw-peering',
        # (string) -- The type of resource, for example a VPC attachment.
        'ResourceOwnerId': 'string',
        'Subnets': [
            # (list) --The subnets associated with the multicast domain.
            {
                # Describes the subnet association with the transit gateway multicast domain.
                'SubnetId': 'string',
                'State': 'pendingAcceptance'|'associating'|'associated'|'disassociating'|'disassociated'|'rejected'|'failed'
                # The state of the subnet association.
            },
        ]
    }
}
```

---

### accept_transit_gateway_peering_attachment(kwargs_)
- Accepts a transit gateway peering attachment request.
- The peering attachment must be in the pendingAcceptance state.

**Request Syntax**

```py
response = ec2client.accept_transit_gateway_peering_attachment(
    TransitGatewayAttachmentId='string',
    DryRun=True|False
)
```

Parameters

- **TransitGatewayAttachmentId** (_string_) -- [REQUIRED] The ID of the transit gateway attachment.

- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
Return
- Return type: dict
- **Response Syntax**

```py
{
    'TransitGatewayPeeringAttachment': {
        'TransitGatewayAttachmentId': 'string',
        'RequesterTgwInfo': {
            'TransitGatewayId': 'string',
            'OwnerId': 'string',
            'Region': 'string'
        },
        'AccepterTgwInfo': {
            'TransitGatewayId': 'string',
            'OwnerId': 'string',
            'Region': 'string'
        },
        'Status': {
            'Code': 'string',
            'Message': 'string'
        },
        'State': 'initiating'|'initiatingRequest'|'pendingAcceptance'|'rollingBack'|'pending'|'available'|'modifying'|'deleting'|'deleted'|'failed'|'rejected'|'rejecting'|'failing',
        'CreationTime': datetime(2015, 1, 1),
        'Tags': [
            {
                'Key': 'string',
                'Value': 'string'
            },
        ]
    }
}
```

**Response Structure**

- _(dict) --_

  - **TransitGatewayPeeringAttachment** _(dict) --_

        The transit gateway peering attachment.

      - **TransitGatewayAttachmentId** (string) -- The ID of the transit gateway peering attachment.
      - **RequesterTgwInfo** _(dict) --_

            Information about the requester transit gateway.
          - **TransitGatewayId** _(string) --_

                The ID of the transit gateway.

          - **OwnerId** _(string) --_

                The AWS account ID of the owner of the transit gateway.

          - **Region** _(string) --_

                The Region of the transit gateway.

      - **AccepterTgwInfo** _(dict) --_

            Information about the accepter transit gateway.
          - **TransitGatewayId** _(string) --_

                The ID of the transit gateway.

          - **OwnerId** _(string) --_

                The AWS account ID of the owner of the transit gateway.

          - **Region** _(string) --_

                The Region of the transit gateway.

      - **Status** _(dict) --_

            The status of the transit gateway peering attachment.
          - **Code** _(string) --_

                The status code.

          - **Message** _(string) --_

                The status message, if applicable.

      - **State** (string) -- The state of the transit gateway peering attachment. Note that the initiating state has been deprecated.
      - **CreationTime** _(datetime) --_

            The time the transit gateway peering attachment was created.
      - **Tags** (list) --The tags for the transit gateway peering attachment.
          - _(dict) --_

                Describes a tag.

              - **Key** _(string) --_

                    The key of the tag.

                    Constraints: Tag keys are case-sensitive and accept a maximum of 127 Unicode characters. May not begin with aws: .

              - **Value** _(string) --_

                    The value of the tag.

                    Constraints: Tag values are case-sensitive and accept a maximum of 255 Unicode characters.


---

### accept_transit_gateway_vpc_attachment(kwargs_)

- Accepts a request to attach a VPC to a transit gateway.
- The VPC attachment must be in the `pendingAcceptance` state.
  - Use `DescribeTransitGatewayVpcAttachments` to view your pending VPC attachment requests.
  - Use `RejectTransitGatewayVpcAttachment` to reject a VPC attachment request.

**Request Syntax**

```py
response = ec2client.accept_transit_gateway_vpc_attachment(
    TransitGatewayAttachmentId='string',
    # (_string_) -- [REQUIRED] The ID of the attachment.
    DryRun=True|False
    # (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
Return
)
```


Return
- Return type: dict
- **Response Syntax**

```py
{
    'TransitGatewayVpcAttachment': {
        'TransitGatewayAttachmentId': 'string',
        'TransitGatewayId': 'string',
        'VpcId': 'string',
        'VpcOwnerId': 'string',
        'State': 'initiating'|'initiatingRequest'|'pendingAcceptance'|'rollingBack'|'pending'|'available'|'modifying'|'deleting'|'deleted'|'failed'|'rejected'|'rejecting'|'failing',
        # (string) -- The state of the VPC attachment. Note that the initiating state has been deprecated.
        'SubnetIds': [
            'string',
        ],
        'CreationTime': datetime(2015, 1, 1),
        'Options': {
            'DnsSupport': 'enable'|'disable',  # Indicates whether DNS support is enabled.
            'Ipv6Support': 'enable'|'disable',
            'ApplianceModeSupport': 'enable'|'disable'
        },
        'Tags': [
            {
                'Key': 'string',
                'Value': 'string'
                # Tag values are case-sensitive and accept a maximum of 255 Unicode characters.
            },
        ]
    }
}
```

---


## vpc endpoint

### accept_vpc_endpoint_connections(kwargs_)

Accepts one or more interface VPC endpoint connection requests to your VPC endpoint service.

**Request Syntax**

```py
response = ec2client.accept_vpc_endpoint_connections(
    DryRun=True|False,
    ServiceId='string',
    VpcEndpointIds=[
        'string',
        # The IDs of one or more interface VPC endpoints.
    ]
)
```

Return
- Return type: dict
- **Response Syntax**

```py
{
    'Unsuccessful': [
        {
            'Error': {
                'Code': 'string',
                'Message': 'string'
            },
            'ResourceId': 'string'
        },
    ]
}
```

---

## vpc_peering

### accept_vpc_peering_connection(kwargs_)

- Accept a VPC peering connection request.
- To accept a request, the VPC peering connection must be in the pending-acceptance state, and you must be the owner of the peer VPC.
- Use `DescribeVpcPeeringConnections` to view your outstanding VPC peering connection requests.

> For an inter-Region VPC peering connection request, you must accept the VPC peering connection in the Region of the accepter VPC.

**Request Syntax**

```py

# DescribeVpcPeeringConnections
paginator = ec2client.get_paginator('describe_vpc_peering_connections')

# Accept a VPC peering connection request
response = ec2client.accept_vpc_peering_connection(
    DryRun=True|False,
    VpcPeeringConnectionId='string'
    # (_string_) -- The ID of the VPC peering connection. You must specify this parameter in the request.
)
```


Return
- Return type: dict
- **Response Syntax**

```py
{
    'VpcPeeringConnection': {
        'AccepterVpcInfo': {
            'CidrBlock': 'string',
            'Ipv6CidrBlockSet': [
                {
                    'Ipv6CidrBlock': 'string'
                },
            ],
            'CidrBlockSet': [
                {
                    'CidrBlock': 'string'
                },
            ],
            'OwnerId': 'string',
            'PeeringOptions': {
                'AllowDnsResolutionFromRemoteVpc': True|False,
                'AllowEgressFromLocalClassicLinkToRemoteVpc': True|False,
                'AllowEgressFromLocalVpcToRemoteClassicLink': True|False
            },
            'VpcId': 'string',
            'Region': 'string'
        },
        'ExpirationTime': datetime(2015, 1, 1),
        'RequesterVpcInfo': {
            'CidrBlock': 'string',
            'Ipv6CidrBlockSet': [
                {
                    'Ipv6CidrBlock': 'string'
                },
            ],
            'CidrBlockSet': [
                {
                    'CidrBlock': 'string'
                },
            ],
            'OwnerId': 'string',
            'PeeringOptions': {
                'AllowDnsResolutionFromRemoteVpc': True|False,
                'AllowEgressFromLocalClassicLinkToRemoteVpc': True|False,
                'AllowEgressFromLocalVpcToRemoteClassicLink': True|False
            },
            'VpcId': 'string',
            'Region': 'string'
        },
        'Status': {
            'Code': 'initiating-request'|'pending-acceptance'|'active'|'deleted'|'rejected'|'failed'|'expired'|'provisioning'|'deleting',
            'Message': 'string'
        },
        'Tags': [
            {
                'Key': 'string',
                'Value': 'string'
            },
        ],
        'VpcPeeringConnectionId': 'string'
    }
}
```



### advertise_byoip_cidr(kwargs_)

- Advertises an IPv4 or IPv6 address range that is provisioned for use with your AWS resources through bring your own IP addresses (BYOIP).

- You can perform this operation at most once every 10 seconds, even if you specify different address ranges each time.

We recommend that you stop advertising the BYOIP CIDR from other locations when you advertise it from AWS.
- To minimize down time, you can configure your AWS resources to use an address from a BYOIP CIDR before it is advertised, and then simultaneously stop advertising it from the current location and start advertising it through AWS.

It can take a few minutes before traffic to the specified addresses starts routing to AWS because of BGP propagation delays.

To stop advertising the BYOIP CIDR, use `WithdrawByoipCidr` .

**Request Syntax**

```py
response = ec2client.advertise_byoip_cidr(
    Cidr='string',
    # (_string_) -- [REQUIRED] The address range, in CIDR notation. This must be the exact range that you provisioned. You can't advertise only a portion of the provisioned range.
    DryRun=True|False
)
```


Return
- Return type: dict
- **Response Syntax**

```py
{
    'ByoipCidr': {
        'Cidr': 'string',
        'Description': 'string',
        'StatusMessage': 'string',
        # (string) -- Upon success, contains the ID of the address pool. Otherwise, contains an error message.
        'State': 'advertised'|'deprovisioned'|'failed-deprovision'|'failed-provision'|'pending-deprovision'|'pending-provision'|'provisioned'|'provisioned-not-publicly-advertisable'
    }
}
```


---

## AWS account.


### allocate_address(kwargs_)

- Allocates an Elastic IP address to your AWS account.
- After you <font color=blue> allocate the Elastic IP address </font> , associate it with an instance or network interface.
  - allocate an Elastic IP address from
    - an address pool owned by AWS
    - or from an address pool created from a public IPv4 address range that you have brought to AWS for use with your AWS resources using bring your own IP addresses (BYOIP).
  - An Elastic IP address is for use either in the EC2-Classic platform or in a VPC.
  - By default, you can allocate
    - 5 Elastic IP addresses for EC2-Classic per Region
    - 5 Elastic IP addresses for EC2-VPC per Region.
  - You can allocate a carrier IP address which is a public IP address from a telecommunication carrier, to a network interface which resides in a subnet in a Wavelength Zone (for example an EC2 instance).

- After you <font color=blue> release an Elastic IP address </font> , it is released to the IP address pool and can be allocated to a different AWS account.
  - [EC2-VPC] If you release an Elastic IP address, you might be able to recover it.
  - You cannot recover an Elastic IP address that you released after it is allocated to another AWS account.
  - You cannot recover an Elastic IP address for EC2-Classic.
  - To attempt to recover an Elastic IP address that you released, specify it in this operation.



**Request Syntax**

```py
response = ec2client.allocate_address(
    Domain='vpc'|'standard',
    # (_string_) -- whether the Elastic IP address is for use with instances in a VPC or instances in EC2-Classic.
    # Default: If the Region supports EC2-Classic, the default is standard . Otherwise, the default is vpc .
    Address='string',
    # (_string_) -- [EC2-VPC] The Elastic IP address to recover or an IPv4 address from an address pool.
    PublicIpv4Pool='string',
    # (_string_) -- The ID of an address pool that you own.
    # Use this parameter to let Amazon EC2 select an address from the address pool.
    # To specify a specific address from the address pool, use the Address parameter instead.
    NetworkBorderGroup='string',
    # (_string_) -- A unique set of Availability Zones, Local Zones, or Wavelength Zones from which AWS advertises IP addresses.
    # Use this parameter to limit the IP address to this location. IP addresses cannot move between network border groups.
    CustomerOwnedIpv4Pool='string',
    # (_string_) -- The ID of a customer-owned address pool. Use this parameter to let Amazon EC2 select an address from the address pool. Alternatively, specify a specific address from the address pool.
    DryRun=True|False,
    TagSpecifications=[
        # The tags to assign to the Elastic IP address.
        {
            'ResourceType': 'client-vpn-endpoint'|'customer-gateway'|'dedicated-host'|'dhcp-options'|'egress-only-internet-gateway'|'elastic-ip'|'elastic-gpu'|'export-image-task'|'export-instance-task'|'fleet'|'fpga-image'|'host-reservation'|'image'|'import-image-task'|'import-snapshot-task'|'instance'|'internet-gateway'|'key-pair'|'launch-template'|'local-gateway-route-table-vpc-association'|'natgateway'|'network-acl'|'network-interface'|'network-insights-analysis'|'network-insights-path'|'placement-group'|'reserved-instances'|'route-table'|'security-group'|'snapshot'|'spot-fleet-request'|'spot-instances-request'|'subnet'|'traffic-mirror-filter'|'traffic-mirror-session'|'traffic-mirror-target'|'transit-gateway'|'transit-gateway-attachment'|'transit-gateway-connect-peer'|'transit-gateway-multicast-domain'|'transit-gateway-route-table'|'volume'|'vpc'|'vpc-peering-connection'|'vpn-connection'|'vpn-gateway'|'vpc-flow-log',
            'Tags': [
                {
                    'Key': 'string',
                    'Value': 'string'
                },
            ]
        },
    ]
)
```

Return
- Return type: dict
- **Response Syntax**

```py
{
    'PublicIp': 'string',
    'AllocationId': 'string',
    # (string) -- [EC2-VPC] The ID that AWS assigns to represent the allocation of the Elastic IP address for use with instances in a VPC.
    'PublicIpv4Pool': 'string',
    'NetworkBorderGroup': 'string',
    # (string) -- The set of Availability Zones, Local Zones, or Wavelength Zones from which AWS advertises IP addresses.
    'Domain': 'vpc'|'standard',
    # (string) -- Indicates whether the Elastic IP address is for use with instances in a VPC (vpc ) or instances in EC2-Classic (standard ).
    'CustomerOwnedIp': 'string',
    'CustomerOwnedIpv4Pool': 'string',
    'CarrierIp': 'string'
    # (string) -- The carrier IP address. This option is only available for network interfaces which reside in a subnet in a Wavelength Zone (for example an EC2 instance).
}
```


**Examples**

```py
# This example allocates an Elastic IP address to use with an instance in a VPC.

response = ec2client.allocate_address(
    Domain='vpc',
)
print(response)

# Expected Output:
# {
#     'AllocationId': 'eipalloc-64d5890a',
#     'Domain': 'vpc',
#     'PublicIp': '203.0.113.0',
#     'ResponseMetadata': {
#         '...': '...',
#     },
# }



# This example allocates an Elastic IP address to use with an instance in EC2-Classic.

response = ec2client.allocate_address(
)
print(response)

# Expected Output:
# {
#     'Domain': 'standard',
#     'PublicIp': '198.51.100.0',
#     'ResponseMetadata': {
#         '...': '...',
#     },
# }
```



### allocate_hosts(kwargs_)

- Allocates a Dedicated Host to your account. At a minimum, specify the supported instance type or instance family, the Availability Zone in which to allocate the host, and the number of hosts to allocate.

**Request Syntax**

```py
response = ec2client.allocate_hosts(
    AutoPlacement='on'|'off',
    AvailabilityZone='string',
    ClientToken='string',
    InstanceType='string',
    InstanceFamily='string',
    Quantity=123,
    TagSpecifications=[
        {
            'ResourceType': 'client-vpn-endpoint'|'customer-gateway'|'dedicated-host'|'dhcp-options'|'egress-only-internet-gateway'|'elastic-ip'|'elastic-gpu'|'export-image-task'|'export-instance-task'|'fleet'|'fpga-image'|'host-reservation'|'image'|'import-image-task'|'import-snapshot-task'|'instance'|'internet-gateway'|'key-pair'|'launch-template'|'local-gateway-route-table-vpc-association'|'natgateway'|'network-acl'|'network-interface'|'network-insights-analysis'|'network-insights-path'|'placement-group'|'reserved-instances'|'route-table'|'security-group'|'snapshot'|'spot-fleet-request'|'spot-instances-request'|'subnet'|'traffic-mirror-filter'|'traffic-mirror-session'|'traffic-mirror-target'|'transit-gateway'|'transit-gateway-attachment'|'transit-gateway-connect-peer'|'transit-gateway-multicast-domain'|'transit-gateway-route-table'|'volume'|'vpc'|'vpc-peering-connection'|'vpn-connection'|'vpn-gateway'|'vpc-flow-log',
            'Tags': [
                {
                    'Key': 'string',
                    'Value': 'string'
                },
            ]
        },
    ],
    HostRecovery='on'|'off'
)
```

Parameters

- **AutoPlacement** (_string_) --

    Indicates whether the host accepts any untargeted instance launches that match its instance type configuration, or if it only accepts Host tenancy instance launches that specify its unique host ID. For more information, see [Understanding Instance Placement and Host Affinity](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/how-dedicated-hosts-work.html#dedicated-hosts-understanding) in the _Amazon EC2 User Guide for Linux Instances_ .

    Default: on

- **AvailabilityZone** (_string_) -- [REQUIRED] The Availability Zone in which to allocate the Dedicated Host.

- **ClientToken** (_string_) -- Unique, case-sensitive identifier that you provide to ensure the idempotency of the request. For more information, see [How to Ensure Idempotency](https://docs.aws.amazon.com/AWSEC2/latest/APIReference/Run_Instance_Idempotency.html) .
- **InstanceType** (_string_) --

    Specifies the instance type to be supported by the Dedicated Hosts. If you specify an instance type, the Dedicated Hosts support instances of the specified instance type only.

    If you want the Dedicated Hosts to support multiple instance types in a specific instance family, omit this parameter and specify **InstanceFamily** instead. You cannot specify **InstanceType** and **InstanceFamily** in the same request.

- **InstanceFamily** (_string_) --

    Specifies the instance family to be supported by the Dedicated Hosts. If you specify an instance family, the Dedicated Hosts support multiple instance types within that instance family.

    If you want the Dedicated Hosts to support a specific instance type only, omit this parameter and specify **InstanceType** instead. You cannot specify **InstanceFamily** and **InstanceType** in the same request.

- **Quantity** (_integer_) --

    **[REQUIRED]**

    The number of Dedicated Hosts to allocate to your account with these parameters.

- **TagSpecifications** (_list_) --

    The tags to apply to the Dedicated Host during creation.

  - _(dict) --_

        The tags to apply to a resource when the resource is being created.

      - **ResourceType** (string) -- The type of resource to tag. Currently, the resource types that support tagging on creation are: capacity-reservation | carrier-gateway | client-vpn-endpoint | customer-gateway | dedicated-host | dhcp-options | egress-only-internet-gateway | elastic-ip | elastic-gpu | export-image-task | export-instance-task | fleet | fpga-image | host-reservation | image | import-image-task | import-snapshot-task | instance | internet-gateway | ipv4pool-ec2 | ipv6pool-ec2 | key-pair | launch-template | local-gateway-route-table-vpc-association | placement-group | prefix-list | natgateway | network-acl | network-interface | reserved-instances [|](#id13)route-table | security-group | snapshot | spot-fleet-request | spot-instances-request | snapshot | subnet | traffic-mirror-filter | traffic-mirror-session | traffic-mirror-target | transit-gateway | transit-gateway-attachment | transit-gateway-multicast-domain | transit-gateway-route-table | volume [|](#id15)vpc | vpc-peering-connection | vpc-endpoint (for interface and gateway endpoints) | vpc-endpoint-service (for AWS PrivateLink) | vpc-flow-log | vpn-connection | vpn-gateway .

            To tag a resource after it has been created, see [CreateTags](https://docs.aws.amazon.com/AWSEC2/latest/APIReference/API_CreateTags.html) .
      - **Tags** (list) --The tags to apply to the resource.
          - _(dict) --_

                Describes a tag.

              - **Key** _(string) --_

                    The key of the tag.

                    Constraints: Tag keys are case-sensitive and accept a maximum of 127 Unicode characters. May not begin with aws: .

              - **Value** _(string) --_

                    The value of the tag.

                    Constraints: Tag values are case-sensitive and accept a maximum of 255 Unicode characters.

- **HostRecovery** (_string_) --

    Indicates whether to enable or disable host recovery for the Dedicated Host. Host recovery is disabled by default. For more information, see [Host Recovery](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/dedicated-hosts-recovery.html) in the _Amazon Elastic Compute Cloud User Guide_ .

    Default: off

Return
- Return type: dict
- **Response Syntax**

```py
{
    'HostIds': [
        'string',
    ]
}
```

**Response Structure**

- (dict) -- Contains the output of AllocateHosts.

  - **HostIds** _(list) --_

        The ID of the allocated Dedicated Host. This is used to launch an instance onto a specific host.

      - _(string) --_

apply_security_groups_to_client_vpn_target_network(kwargs_)[¶](#EC2.Client.apply_security_groups_to_client_vpn_target_network "Permalink to this definition")

Applies a security group to the association between the target network and the Client VPN endpoint. This action replaces the existing security groups with the specified security groups.

**Request Syntax**

```py
response = ec2client.apply_security_groups_to_client_vpn_target_network(
    ClientVpnEndpointId='string',
    VpcId='string',
    SecurityGroupIds=[
        'string',
    ],
    DryRun=True|False
)
```

Parameters

- **ClientVpnEndpointId** (_string_) -- [REQUIRED] The ID of the Client VPN endpoint.

- **VpcId** (_string_) -- [REQUIRED] The ID of the VPC in which the associated target network is located.

- **SecurityGroupIds** (_list_) --

    **[REQUIRED]**

    The IDs of the security groups to apply to the associated target network. Up to 5 security groups can be applied to an associated target network.

  - _(string) --_
- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
Return
- Return type: dict
- **Response Syntax**

```py
{
    'SecurityGroupIds': [
        'string',
    ]
}
```

**Response Structure**

- _(dict) --_

  - **SecurityGroupIds** _(list) --_

        The IDs of the applied security groups.

      - _(string) --_



---

### assign_ipv6_addresses(kwargs_)

- Assigns one or more IPv6 addresses to the specified network interface.
  - specify one or more specific IPv6 addresses,
  - specify the number of IPv6 addresses to be automatically assigned from within the subnet's IPv6 CIDR block range.
- You can assign as many IPv6 addresses to a network interface as you can assign private IPv4 addresses, and the limit varies per instance type.

You must specify either the IPv6 addresses or the IPv6 address count in the request.

**Request Syntax**

```py
response = ec2client.assign_ipv6_addresses(
    Ipv6AddressCount=123,
    # (_integer_) -- The number of IPv6 addresses to assign to the network interface.
    # Amazon EC2 automatically selects the IPv6 addresses from the subnet range. You can't use this option if specifying specific IPv6 addresses.
    Ipv6Addresses=[
        'string',
        # _(string) -- One or more specific IPv6 addresses to be assigned to the network interface.
        # You can't use this option if you're specifying a number of IPv6 addresses.
    ],
    NetworkInterfaceId='string'
    # (_string_) -- [REQUIRED] The ID of the network interface.
)
```


Return
- Return type: dict
- **Response Syntax**
