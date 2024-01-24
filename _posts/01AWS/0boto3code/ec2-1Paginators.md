
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


# EC2 - Paginators

The available paginators are:

- [EC2.Paginator.DescribeByoipCidrs](#EC2.Paginator.DescribeByoipCidrs "EC2.Paginator.DescribeByoipCidrs")
- [EC2.Paginator.DescribeCapacityReservations](#EC2.Paginator.DescribeCapacityReservations "EC2.Paginator.DescribeCapacityReservations")
- [EC2.Paginator.DescribeCarrierGateways](#EC2.Paginator.DescribeCarrierGateways "EC2.Paginator.DescribeCarrierGateways")
- [EC2.Paginator.DescribeClassicLinkInstances](#EC2.Paginator.DescribeClassicLinkInstances "EC2.Paginator.DescribeClassicLinkInstances")
- [EC2.Paginator.DescribeClientVpnAuthorizationRules](#EC2.Paginator.DescribeClientVpnAuthorizationRules "EC2.Paginator.DescribeClientVpnAuthorizationRules")
- [EC2.Paginator.DescribeClientVpnConnections](#EC2.Paginator.DescribeClientVpnConnections "EC2.Paginator.DescribeClientVpnConnections")
- [EC2.Paginator.DescribeClientVpnEndpoints](#EC2.Paginator.DescribeClientVpnEndpoints "EC2.Paginator.DescribeClientVpnEndpoints")
- [EC2.Paginator.DescribeClientVpnRoutes](#EC2.Paginator.DescribeClientVpnRoutes "EC2.Paginator.DescribeClientVpnRoutes")
- [EC2.Paginator.DescribeClientVpnTargetNetworks](#EC2.Paginator.DescribeClientVpnTargetNetworks "EC2.Paginator.DescribeClientVpnTargetNetworks")
- [EC2.Paginator.DescribeCoipPools](#EC2.Paginator.DescribeCoipPools "EC2.Paginator.DescribeCoipPools")
- [EC2.Paginator.DescribeDhcpOptions](#EC2.Paginator.DescribeDhcpOptions "EC2.Paginator.DescribeDhcpOptions")
- [EC2.Paginator.DescribeEgressOnlyInternetGateways](#EC2.Paginator.DescribeEgressOnlyInternetGateways "EC2.Paginator.DescribeEgressOnlyInternetGateways")
- [EC2.Paginator.DescribeExportImageTasks](#EC2.Paginator.DescribeExportImageTasks "EC2.Paginator.DescribeExportImageTasks")
- [EC2.Paginator.DescribeFastSnapshotRestores](#EC2.Paginator.DescribeFastSnapshotRestores "EC2.Paginator.DescribeFastSnapshotRestores")
- [EC2.Paginator.DescribeFleets](#EC2.Paginator.DescribeFleets "EC2.Paginator.DescribeFleets")
- [EC2.Paginator.DescribeFlowLogs](#EC2.Paginator.DescribeFlowLogs "EC2.Paginator.DescribeFlowLogs")
- [EC2.Paginator.DescribeFpgaImages](#EC2.Paginator.DescribeFpgaImages "EC2.Paginator.DescribeFpgaImages")
- [EC2.Paginator.DescribeHostReservationOfferings](#EC2.Paginator.DescribeHostReservationOfferings "EC2.Paginator.DescribeHostReservationOfferings")
- [EC2.Paginator.DescribeHostReservations](#EC2.Paginator.DescribeHostReservations "EC2.Paginator.DescribeHostReservations")
- [EC2.Paginator.DescribeHosts](#EC2.Paginator.DescribeHosts "EC2.Paginator.DescribeHosts")
- [EC2.Paginator.DescribeIamInstanceProfileAssociations](#EC2.Paginator.DescribeIamInstanceProfileAssociations "EC2.Paginator.DescribeIamInstanceProfileAssociations")
- [EC2.Paginator.DescribeImportImageTasks](#EC2.Paginator.DescribeImportImageTasks "EC2.Paginator.DescribeImportImageTasks")
- [EC2.Paginator.DescribeImportSnapshotTasks](#EC2.Paginator.DescribeImportSnapshotTasks "EC2.Paginator.DescribeImportSnapshotTasks")
- [EC2.Paginator.DescribeInstanceCreditSpecifications](#EC2.Paginator.DescribeInstanceCreditSpecifications "EC2.Paginator.DescribeInstanceCreditSpecifications")
- [EC2.Paginator.DescribeInstanceStatus](#EC2.Paginator.DescribeInstanceStatus "EC2.Paginator.DescribeInstanceStatus")
- [EC2.Paginator.DescribeInstanceTypeOfferings](#EC2.Paginator.DescribeInstanceTypeOfferings "EC2.Paginator.DescribeInstanceTypeOfferings")
- [EC2.Paginator.DescribeInstanceTypes](#EC2.Paginator.DescribeInstanceTypes "EC2.Paginator.DescribeInstanceTypes")
- [EC2.Paginator.DescribeInstances](#EC2.Paginator.DescribeInstances "EC2.Paginator.DescribeInstances")
- [EC2.Paginator.DescribeInternetGateways](#EC2.Paginator.DescribeInternetGateways "EC2.Paginator.DescribeInternetGateways")
- [EC2.Paginator.DescribeIpv6Pools](#EC2.Paginator.DescribeIpv6Pools "EC2.Paginator.DescribeIpv6Pools")
- [EC2.Paginator.DescribeLaunchTemplateVersions](#EC2.Paginator.DescribeLaunchTemplateVersions "EC2.Paginator.DescribeLaunchTemplateVersions")
- [EC2.Paginator.DescribeLaunchTemplates](#EC2.Paginator.DescribeLaunchTemplates "EC2.Paginator.DescribeLaunchTemplates")
- [EC2.Paginator.DescribeLocalGatewayRouteTableVirtualInterfaceGroupAssociations](#EC2.Paginator.DescribeLocalGatewayRouteTableVirtualInterfaceGroupAssociations "EC2.Paginator.DescribeLocalGatewayRouteTableVirtualInterfaceGroupAssociations")
- [EC2.Paginator.DescribeLocalGatewayRouteTableVpcAssociations](#EC2.Paginator.DescribeLocalGatewayRouteTableVpcAssociations "EC2.Paginator.DescribeLocalGatewayRouteTableVpcAssociations")
- [EC2.Paginator.DescribeLocalGatewayRouteTables](#EC2.Paginator.DescribeLocalGatewayRouteTables "EC2.Paginator.DescribeLocalGatewayRouteTables")
- [EC2.Paginator.DescribeLocalGatewayVirtualInterfaceGroups](#EC2.Paginator.DescribeLocalGatewayVirtualInterfaceGroups "EC2.Paginator.DescribeLocalGatewayVirtualInterfaceGroups")
- [EC2.Paginator.DescribeLocalGatewayVirtualInterfaces](#EC2.Paginator.DescribeLocalGatewayVirtualInterfaces "EC2.Paginator.DescribeLocalGatewayVirtualInterfaces")
- [EC2.Paginator.DescribeLocalGateways](#EC2.Paginator.DescribeLocalGateways "EC2.Paginator.DescribeLocalGateways")
- [EC2.Paginator.DescribeManagedPrefixLists](#EC2.Paginator.DescribeManagedPrefixLists "EC2.Paginator.DescribeManagedPrefixLists")
- [EC2.Paginator.DescribeMovingAddresses](#EC2.Paginator.DescribeMovingAddresses "EC2.Paginator.DescribeMovingAddresses")
- [EC2.Paginator.DescribeNatGateways](#EC2.Paginator.DescribeNatGateways "EC2.Paginator.DescribeNatGateways")
- [EC2.Paginator.DescribeNetworkAcls](#EC2.Paginator.DescribeNetworkAcls "EC2.Paginator.DescribeNetworkAcls")
- [EC2.Paginator.DescribeNetworkInsightsAnalyses](#EC2.Paginator.DescribeNetworkInsightsAnalyses "EC2.Paginator.DescribeNetworkInsightsAnalyses")
- [EC2.Paginator.DescribeNetworkInsightsPaths](#EC2.Paginator.DescribeNetworkInsightsPaths "EC2.Paginator.DescribeNetworkInsightsPaths")
- [EC2.Paginator.DescribeNetworkInterfacePermissions](#EC2.Paginator.DescribeNetworkInterfacePermissions "EC2.Paginator.DescribeNetworkInterfacePermissions")
- [EC2.Paginator.DescribeNetworkInterfaces](#EC2.Paginator.DescribeNetworkInterfaces "EC2.Paginator.DescribeNetworkInterfaces")
- [EC2.Paginator.DescribePrefixLists](#EC2.Paginator.DescribePrefixLists "EC2.Paginator.DescribePrefixLists")
- [EC2.Paginator.DescribePrincipalIdFormat](#EC2.Paginator.DescribePrincipalIdFormat "EC2.Paginator.DescribePrincipalIdFormat")
- [EC2.Paginator.DescribePublicIpv4Pools](#EC2.Paginator.DescribePublicIpv4Pools "EC2.Paginator.DescribePublicIpv4Pools")
- [EC2.Paginator.DescribeReservedInstancesModifications](#EC2.Paginator.DescribeReservedInstancesModifications "EC2.Paginator.DescribeReservedInstancesModifications")
- [EC2.Paginator.DescribeReservedInstancesOfferings](#EC2.Paginator.DescribeReservedInstancesOfferings "EC2.Paginator.DescribeReservedInstancesOfferings")
- [EC2.Paginator.DescribeRouteTables](#EC2.Paginator.DescribeRouteTables "EC2.Paginator.DescribeRouteTables")
- [EC2.Paginator.DescribeScheduledInstanceAvailability](#EC2.Paginator.DescribeScheduledInstanceAvailability "EC2.Paginator.DescribeScheduledInstanceAvailability")
- [EC2.Paginator.DescribeScheduledInstances](#EC2.Paginator.DescribeScheduledInstances "EC2.Paginator.DescribeScheduledInstances")
- [EC2.Paginator.DescribeSecurityGroups](#EC2.Paginator.DescribeSecurityGroups "EC2.Paginator.DescribeSecurityGroups")
- [EC2.Paginator.DescribeSnapshots](#EC2.Paginator.DescribeSnapshots "EC2.Paginator.DescribeSnapshots")
- [EC2.Paginator.DescribeSpotFleetInstances](#EC2.Paginator.DescribeSpotFleetInstances "EC2.Paginator.DescribeSpotFleetInstances")
- [EC2.Paginator.DescribeSpotFleetRequests](#EC2.Paginator.DescribeSpotFleetRequests "EC2.Paginator.DescribeSpotFleetRequests")
- [EC2.Paginator.DescribeSpotInstanceRequests](#EC2.Paginator.DescribeSpotInstanceRequests "EC2.Paginator.DescribeSpotInstanceRequests")
- [EC2.Paginator.DescribeSpotPriceHistory](#EC2.Paginator.DescribeSpotPriceHistory "EC2.Paginator.DescribeSpotPriceHistory")
- [EC2.Paginator.DescribeStaleSecurityGroups](#EC2.Paginator.DescribeStaleSecurityGroups "EC2.Paginator.DescribeStaleSecurityGroups")
- [EC2.Paginator.DescribeSubnets](#EC2.Paginator.DescribeSubnets "EC2.Paginator.DescribeSubnets")
- [EC2.Paginator.DescribeTags](#EC2.Paginator.DescribeTags "EC2.Paginator.DescribeTags")
- [EC2.Paginator.DescribeTrafficMirrorFilters](#EC2.Paginator.DescribeTrafficMirrorFilters "EC2.Paginator.DescribeTrafficMirrorFilters")
- [EC2.Paginator.DescribeTrafficMirrorSessions](#EC2.Paginator.DescribeTrafficMirrorSessions "EC2.Paginator.DescribeTrafficMirrorSessions")
- [EC2.Paginator.DescribeTrafficMirrorTargets](#EC2.Paginator.DescribeTrafficMirrorTargets "EC2.Paginator.DescribeTrafficMirrorTargets")
- [EC2.Paginator.DescribeTransitGatewayAttachments](#EC2.Paginator.DescribeTransitGatewayAttachments "EC2.Paginator.DescribeTransitGatewayAttachments")
- [EC2.Paginator.DescribeTransitGatewayConnectPeers](#EC2.Paginator.DescribeTransitGatewayConnectPeers "EC2.Paginator.DescribeTransitGatewayConnectPeers")
- [EC2.Paginator.DescribeTransitGatewayConnects](#EC2.Paginator.DescribeTransitGatewayConnects "EC2.Paginator.DescribeTransitGatewayConnects")
- [EC2.Paginator.DescribeTransitGatewayMulticastDomains](#EC2.Paginator.DescribeTransitGatewayMulticastDomains "EC2.Paginator.DescribeTransitGatewayMulticastDomains")
- [EC2.Paginator.DescribeTransitGatewayPeeringAttachments](#EC2.Paginator.DescribeTransitGatewayPeeringAttachments "EC2.Paginator.DescribeTransitGatewayPeeringAttachments")
- [EC2.Paginator.DescribeTransitGatewayRouteTables](#EC2.Paginator.DescribeTransitGatewayRouteTables "EC2.Paginator.DescribeTransitGatewayRouteTables")
- [EC2.Paginator.DescribeTransitGatewayVpcAttachments](#EC2.Paginator.DescribeTransitGatewayVpcAttachments "EC2.Paginator.DescribeTransitGatewayVpcAttachments")
- [EC2.Paginator.DescribeTransitGateways](#EC2.Paginator.DescribeTransitGateways "EC2.Paginator.DescribeTransitGateways")
- [EC2.Paginator.DescribeVolumeStatus](#EC2.Paginator.DescribeVolumeStatus "EC2.Paginator.DescribeVolumeStatus")
- [EC2.Paginator.DescribeVolumes](#EC2.Paginator.DescribeVolumes "EC2.Paginator.DescribeVolumes")
- [EC2.Paginator.DescribeVolumesModifications](#EC2.Paginator.DescribeVolumesModifications "EC2.Paginator.DescribeVolumesModifications")
- [EC2.Paginator.DescribeVpcClassicLinkDnsSupport](#EC2.Paginator.DescribeVpcClassicLinkDnsSupport "EC2.Paginator.DescribeVpcClassicLinkDnsSupport")
- [EC2.Paginator.DescribeVpcEndpointConnectionNotifications](#EC2.Paginator.DescribeVpcEndpointConnectionNotifications "EC2.Paginator.DescribeVpcEndpointConnectionNotifications")
- [EC2.Paginator.DescribeVpcEndpointConnections](#EC2.Paginator.DescribeVpcEndpointConnections "EC2.Paginator.DescribeVpcEndpointConnections")
- [EC2.Paginator.DescribeVpcEndpointServiceConfigurations](#EC2.Paginator.DescribeVpcEndpointServiceConfigurations "EC2.Paginator.DescribeVpcEndpointServiceConfigurations")
- [EC2.Paginator.DescribeVpcEndpointServicePermissions](#EC2.Paginator.DescribeVpcEndpointServicePermissions "EC2.Paginator.DescribeVpcEndpointServicePermissions")
- [EC2.Paginator.DescribeVpcEndpointServices](#EC2.Paginator.DescribeVpcEndpointServices "EC2.Paginator.DescribeVpcEndpointServices")
- [EC2.Paginator.DescribeVpcEndpoints](#EC2.Paginator.DescribeVpcEndpoints "EC2.Paginator.DescribeVpcEndpoints")
- [EC2.Paginator.DescribeVpcPeeringConnections](#EC2.Paginator.DescribeVpcPeeringConnections "EC2.Paginator.DescribeVpcPeeringConnections")
- [EC2.Paginator.DescribeVpcs](#EC2.Paginator.DescribeVpcs "EC2.Paginator.DescribeVpcs")
- [EC2.Paginator.GetAssociatedIpv6PoolCidrs](#EC2.Paginator.GetAssociatedIpv6PoolCidrs "EC2.Paginator.GetAssociatedIpv6PoolCidrs")
- [EC2.Paginator.GetGroupsForCapacityReservation](#EC2.Paginator.GetGroupsForCapacityReservation "EC2.Paginator.GetGroupsForCapacityReservation")
- [EC2.Paginator.GetManagedPrefixListAssociations](#EC2.Paginator.GetManagedPrefixListAssociations "EC2.Paginator.GetManagedPrefixListAssociations")
- [EC2.Paginator.GetManagedPrefixListEntries](#EC2.Paginator.GetManagedPrefixListEntries "EC2.Paginator.GetManagedPrefixListEntries")
- [EC2.Paginator.GetTransitGatewayAttachmentPropagations](#EC2.Paginator.GetTransitGatewayAttachmentPropagations "EC2.Paginator.GetTransitGatewayAttachmentPropagations")
- [EC2.Paginator.GetTransitGatewayMulticastDomainAssociations](#EC2.Paginator.GetTransitGatewayMulticastDomainAssociations "EC2.Paginator.GetTransitGatewayMulticastDomainAssociations")
- [EC2.Paginator.GetTransitGatewayPrefixListReferences](#EC2.Paginator.GetTransitGatewayPrefixListReferences "EC2.Paginator.GetTransitGatewayPrefixListReferences")
- [EC2.Paginator.GetTransitGatewayRouteTableAssociations](#EC2.Paginator.GetTransitGatewayRouteTableAssociations "EC2.Paginator.GetTransitGatewayRouteTableAssociations")
- [EC2.Paginator.GetTransitGatewayRouteTablePropagations](#EC2.Paginator.GetTransitGatewayRouteTablePropagations "EC2.Paginator.GetTransitGatewayRouteTablePropagations")
- [EC2.Paginator.SearchLocalGatewayRoutes](#EC2.Paginator.SearchLocalGatewayRoutes "EC2.Paginator.SearchLocalGatewayRoutes")
- [EC2.Paginator.SearchTransitGatewayMulticastGroups](#EC2.Paginator.SearchTransitGatewayMulticastGroups "EC2.Paginator.SearchTransitGatewayMulticastGroups")

_class_ EC2.Paginator.DescribeByoipCidrs

paginator = client.get_paginator('describe_byoip_cidrs')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_byoip_cidrs()](#EC2.Client.describe_byoip_cidrs "EC2.Client.describe_byoip_cidrs").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeByoipCidrs)

**Request Syntax**

response_iterator = paginator.paginate(
    DryRun=True|False,
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'ByoipCidrs': [
        {
            'Cidr': 'string',
            'Description': 'string',
            'StatusMessage': 'string',
            'State': 'advertised'|'deprovisioned'|'failed-deprovision'|'failed-provision'|'pending-deprovision'|'pending-provision'|'provisioned'|'provisioned-not-publicly-advertisable'
        },
    ],

}

**Response Structure**

- _(dict) --_

    - **ByoipCidrs** _(list) --_

        Information about your address ranges.

        - _(dict) --_

            Information about an address range that is provisioned for use with your AWS resources through bring your own IP addresses (BYOIP).

            - **Cidr** _(string) --_

                The address range, in CIDR notation.

            - **Description** _(string) --_

                The description of the address range.

            - **StatusMessage** _(string) --_

                Upon success, contains the ID of the address pool. Otherwise, contains an error message.

            - **State** _(string) --_

                The state of the address pool.


_class_ EC2.Paginator.DescribeCapacityReservations

paginator = client.get_paginator('describe_capacity_reservations')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_capacity_reservations()](#EC2.Client.describe_capacity_reservations "EC2.Client.describe_capacity_reservations").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeCapacityReservations)

**Request Syntax**

response_iterator = paginator.paginate(
    CapacityReservationIds=[
        'string',
    ],
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    DryRun=True|False,
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **CapacityReservationIds** (_list_) --

    The ID of the Capacity Reservation.

    - _(string) --_
- **Filters** (_list_) --

    One or more filters.

    - instance-type - The type of instance for which the Capacity Reservation reserves capacity.
    - owner-id - The ID of the AWS account that owns the Capacity Reservation.
    - availability-zone-id - The Availability Zone ID of the Capacity Reservation.
    - instance-platform - The type of operating system for which the Capacity Reservation reserves capacity.
    - availability-zone - The Availability Zone ID of the Capacity Reservation.
    - tenancy - Indicates the tenancy of the Capacity Reservation. A Capacity Reservation can have one of the following tenancy settings:
        - default - The Capacity Reservation is created on hardware that is shared with other AWS accounts.
        - dedicated - The Capacity Reservation is created on single-tenant hardware that is dedicated to a single AWS account.
    - state - The current state of the Capacity Reservation. A Capacity Reservation can be in one of the following states:
        - active - The Capacity Reservation is active and the capacity is available for your use.
        - expired - The Capacity Reservation expired automatically at the date and time specified in your request. The reserved capacity is no longer available for your use.
        - cancelled - The Capacity Reservation was manually cancelled. The reserved capacity is no longer available for your use.
        - pending - The Capacity Reservation request was successful but the capacity provisioning is still pending.
        - failed - The Capacity Reservation request has failed. A request might fail due to invalid request parameters, capacity constraints, or instance limit constraints. Failed requests are retained for 60 minutes.
    - end-date - The date and time at which the Capacity Reservation expires. When a Capacity Reservation expires, the reserved capacity is released and you can no longer launch instances into it. The Capacity Reservation's state changes to expired when it reaches its end date and time.
    - end-date-type - Indicates the way in which the Capacity Reservation ends. A Capacity Reservation can have one of the following end types:
        - unlimited - The Capacity Reservation remains active until you explicitly cancel it.
        - limited - The Capacity Reservation expires automatically at a specified date and time.
    - instance-match-criteria - Indicates the type of instance launches that the Capacity Reservation accepts. The options include:
        - open - The Capacity Reservation accepts all instances that have matching attributes (instance type, platform, and Availability Zone). Instances that have matching attributes launch into the Capacity Reservation automatically without specifying any additional parameters.
        - targeted - The Capacity Reservation only accepts instances that have matching attributes (instance type, platform, and Availability Zone), and explicitly target the Capacity Reservation. This ensures that only permitted instances can use the reserved capacity.

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'CapacityReservations': [
        {
            'CapacityReservationId': 'string',
            'OwnerId': 'string',
            'CapacityReservationArn': 'string',
            'AvailabilityZoneId': 'string',
            'InstanceType': 'string',
            'InstancePlatform': 'Linux/UNIX'|'Red Hat Enterprise Linux'|'SUSE Linux'|'Windows'|'Windows with SQL Server'|'Windows with SQL Server Enterprise'|'Windows with SQL Server Standard'|'Windows with SQL Server Web'|'Linux with SQL Server Standard'|'Linux with SQL Server Web'|'Linux with SQL Server Enterprise',
            'AvailabilityZone': 'string',
            'Tenancy': 'default'|'dedicated',
            'TotalInstanceCount': 123,
            'AvailableInstanceCount': 123,
            'EbsOptimized': True|False,
            'EphemeralStorage': True|False,
            'State': 'active'|'expired'|'cancelled'|'pending'|'failed',
            'EndDate': datetime(2015, 1, 1),
            'EndDateType': 'unlimited'|'limited',
            'InstanceMatchCriteria': 'open'|'targeted',
            'CreateDate': datetime(2015, 1, 1),
            'Tags': [
                {
                    'Key': 'string',
                    'Value': 'string'
                },
            ]
        },
    ]
}

**Response Structure**

- _(dict) --_

    - **CapacityReservations** _(list) --_

        Information about the Capacity Reservations.

        - _(dict) --_

            Describes a Capacity Reservation.

            - **CapacityReservationId** _(string) --_

                The ID of the Capacity Reservation.

            - **OwnerId** _(string) --_

                The ID of the AWS account that owns the Capacity Reservation.

            - **CapacityReservationArn** _(string) --_

                The Amazon Resource Name (ARN) of the Capacity Reservation.

            - **AvailabilityZoneId** _(string) --_

                The Availability Zone ID of the Capacity Reservation.

            - **InstanceType** _(string) --_

                The type of instance for which the Capacity Reservation reserves capacity.

            - **InstancePlatform** _(string) --_

                The type of operating system for which the Capacity Reservation reserves capacity.

            - **AvailabilityZone** _(string) --_

                The Availability Zone in which the capacity is reserved.

            - **Tenancy** _(string) --_

                Indicates the tenancy of the Capacity Reservation. A Capacity Reservation can have one of the following tenancy settings:

                - default - The Capacity Reservation is created on hardware that is shared with other AWS accounts.
                - dedicated - The Capacity Reservation is created on single-tenant hardware that is dedicated to a single AWS account.
            - **TotalInstanceCount** _(integer) --_

                The total number of instances for which the Capacity Reservation reserves capacity.

            - **AvailableInstanceCount** _(integer) --_

                The remaining capacity. Indicates the number of instances that can be launched in the Capacity Reservation.

            - **EbsOptimized** _(boolean) --_

                Indicates whether the Capacity Reservation supports EBS-optimized instances. This optimization provides dedicated throughput to Amazon EBS and an optimized configuration stack to provide optimal I/O performance. This optimization isn't available with all instance types. Additional usage charges apply when using an EBS- optimized instance.

            - **EphemeralStorage** _(boolean) --_

                Indicates whether the Capacity Reservation supports instances with temporary, block-level storage.

            - **State** _(string) --_

                The current state of the Capacity Reservation. A Capacity Reservation can be in one of the following states:

                - active - The Capacity Reservation is active and the capacity is available for your use.
                - expired - The Capacity Reservation expired automatically at the date and time specified in your request. The reserved capacity is no longer available for your use.
                - cancelled - The Capacity Reservation was manually cancelled. The reserved capacity is no longer available for your use.
                - pending - The Capacity Reservation request was successful but the capacity provisioning is still pending.
                - failed - The Capacity Reservation request has failed. A request might fail due to invalid request parameters, capacity constraints, or instance limit constraints. Failed requests are retained for 60 minutes.
            - **EndDate** _(datetime) --_

                The date and time at which the Capacity Reservation expires. When a Capacity Reservation expires, the reserved capacity is released and you can no longer launch instances into it. The Capacity Reservation's state changes to expired when it reaches its end date and time.

            - **EndDateType** _(string) --_

                Indicates the way in which the Capacity Reservation ends. A Capacity Reservation can have one of the following end types:

                - unlimited - The Capacity Reservation remains active until you explicitly cancel it.
                - limited - The Capacity Reservation expires automatically at a specified date and time.
            - **InstanceMatchCriteria** _(string) --_

                Indicates the type of instance launches that the Capacity Reservation accepts. The options include:

                - open - The Capacity Reservation accepts all instances that have matching attributes (instance type, platform, and Availability Zone). Instances that have matching attributes launch into the Capacity Reservation automatically without specifying any additional parameters.
                - targeted - The Capacity Reservation only accepts instances that have matching attributes (instance type, platform, and Availability Zone), and explicitly target the Capacity Reservation. This ensures that only permitted instances can use the reserved capacity.
            - **CreateDate** _(datetime) --_

                The date and time at which the Capacity Reservation was created.

            - **Tags** _(list) --_

                Any tags assigned to the Capacity Reservation.

                - _(dict) --_

                    Describes a tag.

                    - **Key** _(string) --_

                        The key of the tag.

                        Constraints: Tag keys are case-sensitive and accept a maximum of 127 Unicode characters. May not begin with aws: .

                    - **Value** _(string) --_

                        The value of the tag.

                        Constraints: Tag values are case-sensitive and accept a maximum of 255 Unicode characters.


_class_ EC2.Paginator.DescribeCarrierGateways

paginator = client.get_paginator('describe_carrier_gateways')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_carrier_gateways()](#EC2.Client.describe_carrier_gateways "EC2.Client.describe_carrier_gateways").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeCarrierGateways)

**Request Syntax**

response_iterator = paginator.paginate(
    CarrierGatewayIds=[
        'string',
    ],
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    DryRun=True|False,
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **CarrierGatewayIds** (_list_) --

    One or more carrier gateway IDs.

    - _(string) --_
- **Filters** (_list_) --

    One or more filters.

    - carrier-gateway-id - The ID of the carrier gateway.
    - state - The state of the carrier gateway (pending | failed | available | deleting | deleted ).
    - owner-id - The AWS account ID of the owner of the carrier gateway.
    - tag :<key> - The key/value combination of a tag assigned to the resource. Use the tag key in the filter name and the tag value as the filter value. For example, to find all resources that have a tag with the key Owner and the value TeamA , specify tag:Owner for the filter name and TeamA for the filter value.
    - tag-key - The key of a tag assigned to the resource. Use this filter to find all resources assigned a tag with a specific key, regardless of the tag value.
    - vpc-id - The ID of the VPC associated with the carrier gateway.

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'CarrierGateways': [
        {
            'CarrierGatewayId': 'string',
            'VpcId': 'string',
            'State': 'pending'|'available'|'deleting'|'deleted',
            'OwnerId': 'string',
            'Tags': [
                {
                    'Key': 'string',
                    'Value': 'string'
                },
            ]
        },
    ],

}

**Response Structure**

- _(dict) --_

    - **CarrierGateways** _(list) --_

        Information about the carrier gateway.

        - _(dict) --_

            Describes a carrier gateway.

            - **CarrierGatewayId** _(string) --_

                The ID of the carrier gateway.

            - **VpcId** _(string) --_

                The ID of the VPC associated with the carrier gateway.

            - **State** _(string) --_

                The state of the carrier gateway.

            - **OwnerId** _(string) --_

                The AWS account ID of the owner of the carrier gateway.

            - **Tags** _(list) --_

                The tags assigned to the carrier gateway.

                - _(dict) --_

                    Describes a tag.

                    - **Key** _(string) --_

                        The key of the tag.

                        Constraints: Tag keys are case-sensitive and accept a maximum of 127 Unicode characters. May not begin with aws: .

                    - **Value** _(string) --_

                        The value of the tag.

                        Constraints: Tag values are case-sensitive and accept a maximum of 255 Unicode characters.


_class_ EC2.Paginator.DescribeClassicLinkInstances

paginator = client.get_paginator('describe_classic_link_instances')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_classic_link_instances()](#EC2.Client.describe_classic_link_instances "EC2.Client.describe_classic_link_instances").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeClassicLinkInstances)

**Request Syntax**

response_iterator = paginator.paginate(
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    DryRun=True|False,
    InstanceIds=[
        'string',
    ],
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **Filters** (_list_) --

    One or more filters.

    - group-id - The ID of a VPC security group that's associated with the instance.
    - instance-id - The ID of the instance.
    - tag :<key> - The key/value combination of a tag assigned to the resource. Use the tag key in the filter name and the tag value as the filter value. For example, to find all resources that have a tag with the key Owner and the value TeamA , specify tag:Owner for the filter name and TeamA for the filter value.
    - tag-key - The key of a tag assigned to the resource. Use this filter to find all resources assigned a tag with a specific key, regardless of the tag value.
    - vpc-id - The ID of the VPC to which the instance is linked. vpc-id - The ID of the VPC that the instance is linked to.

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **InstanceIds** (_list_) --

    One or more instance IDs. Must be instances linked to a VPC through ClassicLink.

    - _(string) --_
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'Instances': [
        {
            'Groups': [
                {
                    'GroupName': 'string',
                    'GroupId': 'string'
                },
            ],
            'InstanceId': 'string',
            'Tags': [
                {
                    'Key': 'string',
                    'Value': 'string'
                },
            ],
            'VpcId': 'string'
        },
    ],

}

**Response Structure**

- _(dict) --_

    - **Instances** _(list) --_

        Information about one or more linked EC2-Classic instances.

        - _(dict) --_

            Describes a linked EC2-Classic instance.

            - **Groups** _(list) --_

                A list of security groups.

                - _(dict) --_

                    Describes a security group.

                    - **GroupName** _(string) --_

                        The name of the security group.

                    - **GroupId** _(string) --_

                        The ID of the security group.

            - **InstanceId** _(string) --_

                The ID of the instance.

            - **Tags** _(list) --_

                Any tags assigned to the instance.

                - _(dict) --_

                    Describes a tag.

                    - **Key** _(string) --_

                        The key of the tag.

                        Constraints: Tag keys are case-sensitive and accept a maximum of 127 Unicode characters. May not begin with aws: .

                    - **Value** _(string) --_

                        The value of the tag.

                        Constraints: Tag values are case-sensitive and accept a maximum of 255 Unicode characters.

            - **VpcId** _(string) --_

                The ID of the VPC.


_class_ EC2.Paginator.DescribeClientVpnAuthorizationRules

paginator = client.get_paginator('describe_client_vpn_authorization_rules')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_client_vpn_authorization_rules()](#EC2.Client.describe_client_vpn_authorization_rules "EC2.Client.describe_client_vpn_authorization_rules").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeClientVpnAuthorizationRules)

**Request Syntax**

response_iterator = paginator.paginate(
    ClientVpnEndpointId='string',
    DryRun=True|False,
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **ClientVpnEndpointId** (_string_) --

    **[REQUIRED]**

    The ID of the Client VPN endpoint.

- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **Filters** (_list_) --

    One or more filters. Filter names and values are case-sensitive.

    - description - The description of the authorization rule.
    - destination-cidr - The CIDR of the network to which the authorization rule applies.
    - group-id - The ID of the Active Directory group to which the authorization rule grants access.

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'AuthorizationRules': [
        {
            'ClientVpnEndpointId': 'string',
            'Description': 'string',
            'GroupId': 'string',
            'AccessAll': True|False,
            'DestinationCidr': 'string',
            'Status': {
                'Code': 'authorizing'|'active'|'failed'|'revoking',
                'Message': 'string'
            }
        },
    ],

}

**Response Structure**

- _(dict) --_

    - **AuthorizationRules** _(list) --_

        Information about the authorization rules.

        - _(dict) --_

            Information about an authorization rule.

            - **ClientVpnEndpointId** _(string) --_

                The ID of the Client VPN endpoint with which the authorization rule is associated.

            - **Description** _(string) --_

                A brief description of the authorization rule.

            - **GroupId** _(string) --_

                The ID of the Active Directory group to which the authorization rule grants access.

            - **AccessAll** _(boolean) --_

                Indicates whether the authorization rule grants access to all clients.

            - **DestinationCidr** _(string) --_

                The IPv4 address range, in CIDR notation, of the network to which the authorization rule applies.

            - **Status** _(dict) --_

                The current state of the authorization rule.

                - **Code** _(string) --_

                    The state of the authorization rule.

                - **Message** _(string) --_

                    A message about the status of the authorization rule, if applicable.


_class_ EC2.Paginator.DescribeClientVpnConnections

paginator = client.get_paginator('describe_client_vpn_connections')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_client_vpn_connections()](#EC2.Client.describe_client_vpn_connections "EC2.Client.describe_client_vpn_connections").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeClientVpnConnections)

**Request Syntax**

response_iterator = paginator.paginate(
    ClientVpnEndpointId='string',
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    DryRun=True|False,
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **ClientVpnEndpointId** (_string_) --

    **[REQUIRED]**

    The ID of the Client VPN endpoint.

- **Filters** (_list_) --

    One or more filters. Filter names and values are case-sensitive.

    - connection-id - The ID of the connection.
    - username - For Active Directory client authentication, the user name of the client who established the client connection.

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'Connections': [
        {
            'ClientVpnEndpointId': 'string',
            'Timestamp': 'string',
            'ConnectionId': 'string',
            'Username': 'string',
            'ConnectionEstablishedTime': 'string',
            'IngressBytes': 'string',
            'EgressBytes': 'string',
            'IngressPackets': 'string',
            'EgressPackets': 'string',
            'ClientIp': 'string',
            'CommonName': 'string',
            'Status': {
                'Code': 'active'|'failed-to-terminate'|'terminating'|'terminated',
                'Message': 'string'
            },
            'ConnectionEndTime': 'string',
            'PostureComplianceStatuses': [
                'string',
            ]
        },
    ],

}

**Response Structure**

- _(dict) --_

    - **Connections** _(list) --_

        Information about the active and terminated client connections.

        - _(dict) --_

            Describes a client connection.

            - **ClientVpnEndpointId** _(string) --_

                The ID of the Client VPN endpoint to which the client is connected.

            - **Timestamp** _(string) --_

                The current date and time.

            - **ConnectionId** _(string) --_

                The ID of the client connection.

            - **Username** _(string) --_

                The username of the client who established the client connection. This information is only provided if Active Directory client authentication is used.

            - **ConnectionEstablishedTime** _(string) --_

                The date and time the client connection was established.

            - **IngressBytes** _(string) --_

                The number of bytes sent by the client.

            - **EgressBytes** _(string) --_

                The number of bytes received by the client.

            - **IngressPackets** _(string) --_

                The number of packets sent by the client.

            - **EgressPackets** _(string) --_

                The number of packets received by the client.

            - **ClientIp** _(string) --_

                The IP address of the client.

            - **CommonName** _(string) --_

                The common name associated with the client. This is either the name of the client certificate, or the Active Directory user name.

            - **Status** _(dict) --_

                The current state of the client connection.

                - **Code** _(string) --_

                    The state of the client connection.

                - **Message** _(string) --_

                    A message about the status of the client connection, if applicable.

            - **ConnectionEndTime** _(string) --_

                The date and time the client connection was terminated.

            - **PostureComplianceStatuses** _(list) --_

                The statuses returned by the client connect handler for posture compliance, if applicable.

                - _(string) --_

_class_ EC2.Paginator.DescribeClientVpnEndpoints

paginator = client.get_paginator('describe_client_vpn_endpoints')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_client_vpn_endpoints()](#EC2.Client.describe_client_vpn_endpoints "EC2.Client.describe_client_vpn_endpoints").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeClientVpnEndpoints)

**Request Syntax**

response_iterator = paginator.paginate(
    ClientVpnEndpointIds=[
        'string',
    ],
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    DryRun=True|False,
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **ClientVpnEndpointIds** (_list_) --

    The ID of the Client VPN endpoint.

    - _(string) --_
- **Filters** (_list_) --

    One or more filters. Filter names and values are case-sensitive.

    - endpoint-id - The ID of the Client VPN endpoint.
    - transport-protocol - The transport protocol (tcp | udp ).

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'ClientVpnEndpoints': [
        {
            'ClientVpnEndpointId': 'string',
            'Description': 'string',
            'Status': {
                'Code': 'pending-associate'|'available'|'deleting'|'deleted',
                'Message': 'string'
            },
            'CreationTime': 'string',
            'DeletionTime': 'string',
            'DnsName': 'string',
            'ClientCidrBlock': 'string',
            'DnsServers': [
                'string',
            ],
            'SplitTunnel': True|False,
            'VpnProtocol': 'openvpn',
            'TransportProtocol': 'tcp'|'udp',
            'VpnPort': 123,
            'AssociatedTargetNetworks': [
                {
                    'NetworkId': 'string',
                    'NetworkType': 'vpc'
                },
            ],
            'ServerCertificateArn': 'string',
            'AuthenticationOptions': [
                {
                    'Type': 'certificate-authentication'|'directory-service-authentication'|'federated-authentication',
                    'ActiveDirectory': {
                        'DirectoryId': 'string'
                    },
                    'MutualAuthentication': {
                        'ClientRootCertificateChain': 'string'
                    },
                    'FederatedAuthentication': {
                        'SamlProviderArn': 'string',
                        'SelfServiceSamlProviderArn': 'string'
                    }
                },
            ],
            'ConnectionLogOptions': {
                'Enabled': True|False,
                'CloudwatchLogGroup': 'string',
                'CloudwatchLogStream': 'string'
            },
            'Tags': [
                {
                    'Key': 'string',
                    'Value': 'string'
                },
            ],
            'SecurityGroupIds': [
                'string',
            ],
            'VpcId': 'string',
            'SelfServicePortalUrl': 'string',
            'ClientConnectOptions': {
                'Enabled': True|False,
                'LambdaFunctionArn': 'string&##39;,
                'Status': {
                    'Code': 'applying'|'applied',
                    'Message': 'string'
                }
            }
        },
    ],

}

**Response Structure**

- _(dict) --_

    - **ClientVpnEndpoints** _(list) --_

        Information about the Client VPN endpoints.

        - _(dict) --_

            Describes a Client VPN endpoint.

            - **ClientVpnEndpointId** _(string) --_

                The ID of the Client VPN endpoint.

            - **Description** _(string) --_

                A brief description of the endpoint.

            - **Status** _(dict) --_

                The current state of the Client VPN endpoint.

                - **Code** _(string) --_

                    The state of the Client VPN endpoint. Possible states include:

                    - pending-associate - The Client VPN endpoint has been created but no target networks have been associated. The Client VPN endpoint cannot accept connections.
                    - available - The Client VPN endpoint has been created and a target network has been associated. The Client VPN endpoint can accept connections.
                    - deleting - The Client VPN endpoint is being deleted. The Client VPN endpoint cannot accept connections.
                    - deleted - The Client VPN endpoint has been deleted. The Client VPN endpoint cannot accept connections.
                - **Message** _(string) --_

                    A message about the status of the Client VPN endpoint.

            - **CreationTime** _(string) --_

                The date and time the Client VPN endpoint was created.

            - **DeletionTime** _(string) --_

                The date and time the Client VPN endpoint was deleted, if applicable.

            - **DnsName** _(string) --_

                The DNS name to be used by clients when connecting to the Client VPN endpoint.

            - **ClientCidrBlock** _(string) --_

                The IPv4 address range, in CIDR notation, from which client IP addresses are assigned.

            - **DnsServers** _(list) --_

                Information about the DNS servers to be used for DNS resolution.

                - _(string) --_
            - **SplitTunnel** _(boolean) --_

                Indicates whether split-tunnel is enabled in the AWS Client VPN endpoint.

                For information about split-tunnel VPN endpoints, see [Split-Tunnel AWS Client VPN Endpoint](https://docs.aws.amazon.com/vpn/latest/clientvpn-admin/split-tunnel-vpn.html) in the _AWS Client VPN Administrator Guide_ .

            - **VpnProtocol** _(string) --_

                The protocol used by the VPN session.

            - **TransportProtocol** _(string) --_

                The transport protocol used by the Client VPN endpoint.

            - **VpnPort** _(integer) --_

                The port number for the Client VPN endpoint.

            - **AssociatedTargetNetworks** _(list) --_

                Information about the associated target networks. A target network is a subnet in a VPC.

                - _(dict) --_

                    Describes a target network that is associated with a Client VPN endpoint. A target network is a subnet in a VPC.

                    - **NetworkId** _(string) --_

                        The ID of the subnet.

                    - **NetworkType** _(string) --_

                        The target network type.

            - **ServerCertificateArn** _(string) --_

                The ARN of the server certificate.

            - **AuthenticationOptions** _(list) --_

                Information about the authentication method used by the Client VPN endpoint.

                - _(dict) --_

                    Describes the authentication methods used by a Client VPN endpoint. For more information, see [Authentication](https://docs.aws.amazon.com/vpn/latest/clientvpn-admin/client-authentication.html) in the _AWS Client VPN Administrator Guide_ .

                    - **Type** _(string) --_

                        The authentication type used.

                    - **ActiveDirectory** _(dict) --_

                        Information about the Active Directory, if applicable.

                        - **DirectoryId** _(string) --_

                            The ID of the Active Directory used for authentication.

                    - **MutualAuthentication** _(dict) --_

                        Information about the authentication certificates, if applicable.

                        - **ClientRootCertificateChain** _(string) --_

                            The ARN of the client certificate.

                    - **FederatedAuthentication** _(dict) --_

                        Information about the IAM SAML identity provider, if applicable.

                        - **SamlProviderArn** _(string) --_

                            The Amazon Resource Name (ARN) of the IAM SAML identity provider.

                        - **SelfServiceSamlProviderArn** _(string) --_

                            The Amazon Resource Name (ARN) of the IAM SAML identity provider for the self-service portal.

            - **ConnectionLogOptions** _(dict) --_

                Information about the client connection logging options for the Client VPN endpoint.

                - **Enabled** _(boolean) --_

                    Indicates whether client connection logging is enabled for the Client VPN endpoint.

                - **CloudwatchLogGroup** _(string) --_

                    The name of the Amazon CloudWatch Logs log group to which connection logging data is published.

                - **CloudwatchLogStream** _(string) --_

                    The name of the Amazon CloudWatch Logs log stream to which connection logging data is published.

            - **Tags** _(list) --_

                Any tags assigned to the Client VPN endpoint.

                - _(dict) --_

                    Describes a tag.

                    - **Key** _(string) --_

                        The key of the tag.

                        Constraints: Tag keys are case-sensitive and accept a maximum of 127 Unicode characters. May not begin with aws: .

                    - **Value** _(string) --_

                        The value of the tag.

                        Constraints: Tag values are case-sensitive and accept a maximum of 255 Unicode characters.

            - **SecurityGroupIds** _(list) --_

                The IDs of the security groups for the target network.

                - _(string) --_
            - **VpcId** _(string) --_

                The ID of the VPC.

            - **SelfServicePortalUrl** _(string) --_

                The URL of the self-service portal.

            - **ClientConnectOptions** _(dict) --_

                The options for managing connection authorization for new client connections.

                - **Enabled** _(boolean) --_

                    Indicates whether client connect options are enabled.

                - **LambdaFunctionArn** _(string) --_

                    The Amazon Resource Name (ARN) of the AWS Lambda function used for connection authorization.

                - **Status** _(dict) --_

                    The status of any updates to the client connect options.

                    - **Code** _(string) --_

                        The status code.

                    - **Message** _(string) --_

                        The status message.


_class_ EC2.Paginator.DescribeClientVpnRoutes

paginator = client.get_paginator('describe_client_vpn_routes')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_client_vpn_routes()](#EC2.Client.describe_client_vpn_routes "EC2.Client.describe_client_vpn_routes").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeClientVpnRoutes)

**Request Syntax**

response_iterator = paginator.paginate(
    ClientVpnEndpointId='string',
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    DryRun=True|False,
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **ClientVpnEndpointId** (_string_) --

    **[REQUIRED]**

    The ID of the Client VPN endpoint.

- **Filters** (_list_) --

    One or more filters. Filter names and values are case-sensitive.

    - destination-cidr - The CIDR of the route destination.
    - origin - How the route was associated with the Client VPN endpoint (associate | add-route ).
    - target-subnet - The ID of the subnet through which traffic is routed.

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'Routes': [
        {
            'ClientVpnEndpointId': 'string',
            'DestinationCidr': 'string',
            'TargetSubnet': 'string',
            'Type': 'string',
            'Origin': 'string',
            'Status': {
                'Code': 'creating'|'active'|'failed'|'deleting',
                'Message': 'string'
            },
            'Description': 'string'
        },
    ],

}

**Response Structure**

- _(dict) --_

    - **Routes** _(list) --_

        Information about the Client VPN endpoint routes.

        - _(dict) --_

            Information about a Client VPN endpoint route.

            - **ClientVpnEndpointId** _(string) --_

                The ID of the Client VPN endpoint with which the route is associated.

            - **DestinationCidr** _(string) --_

                The IPv4 address range, in CIDR notation, of the route destination.

            - **TargetSubnet** _(string) --_

                The ID of the subnet through which traffic is routed.

            - **Type** _(string) --_

                The route type.

            - **Origin** _(string) --_

                Indicates how the route was associated with the Client VPN endpoint. associate indicates that the route was automatically added when the target network was associated with the Client VPN endpoint. add-route indicates that the route was manually added using the **CreateClientVpnRoute** action.

            - **Status** _(dict) --_

                The current state of the route.

                - **Code** _(string) --_

                    The state of the Client VPN endpoint route.

                - **Message** _(string) --_

                    A message about the status of the Client VPN endpoint route, if applicable.

            - **Description** _(string) --_

                A brief description of the route.


_class_ EC2.Paginator.DescribeClientVpnTargetNetworks

paginator = client.get_paginator('describe_client_vpn_target_networks')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_client_vpn_target_networks()](#EC2.Client.describe_client_vpn_target_networks "EC2.Client.describe_client_vpn_target_networks").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeClientVpnTargetNetworks)

**Request Syntax**

response_iterator = paginator.paginate(
    ClientVpnEndpointId='string',
    AssociationIds=[
        'string',
    ],
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    DryRun=True|False,
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **ClientVpnEndpointId** (_string_) --

    **[REQUIRED]**

    The ID of the Client VPN endpoint.

- **AssociationIds** (_list_) --

    The IDs of the target network associations.

    - _(string) --_
- **Filters** (_list_) --

    One or more filters. Filter names and values are case-sensitive.

    - association-id - The ID of the association.
    - target-network-id - The ID of the subnet specified as the target network.
    - vpc-id - The ID of the VPC in which the target network is located.

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'ClientVpnTargetNetworks': [
        {
            'AssociationId': 'string',
            'VpcId': 'string',
            'TargetNetworkId': 'string',
            'ClientVpnEndpointId': 'string',
            'Status': {
                'Code': 'associating'|'associated'|'association-failed'|'disassociating'|'disassociated',
                'Message': 'string'
            },
            'SecurityGroups': [
                'string',
            ]
        },
    ],

}

**Response Structure**

- _(dict) --_

    - **ClientVpnTargetNetworks** _(list) --_

        Information about the associated target networks.

        - _(dict) --_

            Describes a target network associated with a Client VPN endpoint.

            - **AssociationId** _(string) --_

                The ID of the association.

            - **VpcId** _(string) --_

                The ID of the VPC in which the target network (subnet) is located.

            - **TargetNetworkId** _(string) --_

                The ID of the subnet specified as the target network.

            - **ClientVpnEndpointId** _(string) --_

                The ID of the Client VPN endpoint with which the target network is associated.

            - **Status** _(dict) --_

                The current state of the target network association.

                - **Code** _(string) --_

                    The state of the target network association.

                - **Message** _(string) --_

                    A message about the status of the target network association, if applicable.

            - **SecurityGroups** _(list) --_

                The IDs of the security groups applied to the target network association.

                - _(string) --_

_class_ EC2.Paginator.DescribeCoipPools

paginator = client.get_paginator('describe_coip_pools')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_coip_pools()](#EC2.Client.describe_coip_pools "EC2.Client.describe_coip_pools").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeCoipPools)

**Request Syntax**

response_iterator = paginator.paginate(
    PoolIds=[
        'string',
    ],
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    DryRun=True|False,
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **PoolIds** (_list_) --

    The IDs of the address pools.

    - _(string) --_
- **Filters** (_list_) --

    The filters. The following are the possible values:

    - coip-pool.pool-id
    - coip-pool.local-gateway-route-table-id

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'CoipPools': [
        {
            'PoolId': 'string',
            'PoolCidrs': [
                'string',
            ],
            'LocalGatewayRouteTableId': 'string',
            'Tags': [
                {
                    'Key': 'string',
                    'Value': 'string'
                },
            ],
            'PoolArn': 'string'
        },
    ],

}

**Response Structure**

- _(dict) --_

    - **CoipPools** _(list) --_

        Information about the address pools.

        - _(dict) --_

            Describes a customer-owned address pool.

            - **PoolId** _(string) --_

                The ID of the address pool.

            - **PoolCidrs** _(list) --_

                The address ranges of the address pool.

                - _(string) --_
            - **LocalGatewayRouteTableId** _(string) --_

                The ID of the local gateway route table.

            - **Tags** _(list) --_

                The tags.

                - _(dict) --_

                    Describes a tag.

                    - **Key** _(string) --_

                        The key of the tag.

                        Constraints: Tag keys are case-sensitive and accept a maximum of 127 Unicode characters. May not begin with aws: .

                    - **Value** _(string) --_

                        The value of the tag.

                        Constraints: Tag values are case-sensitive and accept a maximum of 255 Unicode characters.

            - **PoolArn** _(string) --_

                The ARN of the address pool.


_class_ EC2.Paginator.DescribeDhcpOptions

paginator = client.get_paginator('describe_dhcp_options')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_dhcp_options()](#EC2.Client.describe_dhcp_options "EC2.Client.describe_dhcp_options").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeDhcpOptions)

**Request Syntax**

response_iterator = paginator.paginate(
    DhcpOptionsIds=[
        'string',
    ],
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    DryRun=True|False,
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **DhcpOptionsIds** (_list_) --

    The IDs of one or more DHCP options sets.

    Default: Describes all your DHCP options sets.

    - _(string) --_
- **Filters** (_list_) --

    One or more filters.

    - dhcp-options-id - The ID of a DHCP options set.
    - key - The key for one of the options (for example, domain-name ).
    - value - The value for one of the options.
    - owner-id - The ID of the AWS account that owns the DHCP options set.
    - tag :<key> - The key/value combination of a tag assigned to the resource. Use the tag key in the filter name and the tag value as the filter value. For example, to find all resources that have a tag with the key Owner and the value TeamA , specify tag:Owner for the filter name and TeamA for the filter value.
    - tag-key - The key of a tag assigned to the resource. Use this filter to find all resources assigned a tag with a specific key, regardless of the tag value.

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'DhcpOptions': [
        {
            'DhcpConfigurations': [
                {
                    'Key': 'string',
                    'Values': [
                        {
                            'Value': 'string'
                        },
                    ]
                },
            ],
            'DhcpOptionsId': 'string',
            'OwnerId': 'string',
            'Tags': [
                {
                    'Key': 'string',
                    'Value': 'string'
                },
            ]
        },
    ],

}

**Response Structure**

- _(dict) --_

    - **DhcpOptions** _(list) --_

        Information about one or more DHCP options sets.

        - _(dict) --_

            Describes a set of DHCP options.

            - **DhcpConfigurations** _(list) --_

                One or more DHCP options in the set.

                - _(dict) --_

                    Describes a DHCP configuration option.

                    - **Key** _(string) --_

                        The name of a DHCP option.

                    - **Values** _(list) --_

                        One or more values for the DHCP option.

                        - _(dict) --_

                            Describes a value for a resource attribute that is a String.

                            - **Value** _(string) --_

                                The attribute value. The value is case-sensitive.

            - **DhcpOptionsId** _(string) --_

                The ID of the set of DHCP options.

            - **OwnerId** _(string) --_

                The ID of the AWS account that owns the DHCP options set.

            - **Tags** _(list) --_

                Any tags assigned to the DHCP options set.

                - _(dict) --_

                    Describes a tag.

                    - **Key** _(string) --_

                        The key of the tag.

                        Constraints: Tag keys are case-sensitive and accept a maximum of 127 Unicode characters. May not begin with aws: .

                    - **Value** _(string) --_

                        The value of the tag.

                        Constraints: Tag values are case-sensitive and accept a maximum of 255 Unicode characters.


_class_ EC2.Paginator.DescribeEgressOnlyInternetGateways

paginator = client.get_paginator('describe_egress_only_internet_gateways')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_egress_only_internet_gateways()](#EC2.Client.describe_egress_only_internet_gateways "EC2.Client.describe_egress_only_internet_gateways").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeEgressOnlyInternetGateways)

**Request Syntax**

response_iterator = paginator.paginate(
    DryRun=True|False,
    EgressOnlyInternetGatewayIds=[
        'string',
    ],
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **EgressOnlyInternetGatewayIds** (_list_) --

    One or more egress-only internet gateway IDs.

    - _(string) --_
- **Filters** (_list_) --

    One or more filters.

    - tag :<key> - The key/value combination of a tag assigned to the resource. Use the tag key in the filter name and the tag value as the filter value. For example, to find all resources that have a tag with the key Owner and the value TeamA , specify tag:Owner for the filter name and TeamA for the filter value.
    - tag-key - The key of a tag assigned to the resource. Use this filter to find all resources assigned a tag with a specific key, regardless of the tag value.

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'EgressOnlyInternetGateways': [
        {
            'Attachments': [
                {
                    'State': 'attaching'|'attached'|'detaching'|'detached',
                    'VpcId': 'string'
                },
            ],
            'EgressOnlyInternetGatewayId': 'string',
            'Tags': [
                {
                    'Key': 'string',
                    'Value': 'string'
                },
            ]
        },
    ],

}

**Response Structure**

- _(dict) --_

    - **EgressOnlyInternetGateways** _(list) --_

        Information about the egress-only internet gateways.

        - _(dict) --_

            Describes an egress-only internet gateway.

            - **Attachments** _(list) --_

                Information about the attachment of the egress-only internet gateway.

                - _(dict) --_

                    Describes the attachment of a VPC to an internet gateway or an egress-only internet gateway.

                    - **State** _(string) --_

                        The current state of the attachment. For an internet gateway, the state is available when attached to a VPC; otherwise, this value is not returned.

                    - **VpcId** _(string) --_

                        The ID of the VPC.

            - **EgressOnlyInternetGatewayId** _(string) --_

                The ID of the egress-only internet gateway.

            - **Tags** _(list) --_

                The tags assigned to the egress-only internet gateway.

                - _(dict) --_

                    Describes a tag.

                    - **Key** _(string) --_

                        The key of the tag.

                        Constraints: Tag keys are case-sensitive and accept a maximum of 127 Unicode characters. May not begin with aws: .

                    - **Value** _(string) --_

                        The value of the tag.

                        Constraints: Tag values are case-sensitive and accept a maximum of 255 Unicode characters.


_class_ EC2.Paginator.DescribeExportImageTasks

paginator = client.get_paginator('describe_export_image_tasks')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_export_image_tasks()](#EC2.Client.describe_export_image_tasks "EC2.Client.describe_export_image_tasks").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeExportImageTasks)

**Request Syntax**

response_iterator = paginator.paginate(
    DryRun=True|False,
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    ExportImageTaskIds=[
        'string',
    ],
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **Filters** (_list_) --

    Filter tasks using the task-state filter and one of the following values: active , completed , deleting , or deleted .

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **ExportImageTaskIds** (_list_) --

    The IDs of the export image tasks.

    - _(string) --_
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'ExportImageTasks': [
        {
            'Description': 'string',
            'ExportImageTaskId': 'string',
            'ImageId': 'string',
            'Progress': 'string',
            'S3ExportLocation': {
                'S3Bucket': 'string',
                'S3Prefix': 'string'
            },
            'Status': 'string',
            'StatusMessage': 'string',
            'Tags': [
                {
                    'Key': 'string',
                    'Value': 'string'
                },
            ]
        },
    ],

}

**Response Structure**

- _(dict) --_

    - **ExportImageTasks** _(list) --_

        Information about the export image tasks.

        - _(dict) --_

            Describes an export image task.

            - **Description** _(string) --_

                A description of the image being exported.

            - **ExportImageTaskId** _(string) --_

                The ID of the export image task.

            - **ImageId** _(string) --_

                The ID of the image.

            - **Progress** _(string) --_

                The percent complete of the export image task.

            - **S3ExportLocation** _(dict) --_

                Information about the destination Amazon S3 bucket.

                - **S3Bucket** _(string) --_

                    The destination Amazon S3 bucket.

                - **S3Prefix** _(string) --_

                    The prefix (logical hierarchy) in the bucket.

            - **Status** _(string) --_

                The status of the export image task. The possible values are active , completed , deleting , and deleted .

            - **StatusMessage** _(string) --_

                The status message for the export image task.

            - **Tags** _(list) --_

                Any tags assigned to the image being exported.

                - _(dict) --_

                    Describes a tag.

                    - **Key** _(string) --_

                        The key of the tag.

                        Constraints: Tag keys are case-sensitive and accept a maximum of 127 Unicode characters. May not begin with aws: .

                    - **Value** _(string) --_

                        The value of the tag.

                        Constraints: Tag values are case-sensitive and accept a maximum of 255 Unicode characters.


_class_ EC2.Paginator.DescribeFastSnapshotRestores

paginator = client.get_paginator('describe_fast_snapshot_restores')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_fast_snapshot_restores()](#EC2.Client.describe_fast_snapshot_restores "EC2.Client.describe_fast_snapshot_restores").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeFastSnapshotRestores)

**Request Syntax**

response_iterator = paginator.paginate(
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    DryRun=True|False,
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **Filters** (_list_) --

    The filters. The possible values are:

    - availability-zone : The Availability Zone of the snapshot.
    - owner-id : The ID of the AWS account that enabled fast snapshot restore on the snapshot.
    - snapshot-id : The ID of the snapshot.
    - state : The state of fast snapshot restores for the snapshot (enabling | optimizing | enabled | disabling | disabled ).

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'FastSnapshotRestores': [
        {
            'SnapshotId': 'string',
            'AvailabilityZone': 'string',
            'State': 'enabling'|'optimizing'|'enabled'|'disabling'|'disabled',
            'StateTransitionReason': 'string',
            'OwnerId': 'string',
            'OwnerAlias': 'string',
            'EnablingTime': datetime(2015, 1, 1),
            'OptimizingTime': datetime(2015, 1, 1),
            'EnabledTime': datetime(2015, 1, 1),
            'DisablingTime': datetime(2015, 1, 1),
            'DisabledTime': datetime(2015, 1, 1)
        },
    ],

}

**Response Structure**

- _(dict) --_

    - **FastSnapshotRestores** _(list) --_

        Information about the state of fast snapshot restores.

        - _(dict) --_

            Describes fast snapshot restores for a snapshot.

            - **SnapshotId** _(string) --_

                The ID of the snapshot.

            - **AvailabilityZone** _(string) --_

                The Availability Zone.

            - **State** _(string) --_

                The state of fast snapshot restores.

            - **StateTransitionReason** _(string) --_

                The reason for the state transition. The possible values are as follows:

                - Client.UserInitiated - The state successfully transitioned to enabling or disabling .
                - Client.UserInitiated - Lifecycle state transition - The state successfully transitioned to optimizing , enabled , or disabled .
            - **OwnerId** _(string) --_

                The ID of the AWS account that enabled fast snapshot restores on the snapshot.

            - **OwnerAlias** _(string) --_

                The AWS owner alias that enabled fast snapshot restores on the snapshot. This is intended for future use.

            - **EnablingTime** _(datetime) --_

                The time at which fast snapshot restores entered the enabling state.

            - **OptimizingTime** _(datetime) --_

                The time at which fast snapshot restores entered the optimizing state.

            - **EnabledTime** _(datetime) --_

                The time at which fast snapshot restores entered the enabled state.

            - **DisablingTime** _(datetime) --_

                The time at which fast snapshot restores entered the disabling state.

            - **DisabledTime** _(datetime) --_

                The time at which fast snapshot restores entered the disabled state.


_class_ EC2.Paginator.DescribeFleets

paginator = client.get_paginator('describe_fleets')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_fleets()](#EC2.Client.describe_fleets "EC2.Client.describe_fleets").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeFleets)

**Request Syntax**

response_iterator = paginator.paginate(
    DryRun=True|False,
    FleetIds=[
        'string',
    ],
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **FleetIds** (_list_) --

    The ID of the EC2 Fleets.

    - _(string) --_
- **Filters** (_list_) --

    The filters.

    - activity-status - The progress of the EC2 Fleet ( error | pending-fulfillment | pending-termination | fulfilled ).
    - excess-capacity-termination-policy - Indicates whether to terminate running instances if the target capacity is decreased below the current EC2 Fleet size (true | false ).
    - fleet-state - The state of the EC2 Fleet (submitted | active | deleted | failed | deleted-running | deleted-terminating | modifying ).
    - replace-unhealthy-instances - Indicates whether EC2 Fleet should replace unhealthy instances (true | false ).
    - type - The type of request (instant | request | maintain ).

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'Fleets': [
        {
            'ActivityStatus': 'error'|'pending_fulfillment'|'pending_termination'|'fulfilled',
            'CreateTime': datetime(2015, 1, 1),
            'FleetId': 'string',
            'FleetState': 'submitted'|'active'|'deleted'|'failed'|'deleted_running'|'deleted_terminating'|'modifying',
            'ClientToken': 'string',
            'ExcessCapacityTerminationPolicy': 'no-termination'|'termination',
            'FulfilledCapacity': 123.0,
            'FulfilledOnDemandCapacity': 123.0,
            'LaunchTemplateConfigs': [
                {
                    'LaunchTemplateSpecification': {
                        'LaunchTemplateId': 'string',
                        'LaunchTemplateName': 'string',
                        'Version': 'string'
                    },
                    'Overrides': [
                        {
                            'InstanceType': 't1.micro'|'t2.nano'|'t2.micro'|'t2.small'|'t2.medium'|'t2.large'|'t2.xlarge'|'t2.2xlarge'|'t3.nano'|'t3.micro'|'t3.small'|'t3.medium'|'t3.large'|'t3.xlarge'|'t3.2xlarge'|'t3a.nano'|'t3a.micro'|'t3a.small'|'t3a.medium'|'t3a.large'|'t3a.xlarge'|'t3a.2xlarge'|'t4g.nano'|'t4g.micro'|'t4g.small'|'t4g.medium'|'t4g.large'|'t4g.xlarge'|'t4g.2xlarge'|'m1.small'|'m1.medium'|'m1.large'|'m1.xlarge'|'m3.medium'|'m3.large'|'m3.xlarge'|'m3.2xlarge'|'m4.large'|'m4.xlarge'|'m4.2xlarge'|'m4.4xlarge'|'m4.10xlarge'|'m4.16xlarge'|'m2.xlarge'|'m2.2xlarge'|'m2.4xlarge'|'cr1.8xlarge'|'r3.large'|'r3.xlarge'|'r3.2xlarge'|'r3.4xlarge'|'r3.8xlarge'|'r4.large'|'r4.xlarge'|'r4.2xlarge'|'r4.4xlarge'|'r4.8xlarge'|'r4.16xlarge'|'r5.large'|'r5.xlarge'|'r5.2xlarge'|'r5.4xlarge'|'r5.8xlarge'|'r5.12xlarge'|'r5.16xlarge'|'r5.24xlarge'|'r5.metal'|'r5a.large'|'r5a.xlarge'|'r5a.2xlarge'|'r5a.4xlarge'|'r5a.8xlarge'|'r5a.12xlarge'|'r5a.16xlarge'|'r5a.24xlarge'|'r5b.large'|'r5b.xlarge'|'r5b.2xlarge'|'r5b.4xlarge'|'r5b.8xlarge'|'r5b.12xlarge'|'r5b.16xlarge'|'r5b.24xlarge'|'r5b.metal'|'r5d.large'|'r5d.xlarge'|'r5d.2xlarge'|'r5d.4xlarge'|'r5d.8xlarge'|'r5d.12xlarge'|'r5d.16xlarge'|'r5d.24xlarge'|'r5d.metal'|'r5ad.large'|'r5ad.xlarge'|'r5ad.2xlarge'|'r5ad.4xlarge'|'r5ad.8xlarge'|'r5ad.12xlarge'|'r5ad.16xlarge'|'r5ad.24xlarge'|'r6g.metal'|'r6g.medium'|'r6g.large'|'r6g.xlarge'|'r6g.2xlarge'|'r6g.4xlarge'|'r6g.8xlarge'|'r6g.12xlarge'|'r6g.16xlarge'|'r6gd.metal'|'r6gd.medium'|'r6gd.large'|'r6gd.xlarge'|'r6gd.2xlarge'|'r6gd.4xlarge'|'r6gd.8xlarge'|'r6gd.12xlarge'|'r6gd.16xlarge'|'x1.16xlarge'|'x1.32xlarge'|'x1e.xlarge'|'x1e.2xlarge'|'x1e.4xlarge'|'x1e.8xlarge'|'x1e.16xlarge'|'x1e.32xlarge'|'i2.xlarge'|'i2.2xlarge'|'i2.4xlarge'|'i2.8xlarge'|'i3.large'|'i3.xlarge'|'i3.2xlarge'|'i3.4xlarge'|'i3.8xlarge'|'i3.16xlarge'|'i3.metal'|'i3en.large'|'i3en.xlarge'|'i3en.2xlarge'|'i3en.3xlarge'|'i3en.6xlarge'|'i3en.12xlarge'|'i3en.24xlarge'|'i3en.metal'|'hi1.4xlarge'|'hs1.8xlarge'|'c1.medium'|'c1.xlarge'|'c3.large'|'c3.xlarge'|'c3.2xlarge'|'c3.4xlarge'|'c3.8xlarge'|'c4.large'|'c4.xlarge'|'c4.2xlarge'|'c4.4xlarge'|'c4.8xlarge'|'c5.large'|'c5.xlarge'|'c5.2xlarge'|'c5.4xlarge'|'c5.9xlarge'|'c5.12xlarge'|'c5.18xlarge'|'c5.24xlarge'|'c5.metal'|'c5a.large'|'c5a.xlarge'|'c5a.2xlarge'|'c5a.4xlarge'|'c5a.8xlarge'|'c5a.12xlarge'|'c5a.16xlarge'|'c5a.24xlarge'|'c5ad.large'|'c5ad.xlarge'|'c5ad.2xlarge'|'c5ad.4xlarge'|'c5ad.8xlarge'|'c5ad.12xlarge'|'c5ad.16xlarge'|'c5ad.24xlarge'|'c5d.large'|'c5d.xlarge'|'c5d.2xlarge'|'c5d.4xlarge'|'c5d.9xlarge'|'c5d.12xlarge'|'c5d.18xlarge'|'c5d.24xlarge'|'c5d.metal'|'c5n.large'|'c5n.xlarge'|'c5n.2xlarge'|'c5n.4xlarge'|'c5n.9xlarge'|'c5n.18xlarge'|'c5n.metal'|'c6g.metal'|'c6g.medium'|'c6g.large'|'c6g.xlarge'|'c6g.2xlarge'|'c6g.4xlarge'|'c6g.8xlarge'|'c6g.12xlarge'|'c6g.16xlarge'|'c6gd.metal'|'c6gd.medium'|'c6gd.large'|'c6gd.xlarge'|'c6gd.2xlarge'|'c6gd.4xlarge'|'c6gd.8xlarge'|'c6gd.12xlarge'|'c6gd.16xlarge'|'c6gn.medium'|'c6gn.large'|'c6gn.xlarge'|'c6gn.2xlarge'|'c6gn.4xlarge'|'c6gn.8xlarge'|'c6gn.12xlarge'|'c6gn.16xlarge'|'cc1.4xlarge'|'cc2.8xlarge'|'g2.2xlarge'|'g2.8xlarge'|'g3.4xlarge'|'g3.8xlarge'|'g3.16xlarge'|'g3s.xlarge'|'g4ad.4xlarge'|'g4ad.8xlarge'|'g4ad.16xlarge'|'g4dn.xlarge'|'g4dn.2xlarge'|'g4dn.4xlarge'|'g4dn.8xlarge'|'g4dn.12xlarge'|'g4dn.16xlarge'|'g4dn.metal'|'cg1.4xlarge'|'p2.xlarge'|'p2.8xlarge'|'p2.16xlarge'|'p3.2xlarge'|'p3.8xlarge'|'p3.16xlarge'|'p3dn.24xlarge'|'p4d.24xlarge'|'d2.xlarge'|'d2.2xlarge'|'d2.4xlarge'|'d2.8xlarge'|'d3.xlarge'|'d3.2xlarge'|'d3.4xlarge'|'d3.8xlarge'|'d3en.xlarge'|'d3en.2xlarge'|'d3en.4xlarge'|'d3en.6xlarge'|'d3en.8xlarge'|'d3en.12xlarge'|'f1.2xlarge'|'f1.4xlarge'|'f1.16xlarge'|'m5.large'|'m5.xlarge'|'m5.2xlarge'|'m5.4xlarge'|'m5.8xlarge'|'m5.12xlarge'|'m5.16xlarge'|'m5.24xlarge'|'m5.metal'|'m5a.large'|'m5a.xlarge'|'m5a.2xlarge'|'m5a.4xlarge'|'m5a.8xlarge'|'m5a.12xlarge'|'m5a.16xlarge'|'m5a.24xlarge'|'m5d.large'|'m5d.xlarge'|'m5d.2xlarge'|'m5d.4xlarge'|'m5d.8xlarge'|'m5d.12xlarge'|'m5d.16xlarge'|'m5d.24xlarge'|'m5d.metal'|'m5ad.large'|'m5ad.xlarge'|'m5ad.2xlarge'|'m5ad.4xlarge'|'m5ad.8xlarge'|'m5ad.12xlarge'|'m5ad.16xlarge'|'m5ad.24xlarge'|'m5zn.large'|'m5zn.xlarge'|'m5zn.2xlarge'|'m5zn.3xlarge'|'m5zn.6xlarge'|'m5zn.12xlarge'|'m5zn.metal'|'h1.2xlarge'|'h1.4xlarge'|'h1.8xlarge'|'h1.16xlarge'|'z1d.large'|'z1d.xlarge'|'z1d.2xlarge'|'z1d.3xlarge'|'z1d.6xlarge'|'z1d.12xlarge'|'z1d.metal'|'u-6tb1.metal'|'u-9tb1.metal'|'u-12tb1.metal'|'u-18tb1.metal'|'u-24tb1.metal'|'a1.medium'|'a1.large'|'a1.xlarge'|'a1.2xlarge'|'a1.4xlarge'|'a1.metal'|'m5dn.large'|'m5dn.xlarge'|'m5dn.2xlarge'|'m5dn.4xlarge'|'m5dn.8xlarge'|'m5dn.12xlarge'|'m5dn.16xlarge'|'m5dn.24xlarge'|'m5n.large'|'m5n.xlarge'|'m5n.2xlarge'|'m5n.4xlarge'|'m5n.8xlarge'|'m5n.12xlarge'|'m5n.16xlarge'|'m5n.24xlarge'|'r5dn.large'|'r5dn.xlarge'|'r5dn.2xlarge'|'r5dn.4xlarge'|'r5dn.8xlarge'|'r5dn.12xlarge'|'r5dn.16xlarge'|'r5dn.24xlarge'|'r5n.large'|'r5n.xlarge'|'r5n.2xlarge'|'r5n.4xlarge'|'r5n.8xlarge'|'r5n.12xlarge'|'r5n.16xlarge'|'r5n.24xlarge'|'inf1.xlarge'|'inf1.2xlarge'|'inf1.6xlarge'|'inf1.24xlarge'|'m6g.metal'|'m6g.medium'|'m6g.large'|'m6g.xlarge'|'m6g.2xlarge'|'m6g.4xlarge'|'m6g.8xlarge'|'m6g.12xlarge'|'m6g.16xlarge'|'m6gd.metal'|'m6gd.medium'|'m6gd.large'|'m6gd.xlarge'|'m6gd.2xlarge'|'m6gd.4xlarge'|'m6gd.8xlarge'|'m6gd.12xlarge'|'m6gd.16xlarge'|'mac1.metal',
                            'MaxPrice': 'string',
                            'SubnetId': 'string',
                            'AvailabilityZone': 'string',
                            'WeightedCapacity': 123.0,
                            'Priority': 123.0,
                            'Placement': {
                                'GroupName': 'string'
                            }
                        },
                    ]
                },
            ],
            'TargetCapacitySpecification': {
                'TotalTargetCapacity': 123,
                'OnDemandTargetCapacity': 123,
                'SpotTargetCapacity': 123,
                'DefaultTargetCapacityType': 'spot'|'on-demand'
            },
            'TerminateInstancesWithExpiration': True|False,
            'Type': 'request'|'maintain'|'instant',
            'ValidFrom': datetime(2015, 1, 1),
            'ValidUntil': datetime(2015, 1, 1),
            'ReplaceUnhealthyInstances': True|False,
            'SpotOptions': {
                'AllocationStrategy': 'lowest-price'|'diversified'|'capacity-optimized',
                'MaintenanceStrategies': {
                    'CapacityRebalance': {
                        'ReplacementStrategy': 'launch'
                    }
                },
                'InstanceInterruptionBehavior': 'hibernate'|'stop'|'terminate',
                'InstancePoolsToUseCount': 123,
                'SingleInstanceType': True|False,
                'SingleAvailabilityZone': True|False,
                'MinTargetCapacity': 123,
                'MaxTotalPrice': 'string'
            },
            'OnDemandOptions': {
                'AllocationStrategy': 'lowest-price'|'prioritized',
                'CapacityReservationOptions': {
                    'UsageStrategy': 'use-capacity-reservations-first'
                },
                'SingleInstanceType': True|False,
                'SingleAvailabilityZone': True|False,
                'MinTargetCapacity': 123,
                'MaxTotalPrice': 'string'
            },
            'Tags': [
                {
                    'Key': 'string',
                    'Value': 'string'
                },
            ],
            'Errors': [
                {
                    'LaunchTemplateAndOverrides': {
                        'LaunchTemplateSpecification': {
                            'LaunchTemplateId': 'string',
                            'LaunchTemplateName': 'string',
                            'Version': 'string'
                        },
                        'Overrides': {
                            'InstanceType': 't1.micro'|'t2.nano'|'t2.micro'|'t2.small'|'t2.medium'|'t2.large'|'t2.xlarge'|'t2.2xlarge'|'t3.nano'|'t3.micro'|'t3.small'|'t3.medium'|'t3.large'|'t3.xlarge'|'t3.2xlarge'|'t3a.nano'|'t3a.micro'|'t3a.small'|'t3a.medium'|'t3a.large'|'t3a.xlarge'|'t3a.2xlarge'|'t4g.nano'|'t4g.micro'|'t4g.small'|'t4g.medium'|'t4g.large'|'t4g.xlarge'|'t4g.2xlarge'|'m1.small'|'m1.medium'|'m1.large'|'m1.xlarge'|'m3.medium'|'m3.large'|'m3.xlarge'|'m3.2xlarge'|'m4.large'|'m4.xlarge'|'m4.2xlarge'|'m4.4xlarge'|'m4.10xlarge'|'m4.16xlarge'|'m2.xlarge'|'m2.2xlarge'|'m2.4xlarge'|'cr1.8xlarge'|'r3.large'|'r3.xlarge'|'r3.2xlarge'|'r3.4xlarge'|'r3.8xlarge'|'r4.large'|'r4.xlarge'|'r4.2xlarge'|'r4.4xlarge'|'r4.8xlarge'|'r4.16xlarge'|'r5.large'|'r5.xlarge'|'r5.2xlarge'|'r5.4xlarge'|'r5.8xlarge'|'r5.12xlarge'|'r5.16xlarge'|'r5.24xlarge'|'r5.metal'|'r5a.large'|'r5a.xlarge'|'r5a.2xlarge'|'r5a.4xlarge'|'r5a.8xlarge'|'r5a.12xlarge'|'r5a.16xlarge'|'r5a.24xlarge'|'r5b.large'|'r5b.xlarge'|'r5b.2xlarge'|'r5b.4xlarge'|'r5b.8xlarge'|'r5b.12xlarge'|'r5b.16xlarge'|'r5b.24xlarge'|'r5b.metal'|'r5d.large'|'r5d.xlarge'|'r5d.2xlarge'|'r5d.4xlarge'|'r5d.8xlarge'|'r5d.12xlarge'|'r5d.16xlarge'|'r5d.24xlarge'|'r5d.metal'|'r5ad.large'|'r5ad.xlarge'|'r5ad.2xlarge'|'r5ad.4xlarge'|'r5ad.8xlarge'|'r5ad.12xlarge'|'r5ad.16xlarge'|'r5ad.24xlarge'|'r6g.metal'|'r6g.medium'|'r6g.large'|'r6g.xlarge'|'r6g.2xlarge'|'r6g.4xlarge'|'r6g.8xlarge'|'r6g.12xlarge'|'r6g.16xlarge'|'r6gd.metal'|'r6gd.medium'|'r6gd.large'|'r6gd.xlarge'|'r6gd.2xlarge'|'r6gd.4xlarge'|'r6gd.8xlarge'|'r6gd.12xlarge'|'r6gd.16xlarge'|'x1.16xlarge'|'x1.32xlarge'|'x1e.xlarge'|'x1e.2xlarge'|'x1e.4xlarge'|'x1e.8xlarge'|'x1e.16xlarge'|'x1e.32xlarge'|'i2.xlarge'|'i2.2xlarge'|'i2.4xlarge'|'i2.8xlarge'|'i3.large'|'i3.xlarge'|'i3.2xlarge'|'i3.4xlarge'|'i3.8xlarge'|'i3.16xlarge'|'i3.metal'|'i3en.large'|'i3en.xlarge'|'i3en.2xlarge'|'i3en.3xlarge'|'i3en.6xlarge'|'i3en.12xlarge'|'i3en.24xlarge'|'i3en.metal'|'hi1.4xlarge'|'hs1.8xlarge'|'c1.medium'|'c1.xlarge'|'c3.large'|'c3.xlarge'|'c3.2xlarge'|'c3.4xlarge'|'c3.8xlarge'|'c4.large'|'c4.xlarge'|'c4.2xlarge'|'c4.4xlarge'|'c4.8xlarge'|'c5.large'|'c5.xlarge'|'c5.2xlarge'|'c5.4xlarge'|'c5.9xlarge'|'c5.12xlarge'|'c5.18xlarge'|'c5.24xlarge'|'c5.metal'|'c5a.large'|'c5a.xlarge'|'c5a.2xlarge'|'c5a.4xlarge'|'c5a.8xlarge'|'c5a.12xlarge'|'c5a.16xlarge'|'c5a.24xlarge'|'c5ad.large'|'c5ad.xlarge'|'c5ad.2xlarge'|'c5ad.4xlarge'|'c5ad.8xlarge'|'c5ad.12xlarge'|'c5ad.16xlarge'|'c5ad.24xlarge'|'c5d.large'|'c5d.xlarge'|'c5d.2xlarge'|'c5d.4xlarge'|'c5d.9xlarge'|'c5d.12xlarge'|'c5d.18xlarge'|'c5d.24xlarge'|'c5d.metal'|'c5n.large'|'c5n.xlarge'|'c5n.2xlarge'|'c5n.4xlarge'|'c5n.9xlarge'|'c5n.18xlarge'|'c5n.metal'|'c6g.metal'|'c6g.medium'|'c6g.large'|'c6g.xlarge'|'c6g.2xlarge'|'c6g.4xlarge'|'c6g.8xlarge'|'c6g.12xlarge'|'c6g.16xlarge'|'c6gd.metal'|'c6gd.medium'|'c6gd.large'|'c6gd.xlarge'|'c6gd.2xlarge'|'c6gd.4xlarge'|'c6gd.8xlarge'|'c6gd.12xlarge'|'c6gd.16xlarge'|'c6gn.medium'|'c6gn.large'|'c6gn.xlarge'|'c6gn.2xlarge'|'c6gn.4xlarge'|'c6gn.8xlarge'|'c6gn.12xlarge'|'c6gn.16xlarge'|'cc1.4xlarge'|'cc2.8xlarge'|'g2.2xlarge'|'g2.8xlarge'|'g3.4xlarge'|'g3.8xlarge'|'g3.16xlarge'|'g3s.xlarge'|'g4ad.4xlarge'|'g4ad.8xlarge'|'g4ad.16xlarge'|'g4dn.xlarge'|'g4dn.2xlarge'|'g4dn.4xlarge'|'g4dn.8xlarge'|'g4dn.12xlarge'|'g4dn.16xlarge'|'g4dn.metal'|'cg1.4xlarge'|'p2.xlarge'|'p2.8xlarge'|'p2.16xlarge'|'p3.2xlarge'|'p3.8xlarge'|'p3.16xlarge'|'p3dn.24xlarge'|'p4d.24xlarge'|'d2.xlarge'|'d2.2xlarge'|'d2.4xlarge'|'d2.8xlarge'|'d3.xlarge'|'d3.2xlarge'|'d3.4xlarge'|'d3.8xlarge'|'d3en.xlarge'|'d3en.2xlarge'|'d3en.4xlarge'|'d3en.6xlarge'|'d3en.8xlarge'|'d3en.12xlarge'|'f1.2xlarge'|'f1.4xlarge'|'f1.16xlarge'|'m5.large'|'m5.xlarge'|'m5.2xlarge'|'m5.4xlarge'|'m5.8xlarge'|'m5.12xlarge'|'m5.16xlarge'|'m5.24xlarge'|'m5.metal'|'m5a.large'|'m5a.xlarge'|'m5a.2xlarge'|'m5a.4xlarge'|'m5a.8xlarge'|'m5a.12xlarge'|'m5a.16xlarge'|'m5a.24xlarge'|'m5d.large'|'m5d.xlarge'|'m5d.2xlarge'|'m5d.4xlarge'|'m5d.8xlarge'|'m5d.12xlarge'|'m5d.16xlarge'|'m5d.24xlarge'|'m5d.metal'|'m5ad.large'|'m5ad.xlarge'|'m5ad.2xlarge'|'m5ad.4xlarge'|'m5ad.8xlarge'|'m5ad.12xlarge'|'m5ad.16xlarge'|'m5ad.24xlarge'|'m5zn.large'|'m5zn.xlarge'|'m5zn.2xlarge'|'m5zn.3xlarge'|'m5zn.6xlarge'|'m5zn.12xlarge'|'m5zn.metal'|'h1.2xlarge'|'h1.4xlarge'|'h1.8xlarge'|'h1.16xlarge'|'z1d.large'|'z1d.xlarge'|'z1d.2xlarge'|'z1d.3xlarge'|'z1d.6xlarge'|'z1d.12xlarge'|'z1d.metal'|'u-6tb1.metal'|'u-9tb1.metal'|'u-12tb1.metal'|'u-18tb1.metal'|'u-24tb1.metal'|'a1.medium'|'a1.large'|'a1.xlarge'|'a1.2xlarge'|'a1.4xlarge'|'a1.metal'|'m5dn.large'|'m5dn.xlarge'|'m5dn.2xlarge'|'m5dn.4xlarge'|'m5dn.8xlarge'|'m5dn.12xlarge'|'m5dn.16xlarge'|'m5dn.24xlarge'|'m5n.large'|'m5n.xlarge'|'m5n.2xlarge'|'m5n.4xlarge'|'m5n.8xlarge'|'m5n.12xlarge'|'m5n.16xlarge'|'m5n.24xlarge'|'r5dn.large'|'r5dn.xlarge'|'r5dn.2xlarge'|'r5dn.4xlarge'|'r5dn.8xlarge'|'r5dn.12xlarge'|'r5dn.16xlarge'|'r5dn.24xlarge'|'r5n.large'|'r5n.xlarge'|'r5n.2xlarge'|'r5n.4xlarge'|'r5n.8xlarge'|'r5n.12xlarge'|'r5n.16xlarge'|'r5n.24xlarge'|'inf1.xlarge'|'inf1.2xlarge'|'inf1.6xlarge'|'inf1.24xlarge'|'m6g.metal'|'m6g.medium'|'m6g.large'|'m6g.xlarge'|'m6g.2xlarge'|'m6g.4xlarge'|'m6g.8xlarge'|'m6g.12xlarge'|'m6g.16xlarge'|'m6gd.metal'|'m6gd.medium'|'m6gd.large'|'m6gd.xlarge'|'m6gd.2xlarge'|'m6gd.4xlarge'|'m6gd.8xlarge'|'m6gd.12xlarge'|'m6gd.16xlarge'|'mac1.metal',
                            'MaxPrice': 'string',
                            'SubnetId': 'string',
                            'AvailabilityZone': 'string',
                            'WeightedCapacity': 123.0,
                            'Priority': 123.0,
                            'Placement': {
                                'GroupName': 'string'
                            }
                        }
                    },
                    'Lifecycle': 'spot'|'on-demand',
                    'ErrorCode': 'string',
                    'ErrorMessage': 'string'
                },
            ],
            'Instances': [
                {
                    'LaunchTemplateAndOverrides': {
                        'LaunchTemplateSpecification': {
                            'LaunchTemplateId': 'string',
                            'LaunchTemplateName': 'string',
                            'Version': 'string'
                        },
                        'Overrides': {
                            'InstanceType': 't1.micro'|'t2.nano'|'t2.micro'|'t2.small'|'t2.medium'|'t2.large'|'t2.xlarge'|'t2.2xlarge'|'t3.nano'|'t3.micro'|'t3.small'|'t3.medium'|'t3.large'|'t3.xlarge'|'t3.2xlarge'|'t3a.nano'|'t3a.micro'|'t3a.small'|'t3a.medium'|'t3a.large'|'t3a.xlarge'|'t3a.2xlarge'|'t4g.nano'|'t4g.micro'|'t4g.small'|'t4g.medium'|'t4g.large'|'t4g.xlarge'|'t4g.2xlarge'|'m1.small'|'m1.medium'|'m1.large'|'m1.xlarge'|'m3.medium'|'m3.large'|'m3.xlarge'|'m3.2xlarge'|'m4.large'|'m4.xlarge'|'m4.2xlarge'|'m4.4xlarge'|'m4.10xlarge'|'m4.16xlarge'|'m2.xlarge'|'m2.2xlarge'|'m2.4xlarge'|'cr1.8xlarge'|'r3.large'|'r3.xlarge'|'r3.2xlarge'|'r3.4xlarge'|'r3.8xlarge'|'r4.large'|'r4.xlarge'|'r4.2xlarge'|'r4.4xlarge'|'r4.8xlarge'|'r4.16xlarge'|'r5.large'|'r5.xlarge'|'r5.2xlarge'|'r5.4xlarge'|'r5.8xlarge'|'r5.12xlarge'|'r5.16xlarge'|'r5.24xlarge'|'r5.metal'|'r5a.large'|'r5a.xlarge'|'r5a.2xlarge'|'r5a.4xlarge'|'r5a.8xlarge'|'r5a.12xlarge'|'r5a.16xlarge'|'r5a.24xlarge'|'r5b.large'|'r5b.xlarge'|'r5b.2xlarge'|'r5b.4xlarge'|'r5b.8xlarge'|'r5b.12xlarge'|'r5b.16xlarge'|'r5b.24xlarge'|'r5b.metal'|'r5d.large'|'r5d.xlarge'|'r5d.2xlarge'|'r5d.4xlarge'|'r5d.8xlarge'|'r5d.12xlarge'|'r5d.16xlarge'|'r5d.24xlarge'|'r5d.metal'|'r5ad.large'|'r5ad.xlarge'|'r5ad.2xlarge'|'r5ad.4xlarge'|'r5ad.8xlarge'|'r5ad.12xlarge'|'r5ad.16xlarge'|'r5ad.24xlarge'|'r6g.metal'|'r6g.medium'|'r6g.large'|'r6g.xlarge'|'r6g.2xlarge'|'r6g.4xlarge'|'r6g.8xlarge'|'r6g.12xlarge'|'r6g.16xlarge'|'r6gd.metal'|'r6gd.medium'|'r6gd.large'|'r6gd.xlarge'|'r6gd.2xlarge'|'r6gd.4xlarge'|'r6gd.8xlarge'|'r6gd.12xlarge'|'r6gd.16xlarge'|'x1.16xlarge'|'x1.32xlarge'|'x1e.xlarge'|'x1e.2xlarge'|'x1e.4xlarge'|'x1e.8xlarge'|'x1e.16xlarge'|'x1e.32xlarge'|'i2.xlarge'|'i2.2xlarge'|'i2.4xlarge'|'i2.8xlarge'|'i3.large'|'i3.xlarge'|'i3.2xlarge'|'i3.4xlarge'|'i3.8xlarge'|'i3.16xlarge'|'i3.metal'|'i3en.large'|'i3en.xlarge'|'i3en.2xlarge'|'i3en.3xlarge'|'i3en.6xlarge'|'i3en.12xlarge'|'i3en.24xlarge'|'i3en.metal'|'hi1.4xlarge'|'hs1.8xlarge'|'c1.medium'|'c1.xlarge'|'c3.large'|'c3.xlarge'|'c3.2xlarge'|'c3.4xlarge'|'c3.8xlarge'|'c4.large'|'c4.xlarge'|'c4.2xlarge'|'c4.4xlarge'|'c4.8xlarge'|'c5.large'|'c5.xlarge'|'c5.2xlarge'|'c5.4xlarge'|'c5.9xlarge'|'c5.12xlarge'|'c5.18xlarge'|'c5.24xlarge'|'c5.metal'|'c5a.large'|'c5a.xlarge'|'c5a.2xlarge'|'c5a.4xlarge'|'c5a.8xlarge'|'c5a.12xlarge'|'c5a.16xlarge'|'c5a.24xlarge'|'c5ad.large'|'c5ad.xlarge'|'c5ad.2xlarge'|'c5ad.4xlarge'|'c5ad.8xlarge'|'c5ad.12xlarge'|'c5ad.16xlarge'|'c5ad.24xlarge'|'c5d.large'|'c5d.xlarge'|'c5d.2xlarge'|'c5d.4xlarge'|'c5d.9xlarge'|'c5d.12xlarge'|'c5d.18xlarge'|'c5d.24xlarge'|'c5d.metal'|'c5n.large'|'c5n.xlarge'|'c5n.2xlarge'|'c5n.4xlarge'|'c5n.9xlarge'|'c5n.18xlarge'|'c5n.metal'|'c6g.metal'|'c6g.medium'|'c6g.large'|'c6g.xlarge'|'c6g.2xlarge'|'c6g.4xlarge'|'c6g.8xlarge'|'c6g.12xlarge'|'c6g.16xlarge'|'c6gd.metal'|'c6gd.medium'|'c6gd.large'|'c6gd.xlarge'|'c6gd.2xlarge'|'c6gd.4xlarge'|'c6gd.8xlarge'|'c6gd.12xlarge'|'c6gd.16xlarge'|'c6gn.medium'|'c6gn.large'|'c6gn.xlarge'|'c6gn.2xlarge'|'c6gn.4xlarge'|'c6gn.8xlarge'|'c6gn.12xlarge'|'c6gn.16xlarge'|'cc1.4xlarge'|'cc2.8xlarge'|'g2.2xlarge'|'g2.8xlarge'|'g3.4xlarge'|'g3.8xlarge'|'g3.16xlarge'|'g3s.xlarge'|'g4ad.4xlarge'|'g4ad.8xlarge'|'g4ad.16xlarge'|'g4dn.xlarge'|'g4dn.2xlarge'|'g4dn.4xlarge'|'g4dn.8xlarge'|'g4dn.12xlarge'|'g4dn.16xlarge'|'g4dn.metal'|'cg1.4xlarge'|'p2.xlarge'|'p2.8xlarge'|'p2.16xlarge'|'p3.2xlarge'|'p3.8xlarge'|'p3.16xlarge'|'p3dn.24xlarge'|'p4d.24xlarge'|'d2.xlarge'|'d2.2xlarge'|'d2.4xlarge'|'d2.8xlarge'|'d3.xlarge'|'d3.2xlarge'|'d3.4xlarge'|'d3.8xlarge'|'d3en.xlarge'|'d3en.2xlarge'|'d3en.4xlarge'|'d3en.6xlarge'|'d3en.8xlarge'|'d3en.12xlarge'|'f1.2xlarge'|'f1.4xlarge'|'f1.16xlarge'|'m5.large'|'m5.xlarge'|'m5.2xlarge'|'m5.4xlarge'|'m5.8xlarge'|'m5.12xlarge'|'m5.16xlarge'|'m5.24xlarge'|'m5.metal'|'m5a.large'|'m5a.xlarge'|'m5a.2xlarge'|'m5a.4xlarge'|'m5a.8xlarge'|'m5a.12xlarge'|'m5a.16xlarge'|'m5a.24xlarge'|'m5d.large'|'m5d.xlarge'|'m5d.2xlarge'|'m5d.4xlarge'|'m5d.8xlarge'|'m5d.12xlarge'|'m5d.16xlarge'|'m5d.24xlarge'|'m5d.metal'|'m5ad.large'|'m5ad.xlarge'|'m5ad.2xlarge'|'m5ad.4xlarge'|'m5ad.8xlarge'|'m5ad.12xlarge'|'m5ad.16xlarge'|'m5ad.24xlarge'|'m5zn.large'|'m5zn.xlarge'|'m5zn.2xlarge'|'m5zn.3xlarge'|'m5zn.6xlarge'|'m5zn.12xlarge'|'m5zn.metal'|'h1.2xlarge'|'h1.4xlarge'|'h1.8xlarge'|'h1.16xlarge'|'z1d.large'|'z1d.xlarge'|'z1d.2xlarge'|'z1d.3xlarge'|'z1d.6xlarge'|'z1d.12xlarge'|'z1d.metal'|'u-6tb1.metal'|'u-9tb1.metal'|'u-12tb1.metal'|'u-18tb1.metal'|'u-24tb1.metal'|'a1.medium'|'a1.large'|'a1.xlarge'|'a1.2xlarge'|'a1.4xlarge'|'a1.metal'|'m5dn.large'|'m5dn.xlarge'|'m5dn.2xlarge'|'m5dn.4xlarge'|'m5dn.8xlarge'|'m5dn.12xlarge'|'m5dn.16xlarge'|'m5dn.24xlarge'|'m5n.large'|'m5n.xlarge'|'m5n.2xlarge'|'m5n.4xlarge'|'m5n.8xlarge'|'m5n.12xlarge'|'m5n.16xlarge'|'m5n.24xlarge'|'r5dn.large'|'r5dn.xlarge'|'r5dn.2xlarge'|'r5dn.4xlarge'|'r5dn.8xlarge'|'r5dn.12xlarge'|'r5dn.16xlarge'|'r5dn.24xlarge'|'r5n.large'|'r5n.xlarge'|'r5n.2xlarge'|'r5n.4xlarge'|'r5n.8xlarge'|'r5n.12xlarge'|'r5n.16xlarge'|'r5n.24xlarge'|'inf1.xlarge'|'inf1.2xlarge'|'inf1.6xlarge'|'inf1.24xlarge'|'m6g.metal'|'m6g.medium'|'m6g.large'|'m6g.xlarge'|'m6g.2xlarge'|'m6g.4xlarge'|'m6g.8xlarge'|'m6g.12xlarge'|'m6g.16xlarge'|'m6gd.metal'|'m6gd.medium'|'m6gd.large'|'m6gd.xlarge'|'m6gd.2xlarge'|'m6gd.4xlarge'|'m6gd.8xlarge'|'m6gd.12xlarge'|'m6gd.16xlarge'|'mac1.metal',
                            'MaxPrice': 'string',
                            'SubnetId': 'string',
                            'AvailabilityZone': 'string',
                            'WeightedCapacity': 123.0,
                            'Priority': 123.0,
                            'Placement': {
                                'GroupName': 'string'
                            }
                        }
                    },
                    'Lifecycle': 'spot'|'on-demand',
                    'InstanceIds': [
                        'string',
                    ],
                    'InstanceType': 't1.micro'|'t2.nano'|'t2.micro'|'t2.small'|'t2.medium'|'t2.large'|'t2.xlarge'|'t2.2xlarge'|'t3.nano'|'t3.micro'|'t3.small'|'t3.medium'|'t3.large'|'t3.xlarge'|'t3.2xlarge'|'t3a.nano'|'t3a.micro'|'t3a.small'|'t3a.medium'|'t3a.large'|'t3a.xlarge'|'t3a.2xlarge'|'t4g.nano'|'t4g.micro'|'t4g.small'|'t4g.medium'|'t4g.large'|'t4g.xlarge'|'t4g.2xlarge'|'m1.small'|'m1.medium'|'m1.large'|'m1.xlarge'|'m3.medium'|'m3.large'|'m3.xlarge'|'m3.2xlarge'|'m4.large'|'m4.xlarge'|'m4.2xlarge'|'m4.4xlarge'|'m4.10xlarge'|'m4.16xlarge'|'m2.xlarge'|'m2.2xlarge'|'m2.4xlarge'|'cr1.8xlarge'|'r3.large'|'r3.xlarge'|'r3.2xlarge'|'r3.4xlarge'|'r3.8xlarge'|'r4.large'|'r4.xlarge'|'r4.2xlarge'|'r4.4xlarge'|'r4.8xlarge'|'r4.16xlarge'|'r5.large'|'r5.xlarge'|'r5.2xlarge'|'r5.4xlarge'|'r5.8xlarge'|'r5.12xlarge'|'r5.16xlarge'|'r5.24xlarge'|'r5.metal'|'r5a.large'|'r5a.xlarge'|'r5a.2xlarge'|'r5a.4xlarge'|'r5a.8xlarge'|'r5a.12xlarge'|'r5a.16xlarge'|'r5a.24xlarge'|'r5b.large'|'r5b.xlarge'|'r5b.2xlarge'|'r5b.4xlarge'|'r5b.8xlarge'|'r5b.12xlarge'|'r5b.16xlarge'|'r5b.24xlarge'|'r5b.metal'|'r5d.large'|'r5d.xlarge'|'r5d.2xlarge'|'r5d.4xlarge'|'r5d.8xlarge'|'r5d.12xlarge'|'r5d.16xlarge'|'r5d.24xlarge'|'r5d.metal'|'r5ad.large'|'r5ad.xlarge'|'r5ad.2xlarge'|'r5ad.4xlarge'|'r5ad.8xlarge'|'r5ad.12xlarge'|'r5ad.16xlarge'|'r5ad.24xlarge'|'r6g.metal'|'r6g.medium'|'r6g.large'|'r6g.xlarge'|'r6g.2xlarge'|'r6g.4xlarge'|'r6g.8xlarge'|'r6g.12xlarge'|'r6g.16xlarge'|'r6gd.metal'|'r6gd.medium'|'r6gd.large'|'r6gd.xlarge'|'r6gd.2xlarge'|'r6gd.4xlarge'|'r6gd.8xlarge'|'r6gd.12xlarge'|'r6gd.16xlarge'|'x1.16xlarge'|'x1.32xlarge'|'x1e.xlarge'|'x1e.2xlarge'|'x1e.4xlarge'|'x1e.8xlarge'|'x1e.16xlarge'|'x1e.32xlarge'|'i2.xlarge'|'i2.2xlarge'|'i2.4xlarge'|'i2.8xlarge'|'i3.large'|'i3.xlarge'|'i3.2xlarge'|'i3.4xlarge'|'i3.8xlarge'|'i3.16xlarge'|'i3.metal'|'i3en.large'|'i3en.xlarge'|'i3en.2xlarge'|'i3en.3xlarge'|'i3en.6xlarge'|'i3en.12xlarge'|'i3en.24xlarge'|'i3en.metal'|'hi1.4xlarge'|'hs1.8xlarge'|'c1.medium'|'c1.xlarge'|'c3.large'|'c3.xlarge'|'c3.2xlarge'|'c3.4xlarge'|'c3.8xlarge'|'c4.large'|'c4.xlarge'|'c4.2xlarge'|'c4.4xlarge'|'c4.8xlarge'|'c5.large'|'c5.xlarge'|'c5.2xlarge'|'c5.4xlarge'|'c5.9xlarge'|'c5.12xlarge'|'c5.18xlarge'|'c5.24xlarge'|'c5.metal'|'c5a.large'|'c5a.xlarge'|'c5a.2xlarge'|'c5a.4xlarge'|'c5a.8xlarge'|'c5a.12xlarge'|'c5a.16xlarge'|'c5a.24xlarge'|'c5ad.large'|'c5ad.xlarge'|'c5ad.2xlarge'|'c5ad.4xlarge'|'c5ad.8xlarge'|'c5ad.12xlarge'|'c5ad.16xlarge'|'c5ad.24xlarge'|'c5d.large'|'c5d.xlarge'|'c5d.2xlarge'|'c5d.4xlarge'|'c5d.9xlarge'|'c5d.12xlarge'|'c5d.18xlarge'|'c5d.24xlarge'|'c5d.metal'|'c5n.large'|'c5n.xlarge'|'c5n.2xlarge'|'c5n.4xlarge'|'c5n.9xlarge'|'c5n.18xlarge'|'c5n.metal'|'c6g.metal'|'c6g.medium'|'c6g.large'|'c6g.xlarge'|'c6g.2xlarge'|'c6g.4xlarge'|'c6g.8xlarge'|'c6g.12xlarge'|'c6g.16xlarge'|'c6gd.metal'|'c6gd.medium'|'c6gd.large'|'c6gd.xlarge'|'c6gd.2xlarge'|'c6gd.4xlarge'|'c6gd.8xlarge'|'c6gd.12xlarge'|'c6gd.16xlarge'|'c6gn.medium'|'c6gn.large'|'c6gn.xlarge'|'c6gn.2xlarge'|'c6gn.4xlarge'|'c6gn.8xlarge'|'c6gn.12xlarge'|'c6gn.16xlarge'|'cc1.4xlarge'|'cc2.8xlarge'|'g2.2xlarge'|'g2.8xlarge'|'g3.4xlarge'|'g3.8xlarge'|'g3.16xlarge'|'g3s.xlarge'|'g4ad.4xlarge'|'g4ad.8xlarge'|'g4ad.16xlarge'|'g4dn.xlarge'|'g4dn.2xlarge'|'g4dn.4xlarge'|'g4dn.8xlarge'|'g4dn.12xlarge'|'g4dn.16xlarge'|'g4dn.metal'|'cg1.4xlarge'|'p2.xlarge'|'p2.8xlarge'|'p2.16xlarge'|'p3.2xlarge'|'p3.8xlarge'|'p3.16xlarge'|'p3dn.24xlarge'|'p4d.24xlarge'|'d2.xlarge'|'d2.2xlarge'|'d2.4xlarge'|'d2.8xlarge'|'d3.xlarge'|'d3.2xlarge'|'d3.4xlarge'|'d3.8xlarge'|'d3en.xlarge'|'d3en.2xlarge'|'d3en.4xlarge'|'d3en.6xlarge'|'d3en.8xlarge'|'d3en.12xlarge'|'f1.2xlarge'|'f1.4xlarge'|'f1.16xlarge'|'m5.large'|'m5.xlarge'|'m5.2xlarge'|'m5.4xlarge'|'m5.8xlarge'|'m5.12xlarge'|'m5.16xlarge'|'m5.24xlarge'|'m5.metal'|'m5a.large'|'m5a.xlarge'|'m5a.2xlarge'|'m5a.4xlarge'|'m5a.8xlarge'|'m5a.12xlarge'|'m5a.16xlarge'|'m5a.24xlarge'|'m5d.large'|'m5d.xlarge'|'m5d.2xlarge'|'m5d.4xlarge'|'m5d.8xlarge'|'m5d.12xlarge'|'m5d.16xlarge'|'m5d.24xlarge'|'m5d.metal'|'m5ad.large'|'m5ad.xlarge'|'m5ad.2xlarge'|'m5ad.4xlarge'|'m5ad.8xlarge'|'m5ad.12xlarge'|'m5ad.16xlarge'|'m5ad.24xlarge'|'m5zn.large'|'m5zn.xlarge'|'m5zn.2xlarge'|'m5zn.3xlarge'|'m5zn.6xlarge'|'m5zn.12xlarge'|'m5zn.metal'|'h1.2xlarge'|'h1.4xlarge'|'h1.8xlarge'|'h1.16xlarge'|'z1d.large'|'z1d.xlarge'|'z1d.2xlarge'|'z1d.3xlarge'|'z1d.6xlarge'|'z1d.12xlarge'|'z1d.metal'|'u-6tb1.metal'|'u-9tb1.metal'|'u-12tb1.metal'|'u-18tb1.metal'|'u-24tb1.metal'|'a1.medium'|'a1.large'|'a1.xlarge'|'a1.2xlarge'|'a1.4xlarge'|'a1.metal'|'m5dn.large'|'m5dn.xlarge'|'m5dn.2xlarge'|'m5dn.4xlarge'|'m5dn.8xlarge'|'m5dn.12xlarge'|'m5dn.16xlarge'|'m5dn.24xlarge'|'m5n.large'|'m5n.xlarge'|'m5n.2xlarge'|'m5n.4xlarge'|'m5n.8xlarge'|'m5n.12xlarge'|'m5n.16xlarge'|'m5n.24xlarge'|'r5dn.large'|'r5dn.xlarge'|'r5dn.2xlarge'|'r5dn.4xlarge'|'r5dn.8xlarge'|'r5dn.12xlarge'|'r5dn.16xlarge'|'r5dn.24xlarge'|'r5n.large'|'r5n.xlarge'|'r5n.2xlarge'|'r5n.4xlarge'|'r5n.8xlarge'|'r5n.12xlarge'|'r5n.16xlarge'|'r5n.24xlarge'|'inf1.xlarge'|'inf1.2xlarge'|'inf1.6xlarge'|'inf1.24xlarge'|'m6g.metal'|'m6g.medium'|'m6g.large'|'m6g.xlarge'|'m6g.2xlarge'|'m6g.4xlarge'|'m6g.8xlarge'|'m6g.12xlarge'|'m6g.16xlarge'|'m6gd.metal'|'m6gd.medium'|'m6gd.large'|'m6gd.xlarge'|'m6gd.2xlarge'|'m6gd.4xlarge'|'m6gd.8xlarge'|'m6gd.12xlarge'|'m6gd.16xlarge'|'mac1.metal',
                    'Platform': 'Windows'
                },
            ]
        },
    ]
}

**Response Structure**

- _(dict) --_

    - **Fleets** _(list) --_

        Information about the EC2 Fleets.

        - _(dict) --_

            Describes an EC2 Fleet.

            - **ActivityStatus** _(string) --_

                The progress of the EC2 Fleet. If there is an error, the status is error . After all requests are placed, the status is pending_fulfillment . If the size of the EC2 Fleet is equal to or greater than its target capacity, the status is fulfilled . If the size of the EC2 Fleet is decreased, the status is pending_termination while instances are terminating.

            - **CreateTime** _(datetime) --_

                The creation date and time of the EC2 Fleet.

            - **FleetId** _(string) --_

                The ID of the EC2 Fleet.

            - **FleetState** _(string) --_

                The state of the EC2 Fleet.

            - **ClientToken** _(string) --_

                Unique, case-sensitive identifier that you provide to ensure the idempotency of the request. For more information, see [Ensuring Idempotency](https://docs.aws.amazon.com/AWSEC2/latest/APIReference/Run_Instance_Idempotency.html) .

                Constraints: Maximum 64 ASCII characters

            - **ExcessCapacityTerminationPolicy** _(string) --_

                Indicates whether running instances should be terminated if the target capacity of the EC2 Fleet is decreased below the current size of the EC2 Fleet.

            - **FulfilledCapacity** _(float) --_

                The number of units fulfilled by this request compared to the set target capacity.

            - **FulfilledOnDemandCapacity** _(float) --_

                The number of units fulfilled by this request compared to the set target On-Demand capacity.

            - **LaunchTemplateConfigs** _(list) --_

                The launch template and overrides.

                - _(dict) --_

                    Describes a launch template and overrides.

                    - **LaunchTemplateSpecification** _(dict) --_

                        The launch template.

                        - **LaunchTemplateId** _(string) --_

                            The ID of the launch template. If you specify the template ID, you can't specify the template name.

                        - **LaunchTemplateName** _(string) --_

                            The name of the launch template. If you specify the template name, you can't specify the template ID.

                        - **Version** _(string) --_

                            The launch template version number, $Latest , or $Default . You must specify a value, otherwise the request fails.

                            If the value is $Latest , Amazon EC2 uses the latest version of the launch template.

                            If the value is $Default , Amazon EC2 uses the default version of the launch template.

                    - **Overrides** _(list) --_

                        Any parameters that you specify override the same parameters in the launch template.

                        - _(dict) --_

                            Describes overrides for a launch template.

                            - **InstanceType** _(string) --_

                                The instance type.

                            - **MaxPrice** _(string) --_

                                The maximum price per unit hour that you are willing to pay for a Spot Instance.

                            - **SubnetId** _(string) --_

                                The ID of the subnet in which to launch the instances.

                            - **AvailabilityZone** _(string) --_

                                The Availability Zone in which to launch the instances.

                            - **WeightedCapacity** _(float) --_

                                The number of units provided by the specified instance type.

                            - **Priority** _(float) --_

                                The priority for the launch template override. If **AllocationStrategy** is set to prioritized , EC2 Fleet uses priority to determine which launch template override to use first in fulfilling On-Demand capacity. The highest priority is launched first. Valid values are whole numbers starting at 0 . The lower the number, the higher the priority. If no number is set, the override has the lowest priority.

                            - **Placement** _(dict) --_

                                The location where the instance launched, if applicable.

                                - **GroupName** _(string) --_

                                    The name of the placement group that the instance is in.

            - **TargetCapacitySpecification** _(dict) --_

                The number of units to request. You can choose to set the target capacity in terms of instances or a performance characteristic that is important to your application workload, such as vCPUs, memory, or I/O. If the request type is maintain , you can specify a target capacity of 0 and add capacity later.

                - **TotalTargetCapacity** _(integer) --_

                    The number of units to request, filled using DefaultTargetCapacityType .

                - **OnDemandTargetCapacity** _(integer) --_

                    The number of On-Demand units to request. If you specify a target capacity for Spot units, you cannot specify a target capacity for On-Demand units.

                - **SpotTargetCapacity** _(integer) --_

                    The maximum number of Spot units to launch. If you specify a target capacity for On-Demand units, you cannot specify a target capacity for Spot units.

                - **DefaultTargetCapacityType** _(string) --_

                    The default TotalTargetCapacity , which is either Spot or On-Demand .

            - **TerminateInstancesWithExpiration** _(boolean) --_

                Indicates whether running instances should be terminated when the EC2 Fleet expires.

            - **Type** _(string) --_

                The type of request. Indicates whether the EC2 Fleet only requests the target capacity, or also attempts to maintain it. If you request a certain target capacity, EC2 Fleet only places the required requests; it does not attempt to replenish instances if capacity is diminished, and it does not submit requests in alternative capacity pools if capacity is unavailable. To maintain a certain target capacity, EC2 Fleet places the required requests to meet this target capacity. It also automatically replenishes any interrupted Spot Instances. Default: maintain .

            - **ValidFrom** _(datetime) --_

                The start date and time of the request, in UTC format (for example, _YYYY_ -_MM_ -_DD_ T*HH* :_MM_ :_SS_ Z). The default is to start fulfilling the request immediately.

            - **ValidUntil** _(datetime) --_

                The end date and time of the request, in UTC format (for example, _YYYY_ -_MM_ -_DD_ T*HH* :_MM_ :_SS_ Z). At this point, no new instance requests are placed or able to fulfill the request. The default end date is 7 days from the current date.

            - **ReplaceUnhealthyInstances** _(boolean) --_

                Indicates whether EC2 Fleet should replace unhealthy instances.

            - **SpotOptions** _(dict) --_

                The configuration of Spot Instances in an EC2 Fleet.

                - **AllocationStrategy** _(string) --_

                    Indicates how to allocate the target Spot Instance capacity across the Spot Instance pools specified by the EC2 Fleet.

                    If the allocation strategy is lowest-price , EC2 Fleet launches instances from the Spot Instance pools with the lowest price. This is the default allocation strategy.

                    If the allocation strategy is diversified , EC2 Fleet launches instances from all of the Spot Instance pools that you specify.

                    If the allocation strategy is capacity-optimized , EC2 Fleet launches instances from Spot Instance pools with optimal capacity for the number of instances that are launching.

                - **MaintenanceStrategies** _(dict) --_

                    The strategies for managing your workloads on your Spot Instances that will be interrupted. Currently only the capacity rebalance strategy is available.

                    - **CapacityRebalance** _(dict) --_

                        The strategy to use when Amazon EC2 emits a signal that your Spot Instance is at an elevated risk of being interrupted.

                        - **ReplacementStrategy** _(string) --_

                            To allow EC2 Fleet to launch a replacement Spot Instance when an instance rebalance notification is emitted for an existing Spot Instance in the fleet, specify launch . Only available for fleets of type maintain .

                            Note

                            When a replacement instance is launched, the instance marked for rebalance is not automatically terminated. You can terminate it, or you can leave it running. You are charged for both instances while they are running.

                - **InstanceInterruptionBehavior** _(string) --_

                    The behavior when a Spot Instance is interrupted. The default is terminate .

                - **InstancePoolsToUseCount** _(integer) --_

                    The number of Spot pools across which to allocate your target Spot capacity. Valid only when **AllocationStrategy** is set to lowest-price . EC2 Fleet selects the cheapest Spot pools and evenly allocates your target Spot capacity across the number of Spot pools that you specify.

                - **SingleInstanceType** _(boolean) --_

                    Indicates that the fleet uses a single instance type to launch all Spot Instances in the fleet. Supported only for fleets of type instant .

                - **SingleAvailabilityZone** _(boolean) --_

                    Indicates that the fleet launches all Spot Instances into a single Availability Zone. Supported only for fleets of type instant .

                - **MinTargetCapacity** _(integer) --_

                    The minimum target capacity for Spot Instances in the fleet. If the minimum target capacity is not reached, the fleet launches no instances.

                - **MaxTotalPrice** _(string) --_

                    The maximum amount per hour for Spot Instances that you're willing to pay.

            - **OnDemandOptions** _(dict) --_

                The allocation strategy of On-Demand Instances in an EC2 Fleet.

                - **AllocationStrategy** _(string) --_

                    The order of the launch template overrides to use in fulfilling On-Demand capacity. If you specify lowest-price , EC2 Fleet uses price to determine the order, launching the lowest price first. If you specify prioritized , EC2 Fleet uses the priority that you assigned to each launch template override, launching the highest priority first. If you do not specify a value, EC2 Fleet defaults to lowest-price .

                - **CapacityReservationOptions** _(dict) --_

                    The strategy for using unused Capacity Reservations for fulfilling On-Demand capacity. Supported only for fleets of type instant .

                    - **UsageStrategy** _(string) --_

                        Indicates whether to use unused Capacity Reservations for fulfilling On-Demand capacity.

                        If you specify use-capacity-reservations-first , the fleet uses unused Capacity Reservations to fulfill On-Demand capacity up to the target On-Demand capacity. If multiple instance pools have unused Capacity Reservations, the On-Demand allocation strategy (lowest-price or prioritized ) is applied. If the number of unused Capacity Reservations is less than the On-Demand target capacity, the remaining On-Demand target capacity is launched according to the On-Demand allocation strategy (lowest-price or prioritized ).

                        If you do not specify a value, the fleet fulfils the On-Demand capacity according to the chosen On-Demand allocation strategy.

                - **SingleInstanceType** _(boolean) --_

                    Indicates that the fleet uses a single instance type to launch all On-Demand Instances in the fleet. Supported only for fleets of type instant .

                - **SingleAvailabilityZone** _(boolean) --_

                    Indicates that the fleet launches all On-Demand Instances into a single Availability Zone. Supported only for fleets of type instant .

                - **MinTargetCapacity** _(integer) --_

                    The minimum target capacity for On-Demand Instances in the fleet. If the minimum target capacity is not reached, the fleet launches no instances.

                - **MaxTotalPrice** _(string) --_

                    The maximum amount per hour for On-Demand Instances that you're willing to pay.

            - **Tags** _(list) --_

                The tags for an EC2 Fleet resource.

                - _(dict) --_

                    Describes a tag.

                    - **Key** _(string) --_

                        The key of the tag.

                        Constraints: Tag keys are case-sensitive and accept a maximum of 127 Unicode characters. May not begin with aws: .

                    - **Value** _(string) --_

                        The value of the tag.

                        Constraints: Tag values are case-sensitive and accept a maximum of 255 Unicode characters.

            - **Errors** _(list) --_

                Information about the instances that could not be launched by the fleet. Valid only when **Type** is set to instant .

                - _(dict) --_

                    Describes the instances that could not be launched by the fleet.

                    - **LaunchTemplateAndOverrides** _(dict) --_

                        The launch templates and overrides that were used for launching the instances. The values that you specify in the Overrides replace the values in the launch template.

                        - **LaunchTemplateSpecification** _(dict) --_

                            The launch template.

                            - **LaunchTemplateId** _(string) --_

                                The ID of the launch template. If you specify the template ID, you can't specify the template name.

                            - **LaunchTemplateName** _(string) --_

                                The name of the launch template. If you specify the template name, you can't specify the template ID.

                            - **Version** _(string) --_

                                The launch template version number, $Latest , or $Default . You must specify a value, otherwise the request fails.

                                If the value is $Latest , Amazon EC2 uses the latest version of the launch template.

                                If the value is $Default , Amazon EC2 uses the default version of the launch template.

                        - **Overrides** _(dict) --_

                            Any parameters that you specify override the same parameters in the launch template.

                            - **InstanceType** _(string) --_

                                The instance type.

                            - **MaxPrice** _(string) --_

                                The maximum price per unit hour that you are willing to pay for a Spot Instance.

                            - **SubnetId** _(string) --_

                                The ID of the subnet in which to launch the instances.

                            - **AvailabilityZone** _(string) --_

                                The Availability Zone in which to launch the instances.

                            - **WeightedCapacity** _(float) --_

                                The number of units provided by the specified instance type.

                            - **Priority** _(float) --_

                                The priority for the launch template override. If **AllocationStrategy** is set to prioritized , EC2 Fleet uses priority to determine which launch template override to use first in fulfilling On-Demand capacity. The highest priority is launched first. Valid values are whole numbers starting at 0 . The lower the number, the higher the priority. If no number is set, the override has the lowest priority.

                            - **Placement** _(dict) --_

                                The location where the instance launched, if applicable.

                                - **GroupName** _(string) --_

                                    The name of the placement group that the instance is in.

                    - **Lifecycle** _(string) --_

                        Indicates if the instance that could not be launched was a Spot Instance or On-Demand Instance.

                    - **ErrorCode** _(string) --_

                        The error code that indicates why the instance could not be launched. For more information about error codes, see [Error Codes](https://docs.aws.amazon.com/AWSEC2/latest/APIReference/errors-overview.html.html) .

                    - **ErrorMessage** _(string) --_

                        The error message that describes why the instance could not be launched. For more information about error messages, see [Error Codes](https://docs.aws.amazon.com/AWSEC2/latest/APIReference/errors-overview.html.html) .

            - **Instances** _(list) --_

                Information about the instances that were launched by the fleet. Valid only when **Type** is set to instant .

                - _(dict) --_

                    Describes the instances that were launched by the fleet.

                    - **LaunchTemplateAndOverrides** _(dict) --_

                        The launch templates and overrides that were used for launching the instances. The values that you specify in the Overrides replace the values in the launch template.

                        - **LaunchTemplateSpecification** _(dict) --_

                            The launch template.

                            - **LaunchTemplateId** _(string) --_

                                The ID of the launch template. If you specify the template ID, you can't specify the template name.

                            - **LaunchTemplateName** _(string) --_

                                The name of the launch template. If you specify the template name, you can't specify the template ID.

                            - **Version** _(string) --_

                                The launch template version number, $Latest , or $Default . You must specify a value, otherwise the request fails.

                                If the value is $Latest , Amazon EC2 uses the latest version of the launch template.

                                If the value is $Default , Amazon EC2 uses the default version of the launch template.

                        - **Overrides** _(dict) --_

                            Any parameters that you specify override the same parameters in the launch template.

                            - **InstanceType** _(string) --_

                                The instance type.

                            - **MaxPrice** _(string) --_

                                The maximum price per unit hour that you are willing to pay for a Spot Instance.

                            - **SubnetId** _(string) --_

                                The ID of the subnet in which to launch the instances.

                            - **AvailabilityZone** _(string) --_

                                The Availability Zone in which to launch the instances.

                            - **WeightedCapacity** _(float) --_

                                The number of units provided by the specified instance type.

                            - **Priority** _(float) --_

                                The priority for the launch template override. If **AllocationStrategy** is set to prioritized , EC2 Fleet uses priority to determine which launch template override to use first in fulfilling On-Demand capacity. The highest priority is launched first. Valid values are whole numbers starting at 0 . The lower the number, the higher the priority. If no number is set, the override has the lowest priority.

                            - **Placement** _(dict) --_

                                The location where the instance launched, if applicable.

                                - **GroupName** _(string) --_

                                    The name of the placement group that the instance is in.

                    - **Lifecycle** _(string) --_

                        Indicates if the instance that was launched is a Spot Instance or On-Demand Instance.

                    - **InstanceIds** _(list) --_

                        The IDs of the instances.

                        - _(string) --_
                    - **InstanceType** _(string) --_

                        The instance type.

                    - **Platform** _(string) --_

                        The value is Windows for Windows instances. Otherwise, the value is blank.


_class_ EC2.Paginator.DescribeFlowLogs

paginator = client.get_paginator('describe_flow_logs')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_flow_logs()](#EC2.Client.describe_flow_logs "EC2.Client.describe_flow_logs").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeFlowLogs)

**Request Syntax**

response_iterator = paginator.paginate(
    DryRun=True|False,
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    FlowLogIds=[
        'string',
    ],
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **Filters** (_list_) --

    One or more filters.

    - deliver-log-status - The status of the logs delivery (SUCCESS | FAILED ).
    - log-destination-type - The type of destination to which the flow log publishes data. Possible destination types include cloud-watch-logs and s3 .
    - flow-log-id - The ID of the flow log.
    - log-group-name - The name of the log group.
    - resource-id - The ID of the VPC, subnet, or network interface.
    - traffic-type - The type of traffic (ACCEPT | REJECT | ALL ).
    - tag :<key> - The key/value combination of a tag assigned to the resource. Use the tag key in the filter name and the tag value as the filter value. For example, to find all resources that have a tag with the key Owner and the value TeamA , specify tag:Owner for the filter name and TeamA for the filter value.
    - tag-key - The key of a tag assigned to the resource. Use this filter to find all resources assigned a tag with a specific key, regardless of the tag value.

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **FlowLogIds** (_list_) --

    One or more flow log IDs.

    Constraint: Maximum of 1000 flow log IDs.

    - _(string) --_
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'FlowLogs': [
        {
            'CreationTime': datetime(2015, 1, 1),
            'DeliverLogsErrorMessage': 'string',
            'DeliverLogsPermissionArn': 'string',
            'DeliverLogsStatus': 'string',
            'FlowLogId': 'string',
            'FlowLogStatus': 'string',
            'LogGroupName': 'string',
            'ResourceId': 'string',
            'TrafficType': 'ACCEPT'|'REJECT'|'ALL',
            'LogDestinationType': 'cloud-watch-logs'|'s3',
            'LogDestination': 'string',
            'LogFormat': 'string',
            'Tags': [
                {
                    'Key': 'string',
                    'Value': 'string'
                },
            ],
            'MaxAggregationInterval': 123
        },
    ],

}

**Response Structure**

- _(dict) --_

    - **FlowLogs** _(list) --_

        Information about the flow logs.

        - _(dict) --_

            Describes a flow log.

            - **CreationTime** _(datetime) --_

                The date and time the flow log was created.

            - **DeliverLogsErrorMessage** _(string) --_

                Information about the error that occurred. Rate limited indicates that CloudWatch Logs throttling has been applied for one or more network interfaces, or that you've reached the limit on the number of log groups that you can create. Access error indicates that the IAM role associated with the flow log does not have sufficient permissions to publish to CloudWatch Logs. Unknown error indicates an internal error.

            - **DeliverLogsPermissionArn** _(string) --_

                The ARN of the IAM role that posts logs to CloudWatch Logs.

            - **DeliverLogsStatus** _(string) --_

                The status of the logs delivery (SUCCESS | FAILED ).

            - **FlowLogId** _(string) --_

                The flow log ID.

            - **FlowLogStatus** _(string) --_

                The status of the flow log (ACTIVE ).

            - **LogGroupName** _(string) --_

                The name of the flow log group.

            - **ResourceId** _(string) --_

                The ID of the resource on which the flow log was created.

            - **TrafficType** _(string) --_

                The type of traffic captured for the flow log.

            - **LogDestinationType** _(string) --_

                Specifies the type of destination to which the flow log data is published. Flow log data can be published to CloudWatch Logs or Amazon S3.

            - **LogDestination** _(string) --_

                Specifies the destination to which the flow log data is published. Flow log data can be published to an CloudWatch Logs log group or an Amazon S3 bucket. If the flow log publishes to CloudWatch Logs, this element indicates the Amazon Resource Name (ARN) of the CloudWatch Logs log group to which the data is published. If the flow log publishes to Amazon S3, this element indicates the ARN of the Amazon S3 bucket to which the data is published.

            - **LogFormat** _(string) --_

                The format of the flow log record.

            - **Tags** _(list) --_

                The tags for the flow log.

                - _(dict) --_

                    Describes a tag.

                    - **Key** _(string) --_

                        The key of the tag.

                        Constraints: Tag keys are case-sensitive and accept a maximum of 127 Unicode characters. May not begin with aws: .

                    - **Value** _(string) --_

                        The value of the tag.

                        Constraints: Tag values are case-sensitive and accept a maximum of 255 Unicode characters.

            - **MaxAggregationInterval** _(integer) --_

                The maximum interval of time, in seconds, during which a flow of packets is captured and aggregated into a flow log record.

                When a network interface is attached to a [Nitro-based instance](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/instance-types.html#ec2-nitro-instances) , the aggregation interval is always 60 seconds (1 minute) or less, regardless of the specified value.

                Valid Values: 60 | 600


_class_ EC2.Paginator.DescribeFpgaImages

paginator = client.get_paginator('describe_fpga_images')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_fpga_images()](#EC2.Client.describe_fpga_images "EC2.Client.describe_fpga_images").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeFpgaImages)

**Request Syntax**

response_iterator = paginator.paginate(
    DryRun=True|False,
    FpgaImageIds=[
        'string',
    ],
    Owners=[
        'string',
    ],
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **FpgaImageIds** (_list_) --

    The AFI IDs.

    - _(string) --_
- **Owners** (_list_) --

    Filters the AFI by owner. Specify an AWS account ID, self (owner is the sender of the request), or an AWS owner alias (valid values are amazon | aws-marketplace ).

    - _(string) --_
- **Filters** (_list_) --

    The filters.

    - create-time - The creation time of the AFI.
    - fpga-image-id - The FPGA image identifier (AFI ID).
    - fpga-image-global-id - The global FPGA image identifier (AGFI ID).
    - name - The name of the AFI.
    - owner-id - The AWS account ID of the AFI owner.
    - product-code - The product code.
    - shell-version - The version of the AWS Shell that was used to create the bitstream.
    - state - The state of the AFI (pending | failed | available | unavailable ).
    - tag :<key> - The key/value combination of a tag assigned to the resource. Use the tag key in the filter name and the tag value as the filter value. For example, to find all resources that have a tag with the key Owner and the value TeamA , specify tag:Owner for the filter name and TeamA for the filter value.
    - tag-key - The key of a tag assigned to the resource. Use this filter to find all resources assigned a tag with a specific key, regardless of the tag value.
    - update-time - The time of the most recent update.

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'FpgaImages': [
        {
            'FpgaImageId': 'string',
            'FpgaImageGlobalId': 'string',
            'Name': 'string',
            'Description': 'string',
            'ShellVersion': 'string',
            'PciId': {
                'DeviceId': 'string',
                'VendorId': 'string',
                'SubsystemId': 'string',
                'SubsystemVendorId': 'string'
            },
            'State': {
                'Code': 'pending'|'failed'|'available'|'unavailable',
                'Message': 'string'
            },
            'CreateTime': datetime(2015, 1, 1),
            'UpdateTime': datetime(2015, 1, 1),
            'OwnerId': 'string',
            'OwnerAlias': 'string',
            'ProductCodes': [
                {
                    'ProductCodeId': 'string',
                    'ProductCodeType': 'devpay'|'marketplace'
                },
            ],
            'Tags': [
                {
                    'Key': 'string',
                    'Value': 'string'
                },
            ],
            'Public': True|False,
            'DataRetentionSupport': True|False
        },
    ],

}

**Response Structure**

- _(dict) --_

    - **FpgaImages** _(list) --_

        Information about the FPGA images.

        - _(dict) --_

            Describes an Amazon FPGA image (AFI).

            - **FpgaImageId** _(string) --_

                The FPGA image identifier (AFI ID).

            - **FpgaImageGlobalId** _(string) --_

                The global FPGA image identifier (AGFI ID).

            - **Name** _(string) --_

                The name of the AFI.

            - **Description** _(string) --_

                The description of the AFI.

            - **ShellVersion** _(string) --_

                The version of the AWS Shell that was used to create the bitstream.

            - **PciId** _(dict) --_

                Information about the PCI bus.

                - **DeviceId** _(string) --_

                    The ID of the device.

                - **VendorId** _(string) --_

                    The ID of the vendor.

                - **SubsystemId** _(string) --_

                    The ID of the subsystem.

                - **SubsystemVendorId** _(string) --_

                    The ID of the vendor for the subsystem.

            - **State** _(dict) --_

                Information about the state of the AFI.

                - **Code** _(string) --_

                    The state. The following are the possible values:

                    - pending - AFI bitstream generation is in progress.
                    - available - The AFI is available for use.
                    - failed - AFI bitstream generation failed.
                    - unavailable - The AFI is no longer available for use.
                - **Message** _(string) --_

                    If the state is failed , this is the error message.

            - **CreateTime** _(datetime) --_

                The date and time the AFI was created.

            - **UpdateTime** _(datetime) --_

                The time of the most recent update to the AFI.

            - **OwnerId** _(string) --_

                The AWS account ID of the AFI owner.

            - **OwnerAlias** _(string) --_

                The alias of the AFI owner. Possible values include self , amazon , and aws-marketplace .

            - **ProductCodes** _(list) --_

                The product codes for the AFI.

                - _(dict) --_

                    Describes a product code.

                    - **ProductCodeId** _(string) --_

                        The product code.

                    - **ProductCodeType** _(string) --_

                        The type of product code.

            - **Tags** _(list) --_

                Any tags assigned to the AFI.

                - _(dict) --_

                    Describes a tag.

                    - **Key** _(string) --_

                        The key of the tag.

                        Constraints: Tag keys are case-sensitive and accept a maximum of 127 Unicode characters. May not begin with aws: .

                    - **Value** _(string) --_

                        The value of the tag.

                        Constraints: Tag values are case-sensitive and accept a maximum of 255 Unicode characters.

            - **Public** _(boolean) --_

                Indicates whether the AFI is public.

            - **DataRetentionSupport** _(boolean) --_

                Indicates whether data retention support is enabled for the AFI.


_class_ EC2.Paginator.DescribeHostReservationOfferings

paginator = client.get_paginator('describe_host_reservation_offerings')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_host_reservation_offerings()](#EC2.Client.describe_host_reservation_offerings "EC2.Client.describe_host_reservation_offerings").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeHostReservationOfferings)

**Request Syntax**

response_iterator = paginator.paginate(
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    MaxDuration=123,
    MinDuration=123,
    OfferingId='string',
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **Filters** (_list_) --

    The filters.

    - instance-family - The instance family of the offering (for example, m4 ).
    - payment-option - The payment option (NoUpfront | PartialUpfront | AllUpfront ).

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **MaxDuration** (_integer_) -- This is the maximum duration of the reservation to purchase, specified in seconds. Reservations are available in one-year and three-year terms. The number of seconds specified must be the number of seconds in a year (365x24x60x60) times one of the supported durations (1 or 3). For example, specify 94608000 for three years.
- **MinDuration** (_integer_) -- This is the minimum duration of the reservation you'd like to purchase, specified in seconds. Reservations are available in one-year and three-year terms. The number of seconds specified must be the number of seconds in a year (365x24x60x60) times one of the supported durations (1 or 3). For example, specify 31536000 for one year.
- **OfferingId** (_string_) -- The ID of the reservation offering.
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'OfferingSet': [
        {
            'CurrencyCode': 'USD',
            'Duration': 123,
            'HourlyPrice': 'string',
            'InstanceFamily': 'string',
            'OfferingId': 'string',
            'PaymentOption': 'AllUpfront'|'PartialUpfront'|'NoUpfront',
            'UpfrontPrice': 'string'
        },
    ]
}

**Response Structure**

- _(dict) --_

    - **OfferingSet** _(list) --_

        Information about the offerings.

        - _(dict) --_

            Details about the Dedicated Host Reservation offering.

            - **CurrencyCode** _(string) --_

                The currency of the offering.

            - **Duration** _(integer) --_

                The duration of the offering (in seconds).

            - **HourlyPrice** _(string) --_

                The hourly price of the offering.

            - **InstanceFamily** _(string) --_

                The instance family of the offering.

            - **OfferingId** _(string) --_

                The ID of the offering.

            - **PaymentOption** _(string) --_

                The available payment option.

            - **UpfrontPrice** _(string) --_

                The upfront price of the offering. Does not apply to No Upfront offerings.


_class_ EC2.Paginator.DescribeHostReservations

paginator = client.get_paginator('describe_host_reservations')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_host_reservations()](#EC2.Client.describe_host_reservations "EC2.Client.describe_host_reservations").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeHostReservations)

**Request Syntax**

response_iterator = paginator.paginate(
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    HostReservationIdSet=[
        'string',
    ],
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **Filters** (_list_) --

    The filters.

    - instance-family - The instance family (for example, m4 ).
    - payment-option - The payment option (NoUpfront | PartialUpfront | AllUpfront ).
    - state - The state of the reservation (payment-pending | payment-failed | active | retired ).
    - tag :<key> - The key/value combination of a tag assigned to the resource. Use the tag key in the filter name and the tag value as the filter value. For example, to find all resources that have a tag with the key Owner and the value TeamA , specify tag:Owner for the filter name and TeamA for the filter value.
    - tag-key - The key of a tag assigned to the resource. Use this filter to find all resources assigned a tag with a specific key, regardless of the tag value.

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **HostReservationIdSet** (_list_) --

    The host reservation IDs.

    - _(string) --_
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'HostReservationSet': [
        {
            'Count': 123,
            'CurrencyCode': 'USD',
            'Duration': 123,
            'End': datetime(2015, 1, 1),
            'HostIdSet': [
                'string',
            ],
            'HostReservationId': 'string',
            'HourlyPrice': 'string',
            'InstanceFamily': 'string',
            'OfferingId': 'string',
            'PaymentOption': 'AllUpfront'|'PartialUpfront'|'NoUpfront',
            'Start': datetime(2015, 1, 1),
            'State': 'payment-pending'|'payment-failed'|'active'|'retired',
            'UpfrontPrice': 'string',
            'Tags': [
                {
                    'Key': 'string',
                    'Value': 'string'
                },
            ]
        },
    ],

}

**Response Structure**

- _(dict) --_

    - **HostReservationSet** _(list) --_

        Details about the reservation's configuration.

        - _(dict) --_

            Details about the Dedicated Host Reservation and associated Dedicated Hosts.

            - **Count** _(integer) --_

                The number of Dedicated Hosts the reservation is associated with.

            - **CurrencyCode** _(string) --_

                The currency in which the upfrontPrice and hourlyPrice amounts are specified. At this time, the only supported currency is USD .

            - **Duration** _(integer) --_

                The length of the reservation's term, specified in seconds. Can be 31536000 (1 year) | 94608000 (3 years) .

            - **End** _(datetime) --_

                The date and time that the reservation ends.

            - **HostIdSet** _(list) --_

                The IDs of the Dedicated Hosts associated with the reservation.

                - _(string) --_
            - **HostReservationId** _(string) --_

                The ID of the reservation that specifies the associated Dedicated Hosts.

            - **HourlyPrice** _(string) --_

                The hourly price of the reservation.

            - **InstanceFamily** _(string) --_

                The instance family of the Dedicated Host Reservation. The instance family on the Dedicated Host must be the same in order for it to benefit from the reservation.

            - **OfferingId** _(string) --_

                The ID of the reservation. This remains the same regardless of which Dedicated Hosts are associated with it.

            - **PaymentOption** _(string) --_

                The payment option selected for this reservation.

            - **Start** _(datetime) --_

                The date and time that the reservation started.

            - **State** _(string) --_

                The state of the reservation.

            - **UpfrontPrice** _(string) --_

                The upfront price of the reservation.

            - **Tags** _(list) --_

                Any tags assigned to the Dedicated Host Reservation.

                - _(dict) --_

                    Describes a tag.

                    - **Key** _(string) --_

                        The key of the tag.

                        Constraints: Tag keys are case-sensitive and accept a maximum of 127 Unicode characters. May not begin with aws: .

                    - **Value** _(string) --_

                        The value of the tag.

                        Constraints: Tag values are case-sensitive and accept a maximum of 255 Unicode characters.


_class_ EC2.Paginator.DescribeHosts

paginator = client.get_paginator('describe_hosts')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_hosts()](#EC2.Client.describe_hosts "EC2.Client.describe_hosts").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeHosts)

**Request Syntax**

response_iterator = paginator.paginate(
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    HostIds=[
        'string',
    ],
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **Filters** (_list_) --

    The filters.

    - auto-placement - Whether auto-placement is enabled or disabled (on | off ).
    - availability-zone - The Availability Zone of the host.
    - client-token - The idempotency token that you provided when you allocated the host.
    - host-reservation-id - The ID of the reservation assigned to this host.
    - instance-type - The instance type size that the Dedicated Host is configured to support.
    - state - The allocation state of the Dedicated Host (available | under-assessment | permanent-failure | released | released-permanent-failure ).
    - tag-key - The key of a tag assigned to the resource. Use this filter to find all resources assigned a tag with a specific key, regardless of the tag value.

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **HostIds** (_list_) --

    The IDs of the Dedicated Hosts. The IDs are used for targeted instance launches.

    - _(string) --_
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'Hosts': [
        {
            'AutoPlacement': 'on'|'off',
            'AvailabilityZone': 'string',
            'AvailableCapacity': {
                'AvailableInstanceCapacity': [
                    {
                        'AvailableCapacity': 123,
                        'InstanceType': 'string',
                        'TotalCapacity': 123
                    },
                ],
                'AvailableVCpus': 123
            },
            'ClientToken': 'string',
            'HostId': 'string',
            'HostProperties': {
                'Cores': 123,
                'InstanceType': 'string',
                'InstanceFamily': 'string',
                'Sockets': 123,
                'TotalVCpus': 123
            },
            'HostReservationId': 'string',
            'Instances': [
                {
                    'InstanceId': 'string',
                    'InstanceType': 'string',
                    'OwnerId': 'string'
                },
            ],
            'State': 'available'|'under-assessment'|'permanent-failure'|'released'|'released-permanent-failure'|'pending',
            'AllocationTime': datetime(2015, 1, 1),
            'ReleaseTime': datetime(2015, 1, 1),
            'Tags': [
                {
                    'Key': 'string',
                    'Value': 'string'
                },
            ],
            'HostRecovery': 'on'|'off',
            'AllowsMultipleInstanceTypes': 'on'|'off',
            'OwnerId': 'string',
            'AvailabilityZoneId': 'string',
            'MemberOfServiceLinkedResourceGroup': True|False
        },
    ],

}

**Response Structure**

- _(dict) --_

    - **Hosts** _(list) --_

        Information about the Dedicated Hosts.

        - _(dict) --_

            Describes the properties of the Dedicated Host.

            - **AutoPlacement** _(string) --_

                Whether auto-placement is on or off.

            - **AvailabilityZone** _(string) --_

                The Availability Zone of the Dedicated Host.

            - **AvailableCapacity** _(dict) --_

                Information about the instances running on the Dedicated Host.

                - **AvailableInstanceCapacity** _(list) --_

                    The number of instances that can be launched onto the Dedicated Host depending on the host's available capacity. For Dedicated Hosts that support multiple instance types, this parameter represents the number of instances for each instance size that is supported on the host.

                    - _(dict) --_

                        Information about the number of instances that can be launched onto the Dedicated Host.

                        - **AvailableCapacity** _(integer) --_

                            The number of instances that can be launched onto the Dedicated Host based on the host's available capacity.

                        - **InstanceType** _(string) --_

                            The instance type supported by the Dedicated Host.

                        - **TotalCapacity** _(integer) --_

                            The total number of instances that can be launched onto the Dedicated Host if there are no instances running on it.

                - **AvailableVCpus** _(integer) --_

                    The number of vCPUs available for launching instances onto the Dedicated Host.

            - **ClientToken** _(string) --_

                Unique, case-sensitive identifier that you provide to ensure the idempotency of the request. For more information, see [How to Ensure Idempotency](https://docs.aws.amazon.com/AWSEC2/latest/APIReference/Run_Instance_Idempotency.html) .

            - **HostId** _(string) --_

                The ID of the Dedicated Host.

            - **HostProperties** _(dict) --_

                The hardware specifications of the Dedicated Host.

                - **Cores** _(integer) --_

                    The number of cores on the Dedicated Host.

                - **InstanceType** _(string) --_

                    The instance type supported by the Dedicated Host. For example, m5.large . If the host supports multiple instance types, no **instanceType** is returned.

                - **InstanceFamily** _(string) --_

                    The instance family supported by the Dedicated Host. For example, m5 .

                - **Sockets** _(integer) --_

                    The number of sockets on the Dedicated Host.

                - **TotalVCpus** _(integer) --_

                    The total number of vCPUs on the Dedicated Host.

            - **HostReservationId** _(string) --_

                The reservation ID of the Dedicated Host. This returns a null response if the Dedicated Host doesn't have an associated reservation.

            - **Instances** _(list) --_

                The IDs and instance type that are currently running on the Dedicated Host.

                - _(dict) --_

                    Describes an instance running on a Dedicated Host.

                    - **InstanceId** _(string) --_

                        The ID of instance that is running on the Dedicated Host.

                    - **InstanceType** _(string) --_

                        The instance type (for example, m3.medium ) of the running instance.

                    - **OwnerId** _(string) --_

                        The ID of the AWS account that owns the instance.

            - **State** _(string) --_

                The Dedicated Host's state.

            - **AllocationTime** _(datetime) --_

                The time that the Dedicated Host was allocated.

            - **ReleaseTime** _(datetime) --_

                The time that the Dedicated Host was released.

            - **Tags** _(list) --_

                Any tags assigned to the Dedicated Host.

                - _(dict) --_

                    Describes a tag.

                    - **Key** _(string) --_

                        The key of the tag.

                        Constraints: Tag keys are case-sensitive and accept a maximum of 127 Unicode characters. May not begin with aws: .

                    - **Value** _(string) --_

                        The value of the tag.

                        Constraints: Tag values are case-sensitive and accept a maximum of 255 Unicode characters.

            - **HostRecovery** _(string) --_

                Indicates whether host recovery is enabled or disabled for the Dedicated Host.

            - **AllowsMultipleInstanceTypes** _(string) --_

                Indicates whether the Dedicated Host supports multiple instance types of the same instance family, or a specific instance type only. one indicates that the Dedicated Host supports multiple instance types in the instance family. off indicates that the Dedicated Host supports a single instance type only.

            - **OwnerId** _(string) --_

                The ID of the AWS account that owns the Dedicated Host.

            - **AvailabilityZoneId** _(string) --_

                The ID of the Availability Zone in which the Dedicated Host is allocated.

            - **MemberOfServiceLinkedResourceGroup** _(boolean) --_

                Indicates whether the Dedicated Host is in a host resource group. If **memberOfServiceLinkedResourceGroup** is true , the host is in a host resource group; otherwise, it is not.


_class_ EC2.Paginator.DescribeIamInstanceProfileAssociations

paginator = client.get_paginator('describe_iam_instance_profile_associations')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_iam_instance_profile_associations()](#EC2.Client.describe_iam_instance_profile_associations "EC2.Client.describe_iam_instance_profile_associations").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeIamInstanceProfileAssociations)

**Request Syntax**

response_iterator = paginator.paginate(
    AssociationIds=[
        'string',
    ],
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **AssociationIds** (_list_) --

    The IAM instance profile associations.

    - _(string) --_
- **Filters** (_list_) --

    The filters.

    - instance-id - The ID of the instance.
    - state - The state of the association (associating | associated | disassociating ).

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'IamInstanceProfileAssociations': [
        {
            'AssociationId': 'string',
            'InstanceId': 'string',
            'IamInstanceProfile': {
                'Arn': 'string',
                'Id': 'string'
            },
            'State': 'associating'|'associated'|'disassociating'|'disassociated',
            'Timestamp': datetime(2015, 1, 1)
        },
    ],

}

**Response Structure**

- _(dict) --_

    - **IamInstanceProfileAssociations** _(list) --_

        Information about the IAM instance profile associations.

        - _(dict) --_

            Describes an association between an IAM instance profile and an instance.

            - **AssociationId** _(string) --_

                The ID of the association.

            - **InstanceId** _(string) --_

                The ID of the instance.

            - **IamInstanceProfile** _(dict) --_

                The IAM instance profile.

                - **Arn** _(string) --_

                    The Amazon Resource Name (ARN) of the instance profile.

                - **Id** _(string) --_

                    The ID of the instance profile.

            - **State** _(string) --_

                The state of the association.

            - **Timestamp** _(datetime) --_

                The time the IAM instance profile was associated with the instance.


_class_ EC2.Paginator.DescribeImportImageTasks

paginator = client.get_paginator('describe_import_image_tasks')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_import_image_tasks()](#EC2.Client.describe_import_image_tasks "EC2.Client.describe_import_image_tasks").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeImportImageTasks)

**Request Syntax**

response_iterator = paginator.paginate(
    DryRun=True|False,
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    ImportTaskIds=[
        'string',
    ],
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **Filters** (_list_) --

    Filter tasks using the task-state filter and one of the following values: active , completed , deleting , or deleted .

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **ImportTaskIds** (_list_) --

    The IDs of the import image tasks.

    - _(string) --_
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'ImportImageTasks': [
        {
            'Architecture': 'string',
            'Description': 'string',
            'Encrypted': True|False,
            'Hypervisor': 'string',
            'ImageId': 'string',
            'ImportTaskId': 'string',
            'KmsKeyId': 'string',
            'LicenseType': 'string',
            'Platform': 'string',
            'Progress': 'string',
            'SnapshotDetails': [
                {
                    'Description': 'string',
                    'DeviceName': 'string',
                    'DiskImageSize': 123.0,
                    'Format': 'string',
                    'Progress': 'string',
                    'SnapshotId': 'string',
                    'Status': 'string',
                    'StatusMessage': 'string',
                    'Url': 'string',
                    'UserBucket': {
                        'S3Bucket': 'string',
                        'S3Key': 'string'
                    }
                },
            ],
            'Status': 'string',
            'StatusMessage': 'string',
            'Tags': [
                {
                    'Key': 'string',
                    'Value': 'string'
                },
            ],
            'LicenseSpecifications': [
                {
                    'LicenseConfigurationArn': 'string'
                },
            ]
        },
    ],

}

**Response Structure**

- _(dict) --_

    - **ImportImageTasks** _(list) --_

        A list of zero or more import image tasks that are currently active or were completed or canceled in the previous 7 days.

        - _(dict) --_

            Describes an import image task.

            - **Architecture** _(string) --_

                The architecture of the virtual machine.

                Valid values: i386 | x86_64 | arm64

            - **Description** _(string) --_

                A description of the import task.

            - **Encrypted** _(boolean) --_

                Indicates whether the image is encrypted.

            - **Hypervisor** _(string) --_

                The target hypervisor for the import task.

                Valid values: xen

            - **ImageId** _(string) --_

                The ID of the Amazon Machine Image (AMI) of the imported virtual machine.

            - **ImportTaskId** _(string) --_

                The ID of the import image task.

            - **KmsKeyId** _(string) --_

                The identifier for the AWS Key Management Service (AWS KMS) customer master key (CMK) that was used to create the encrypted image.

            - **LicenseType** _(string) --_

                The license type of the virtual machine.

            - **Platform** _(string) --_

                The description string for the import image task.

            - **Progress** _(string) --_

                The percentage of progress of the import image task.

            - **SnapshotDetails** _(list) --_

                Information about the snapshots.

                - _(dict) --_

                    Describes the snapshot created from the imported disk.

                    - **Description** _(string) --_

                        A description for the snapshot.

                    - **DeviceName** _(string) --_

                        The block device mapping for the snapshot.

                    - **DiskImageSize** _(float) --_

                        The size of the disk in the snapshot, in GiB.

                    - **Format** _(string) --_

                        The format of the disk image from which the snapshot is created.

                    - **Progress** _(string) --_

                        The percentage of progress for the task.

                    - **SnapshotId** _(string) --_

                        The snapshot ID of the disk being imported.

                    - **Status** _(string) --_

                        A brief status of the snapshot creation.

                    - **StatusMessage** _(string) --_

                        A detailed status message for the snapshot creation.

                    - **Url** _(string) --_

                        The URL used to access the disk image.

                    - **UserBucket** _(dict) --_

                        The Amazon S3 bucket for the disk image.

                        - **S3Bucket** _(string) --_

                            The Amazon S3 bucket from which the disk image was created.

                        - **S3Key** _(string) --_

                            The file name of the disk image.

            - **Status** _(string) --_

                A brief status for the import image task.

            - **StatusMessage** _(string) --_

                A descriptive status message for the import image task.

            - **Tags** _(list) --_

                The tags for the import image task.

                - _(dict) --_

                    Describes a tag.

                    - **Key** _(string) --_

                        The key of the tag.

                        Constraints: Tag keys are case-sensitive and accept a maximum of 127 Unicode characters. May not begin with aws: .

                    - **Value** _(string) --_

                        The value of the tag.

                        Constraints: Tag values are case-sensitive and accept a maximum of 255 Unicode characters.

            - **LicenseSpecifications** _(list) --_

                The ARNs of the license configurations that are associated with the import image task.

                - _(dict) --_

                    The response information for license configurations.

                    - **LicenseConfigurationArn** _(string) --_

                        The ARN of a license configuration.


_class_ EC2.Paginator.DescribeImportSnapshotTasks

paginator = client.get_paginator('describe_import_snapshot_tasks')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_import_snapshot_tasks()](#EC2.Client.describe_import_snapshot_tasks "EC2.Client.describe_import_snapshot_tasks").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeImportSnapshotTasks)

**Request Syntax**

response_iterator = paginator.paginate(
    DryRun=True|False,
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    ImportTaskIds=[
        'string',
    ],
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **Filters** (_list_) --

    The filters.

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **ImportTaskIds** (_list_) --

    A list of import snapshot task IDs.

    - _(string) --_
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'ImportSnapshotTasks': [
        {
            'Description': 'string',
            'ImportTaskId': 'string',
            'SnapshotTaskDetail': {
                'Description': 'string',
                'DiskImageSize': 123.0,
                'Encrypted': True|False,
                'Format': 'string',
                'KmsKeyId': 'string',
                'Progress': 'string',
                'SnapshotId': 'string',
                'Status': 'string',
                'StatusMessage': 'string',
                'Url': 'string',
                'UserBucket': {
                    'S3Bucket': 'string',
                    'S3Key': 'string'
                }
            },
            'Tags': [
                {
                    'Key': 'string',
                    'Value': 'string'
                },
            ]
        },
    ],

}

**Response Structure**

- _(dict) --_

    - **ImportSnapshotTasks** _(list) --_

        A list of zero or more import snapshot tasks that are currently active or were completed or canceled in the previous 7 days.

        - _(dict) --_

            Describes an import snapshot task.

            - **Description** _(string) --_

                A description of the import snapshot task.

            - **ImportTaskId** _(string) --_

                The ID of the import snapshot task.

            - **SnapshotTaskDetail** _(dict) --_

                Describes an import snapshot task.

                - **Description** _(string) --_

                    The description of the snapshot.

                - **DiskImageSize** _(float) --_

                    The size of the disk in the snapshot, in GiB.

                - **Encrypted** _(boolean) --_

                    Indicates whether the snapshot is encrypted.

                - **Format** _(string) --_

                    The format of the disk image from which the snapshot is created.

                - **KmsKeyId** _(string) --_

                    The identifier for the AWS Key Management Service (AWS KMS) customer master key (CMK) that was used to create the encrypted snapshot.

                - **Progress** _(string) --_

                    The percentage of completion for the import snapshot task.

                - **SnapshotId** _(string) --_

                    The snapshot ID of the disk being imported.

                - **Status** _(string) --_

                    A brief status for the import snapshot task.

                - **StatusMessage** _(string) --_

                    A detailed status message for the import snapshot task.

                - **Url** _(string) --_

                    The URL of the disk image from which the snapshot is created.

                - **UserBucket** _(dict) --_

                    The Amazon S3 bucket for the disk image.

                    - **S3Bucket** _(string) --_

                        The Amazon S3 bucket from which the disk image was created.

                    - **S3Key** _(string) --_

                        The file name of the disk image.

            - **Tags** _(list) --_

                The tags for the import snapshot task.

                - _(dict) --_

                    Describes a tag.

                    - **Key** _(string) --_

                        The key of the tag.

                        Constraints: Tag keys are case-sensitive and accept a maximum of 127 Unicode characters. May not begin with aws: .

                    - **Value** _(string) --_

                        The value of the tag.

                        Constraints: Tag values are case-sensitive and accept a maximum of 255 Unicode characters.


_class_ EC2.Paginator.DescribeInstanceCreditSpecifications

paginator = client.get_paginator('describe_instance_credit_specifications')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_instance_credit_specifications()](#EC2.Client.describe_instance_credit_specifications "EC2.Client.describe_instance_credit_specifications").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeInstanceCreditSpecifications)

**Request Syntax**

response_iterator = paginator.paginate(
    DryRun=True|False,
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    InstanceIds=[
        'string',
    ],
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **Filters** (_list_) --

    The filters.

    - instance-id - The ID of the instance.

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **InstanceIds** (_list_) --

    The instance IDs.

    Default: Describes all your instances.

    Constraints: Maximum 1000 explicitly specified instance IDs.

    - _(string) --_
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'InstanceCreditSpecifications': [
        {
            'InstanceId': 'string',
            'CpuCredits': 'string'
        },
    ],

}

**Response Structure**

- _(dict) --_

    - **InstanceCreditSpecifications** _(list) --_

        Information about the credit option for CPU usage of an instance.

        - _(dict) --_

            Describes the credit option for CPU usage of a burstable performance instance.

            - **InstanceId** _(string) --_

                The ID of the instance.

            - **CpuCredits** _(string) --_

                The credit option for CPU usage of the instance. Valid values are standard and unlimited .


_class_ EC2.Paginator.DescribeInstanceStatus

paginator = client.get_paginator('describe_instance_status')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_instance_status()](#EC2.Client.describe_instance_status "EC2.Client.describe_instance_status").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeInstanceStatus)

**Request Syntax**

response_iterator = paginator.paginate(
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    InstanceIds=[
        'string',
    ],
    DryRun=True|False,
    IncludeAllInstances=True|False,
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **Filters** (_list_) --

    The filters.

    - availability-zone - The Availability Zone of the instance.
    - event.code - The code for the scheduled event (instance-reboot | system-reboot | system-maintenance | instance-retirement | instance-stop ).
    - event.description - A description of the event.
    - event.instance-event-id - The ID of the event whose date and time you are modifying.
    - event.not-after - The latest end time for the scheduled event (for example, 2014-09-15T17:15:20.000Z ).
    - event.not-before - The earliest start time for the scheduled event (for example, 2014-09-15T17:15:20.000Z ).
    - event.not-before-deadline - The deadline for starting the event (for example, 2014-09-15T17:15:20.000Z ).
    - instance-state-code - The code for the instance state, as a 16-bit unsigned integer. The high byte is used for internal purposes and should be ignored. The low byte is set based on the state represented. The valid values are 0 (pending), 16 (running), 32 (shutting-down), 48 (terminated), 64 (stopping), and 80 (stopped).
    - instance-state-name - The state of the instance (pending | running | shutting-down | terminated | stopping | stopped ).
    - instance-status.reachability - Filters on instance status where the name is reachability (passed | failed | initializing | insufficient-data ).
    - instance-status.status - The status of the instance (ok | impaired | initializing | insufficient-data | not-applicable ).
    - system-status.reachability - Filters on system status where the name is reachability (passed | failed | initializing | insufficient-data ).
    - system-status.status - The system status of the instance (ok | impaired | initializing | insufficient-data | not-applicable ).

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **InstanceIds** (_list_) --

    The instance IDs.

    Default: Describes all your instances.

    Constraints: Maximum 100 explicitly specified instance IDs.

    - _(string) --_
- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **IncludeAllInstances** (_boolean_) --

    When true , includes the health status for all instances. When false , includes the health status for running instances only.

    Default: false

- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'InstanceStatuses': [
        {
            'AvailabilityZone': 'string',
            'OutpostArn': 'string',
            'Events': [
                {
                    'InstanceEventId': 'string',
                    'Code': 'instance-reboot'|'system-reboot'|'system-maintenance'|'instance-retirement'|'instance-stop',
                    'Description': 'string',
                    'NotAfter': datetime(2015, 1, 1),
                    'NotBefore': datetime(2015, 1, 1),
                    'NotBeforeDeadline': datetime(2015, 1, 1)
                },
            ],
            'InstanceId': 'string',
            'InstanceState': {
                'Code': 123,
                'Name': 'pending'|'running'|'shutting-down'|'terminated'|'stopping'|'stopped'
            },
            'InstanceStatus': {
                'Details': [
                    {
                        'ImpairedSince': datetime(2015, 1, 1),
                        'Name': 'reachability',
                        'Status': 'passed'|'failed'|'insufficient-data'|'initializing'
                    },
                ],
                'Status': 'ok'|'impaired'|'insufficient-data'|'not-applicable'|'initializing'
            },
            'SystemStatus': {
                'Details': [
                    {
                        'ImpairedSince': datetime(2015, 1, 1),
                        'Name': 'reachability',
                        'Status': 'passed'|'failed'|'insufficient-data'|'initializing'
                    },
                ],
                'Status': 'ok'|'impaired'|'insufficient-data'|'not-applicable'|'initializing'
            }
        },
    ],

}

**Response Structure**

- _(dict) --_

    - **InstanceStatuses** _(list) --_

        Information about the status of the instances.

        - _(dict) --_

            Describes the status of an instance.

            - **AvailabilityZone** _(string) --_

                The Availability Zone of the instance.

            - **OutpostArn** _(string) --_

                The Amazon Resource Name (ARN) of the Outpost.

            - **Events** _(list) --_

                Any scheduled events associated with the instance.

                - _(dict) --_

                    Describes a scheduled event for an instance.

                    - **InstanceEventId** _(string) --_

                        The ID of the event.

                    - **Code** _(string) --_

                        The event code.

                    - **Description** _(string) --_

                        A description of the event.

                        After a scheduled event is completed, it can still be described for up to a week. If the event has been completed, this description starts with the following text: [Completed].

                    - **NotAfter** _(datetime) --_

                        The latest scheduled end time for the event.

                    - **NotBefore** _(datetime) --_

                        The earliest scheduled start time for the event.

                    - **NotBeforeDeadline** _(datetime) --_

                        The deadline for starting the event.

            - **InstanceId** _(string) --_

                The ID of the instance.

            - **InstanceState** _(dict) --_

                The intended state of the instance. DescribeInstanceStatus requires that an instance be in the running state.

                - **Code** _(integer) --_

                    The state of the instance as a 16-bit unsigned integer.

                    The high byte is all of the bits between 2^8 and (2^16)-1, which equals decimal values between 256 and 65,535. These numerical values are used for internal purposes and should be ignored.

                    The low byte is all of the bits between 2^0 and (2^8)-1, which equals decimal values between 0 and 255.

                    The valid values for instance-state-code will all be in the range of the low byte and they are:

                    - 0 : pending
                    - 16 : running
                    - 32 : shutting-down
                    - 48 : terminated
                    - 64 : stopping
                    - 80 : stopped

                    You can ignore the high byte value by zeroing out all of the bits above 2^8 or 256 in decimal.

                - **Name** _(string) --_

                    The current state of the instance.

            - **InstanceStatus** _(dict) --_

                Reports impaired functionality that stems from issues internal to the instance, such as impaired reachability.

                - **Details** _(list) --_

                    The system instance health or application instance health.

                    - _(dict) --_

                        Describes the instance status.

                        - **ImpairedSince** _(datetime) --_

                            The time when a status check failed. For an instance that was launched and impaired, this is the time when the instance was launched.

                        - **Name** _(string) --_

                            The type of instance status.

                        - **Status** _(string) --_

                            The status.

                - **Status** _(string) --_

                    The status.

            - **SystemStatus** _(dict) --_

                Reports impaired functionality that stems from issues related to the systems that support an instance, such as hardware failures and network connectivity problems.

                - **Details** _(list) --_

                    The system instance health or application instance health.

                    - _(dict) --_

                        Describes the instance status.

                        - **ImpairedSince** _(datetime) --_

                            The time when a status check failed. For an instance that was launched and impaired, this is the time when the instance was launched.

                        - **Name** _(string) --_

                            The type of instance status.

                        - **Status** _(string) --_

                            The status.

                - **Status** _(string) --_

                    The status.


_class_ EC2.Paginator.DescribeInstanceTypeOfferings

paginator = client.get_paginator('describe_instance_type_offerings')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_instance_type_offerings()](#EC2.Client.describe_instance_type_offerings "EC2.Client.describe_instance_type_offerings").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeInstanceTypeOfferings)

**Request Syntax**

response_iterator = paginator.paginate(
    DryRun=True|False,
    LocationType='region'|'availability-zone'|'availability-zone-id',
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **LocationType** (_string_) -- The location type.
- **Filters** (_list_) --

    One or more filters. Filter names and values are case-sensitive.

    - location - This depends on the location type. For example, if the location type is region (default), the location is the Region code (for example, us-east-2 .)
    - instance-type - The instance type. For example, c5.2xlarge .

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'InstanceTypeOfferings': [
        {
            'InstanceType': 't1.micro'|'t2.nano'|'t2.micro'|'t2.small'|'t2.medium'|'t2.large'|'t2.xlarge'|'t2.2xlarge'|'t3.nano'|'t3.micro'|'t3.small'|'t3.medium'|'t3.large'|'t3.xlarge'|'t3.2xlarge'|'t3a.nano'|'t3a.micro'|'t3a.small'|'t3a.medium'|'t3a.large'|'t3a.xlarge'|'t3a.2xlarge'|'t4g.nano'|'t4g.micro'|'t4g.small'|'t4g.medium'|'t4g.large'|'t4g.xlarge'|'t4g.2xlarge'|'m1.small'|'m1.medium'|'m1.large'|'m1.xlarge'|'m3.medium'|'m3.large'|'m3.xlarge'|'m3.2xlarge'|'m4.large'|'m4.xlarge'|'m4.2xlarge'|'m4.4xlarge'|'m4.10xlarge'|'m4.16xlarge'|'m2.xlarge'|'m2.2xlarge'|'m2.4xlarge'|'cr1.8xlarge'|'r3.large'|'r3.xlarge'|'r3.2xlarge'|'r3.4xlarge'|'r3.8xlarge'|'r4.large'|'r4.xlarge'|'r4.2xlarge'|'r4.4xlarge'|'r4.8xlarge'|'r4.16xlarge'|'r5.large'|'r5.xlarge'|'r5.2xlarge'|'r5.4xlarge'|'r5.8xlarge'|'r5.12xlarge'|'r5.16xlarge'|'r5.24xlarge'|'r5.metal'|'r5a.large'|'r5a.xlarge'|'r5a.2xlarge'|'r5a.4xlarge'|'r5a.8xlarge'|'r5a.12xlarge'|'r5a.16xlarge'|'r5a.24xlarge'|'r5b.large'|'r5b.xlarge'|'r5b.2xlarge'|'r5b.4xlarge'|'r5b.8xlarge'|'r5b.12xlarge'|'r5b.16xlarge'|'r5b.24xlarge'|'r5b.metal'|'r5d.large'|'r5d.xlarge'|'r5d.2xlarge'|'r5d.4xlarge'|'r5d.8xlarge'|'r5d.12xlarge'|'r5d.16xlarge'|'r5d.24xlarge'|'r5d.metal'|'r5ad.large'|'r5ad.xlarge'|'r5ad.2xlarge'|'r5ad.4xlarge'|'r5ad.8xlarge'|'r5ad.12xlarge'|'r5ad.16xlarge'|'r5ad.24xlarge'|'r6g.metal'|'r6g.medium'|'r6g.large'|'r6g.xlarge'|'r6g.2xlarge'|'r6g.4xlarge'|'r6g.8xlarge'|'r6g.12xlarge'|'r6g.16xlarge'|'r6gd.metal'|'r6gd.medium'|'r6gd.large'|'r6gd.xlarge'|'r6gd.2xlarge'|'r6gd.4xlarge'|'r6gd.8xlarge'|'r6gd.12xlarge'|'r6gd.16xlarge'|'x1.16xlarge'|'x1.32xlarge'|'x1e.xlarge'|'x1e.2xlarge'|'x1e.4xlarge'|'x1e.8xlarge'|'x1e.16xlarge'|'x1e.32xlarge'|'i2.xlarge'|'i2.2xlarge'|'i2.4xlarge'|'i2.8xlarge'|'i3.large'|'i3.xlarge'|'i3.2xlarge'|'i3.4xlarge'|'i3.8xlarge'|'i3.16xlarge'|'i3.metal'|'i3en.large'|'i3en.xlarge'|'i3en.2xlarge'|'i3en.3xlarge'|'i3en.6xlarge'|'i3en.12xlarge'|'i3en.24xlarge'|'i3en.metal'|'hi1.4xlarge'|'hs1.8xlarge'|'c1.medium'|'c1.xlarge'|'c3.large'|'c3.xlarge'|'c3.2xlarge'|'c3.4xlarge'|'c3.8xlarge'|'c4.large'|'c4.xlarge'|'c4.2xlarge'|'c4.4xlarge'|'c4.8xlarge'|'c5.large'|'c5.xlarge'|'c5.2xlarge'|'c5.4xlarge'|'c5.9xlarge'|'c5.12xlarge'|'c5.18xlarge'|'c5.24xlarge'|'c5.metal'|'c5a.large'|'c5a.xlarge'|'c5a.2xlarge'|'c5a.4xlarge'|'c5a.8xlarge'|'c5a.12xlarge'|'c5a.16xlarge'|'c5a.24xlarge'|'c5ad.large'|'c5ad.xlarge'|'c5ad.2xlarge'|'c5ad.4xlarge'|'c5ad.8xlarge'|'c5ad.12xlarge'|'c5ad.16xlarge'|'c5ad.24xlarge'|'c5d.large'|'c5d.xlarge'|'c5d.2xlarge'|'c5d.4xlarge'|'c5d.9xlarge'|'c5d.12xlarge'|'c5d.18xlarge'|'c5d.24xlarge'|'c5d.metal'|'c5n.large'|'c5n.xlarge'|'c5n.2xlarge'|'c5n.4xlarge'|'c5n.9xlarge'|'c5n.18xlarge'|'c5n.metal'|'c6g.metal'|'c6g.medium'|'c6g.large'|'c6g.xlarge'|'c6g.2xlarge'|'c6g.4xlarge'|'c6g.8xlarge'|'c6g.12xlarge'|'c6g.16xlarge'|'c6gd.metal'|'c6gd.medium'|'c6gd.large'|'c6gd.xlarge'|'c6gd.2xlarge'|'c6gd.4xlarge'|'c6gd.8xlarge'|'c6gd.12xlarge'|'c6gd.16xlarge'|'c6gn.medium'|'c6gn.large'|'c6gn.xlarge'|'c6gn.2xlarge'|'c6gn.4xlarge'|'c6gn.8xlarge'|'c6gn.12xlarge'|'c6gn.16xlarge'|'cc1.4xlarge'|'cc2.8xlarge'|'g2.2xlarge'|'g2.8xlarge'|'g3.4xlarge'|'g3.8xlarge'|'g3.16xlarge'|'g3s.xlarge'|'g4ad.4xlarge'|'g4ad.8xlarge'|'g4ad.16xlarge'|'g4dn.xlarge'|'g4dn.2xlarge'|'g4dn.4xlarge'|'g4dn.8xlarge'|'g4dn.12xlarge'|'g4dn.16xlarge'|'g4dn.metal'|'cg1.4xlarge'|'p2.xlarge'|'p2.8xlarge'|'p2.16xlarge'|'p3.2xlarge'|'p3.8xlarge'|'p3.16xlarge'|'p3dn.24xlarge'|'p4d.24xlarge'|'d2.xlarge'|'d2.2xlarge'|'d2.4xlarge'|'d2.8xlarge'|'d3.xlarge'|'d3.2xlarge'|'d3.4xlarge'|'d3.8xlarge'|'d3en.xlarge'|'d3en.2xlarge'|'d3en.4xlarge'|'d3en.6xlarge'|'d3en.8xlarge'|'d3en.12xlarge'|'f1.2xlarge'|'f1.4xlarge'|'f1.16xlarge'|'m5.large'|'m5.xlarge'|'m5.2xlarge'|'m5.4xlarge'|'m5.8xlarge'|'m5.12xlarge'|'m5.16xlarge'|'m5.24xlarge'|'m5.metal'|'m5a.large'|'m5a.xlarge'|'m5a.2xlarge'|'m5a.4xlarge'|'m5a.8xlarge'|'m5a.12xlarge'|'m5a.16xlarge'|'m5a.24xlarge'|'m5d.large'|'m5d.xlarge'|'m5d.2xlarge'|'m5d.4xlarge'|'m5d.8xlarge'|'m5d.12xlarge'|'m5d.16xlarge'|'m5d.24xlarge'|'m5d.metal'|'m5ad.large'|'m5ad.xlarge'|'m5ad.2xlarge'|'m5ad.4xlarge'|'m5ad.8xlarge'|'m5ad.12xlarge'|'m5ad.16xlarge'|'m5ad.24xlarge'|'m5zn.large'|'m5zn.xlarge'|'m5zn.2xlarge'|'m5zn.3xlarge'|'m5zn.6xlarge'|'m5zn.12xlarge'|'m5zn.metal'|'h1.2xlarge'|'h1.4xlarge'|'h1.8xlarge'|'h1.16xlarge'|'z1d.large'|'z1d.xlarge'|'z1d.2xlarge'|'z1d.3xlarge'|'z1d.6xlarge'|'z1d.12xlarge'|'z1d.metal'|'u-6tb1.metal'|'u-9tb1.metal'|'u-12tb1.metal'|'u-18tb1.metal'|'u-24tb1.metal'|'a1.medium'|'a1.large'|'a1.xlarge'|'a1.2xlarge'|'a1.4xlarge'|'a1.metal'|'m5dn.large'|'m5dn.xlarge'|'m5dn.2xlarge'|'m5dn.4xlarge'|'m5dn.8xlarge'|'m5dn.12xlarge'|'m5dn.16xlarge'|'m5dn.24xlarge'|'m5n.large'|'m5n.xlarge'|'m5n.2xlarge'|'m5n.4xlarge'|'m5n.8xlarge'|'m5n.12xlarge'|'m5n.16xlarge'|'m5n.24xlarge'|'r5dn.large'|'r5dn.xlarge'|'r5dn.2xlarge'|'r5dn.4xlarge'|'r5dn.8xlarge'|'r5dn.12xlarge'|'r5dn.16xlarge'|'r5dn.24xlarge'|'r5n.large'|'r5n.xlarge'|'r5n.2xlarge'|'r5n.4xlarge'|'r5n.8xlarge'|'r5n.12xlarge'|'r5n.16xlarge'|'r5n.24xlarge'|'inf1.xlarge'|'inf1.2xlarge'|'inf1.6xlarge'|'inf1.24xlarge'|'m6g.metal'|'m6g.medium'|'m6g.large'|'m6g.xlarge'|'m6g.2xlarge'|'m6g.4xlarge'|'m6g.8xlarge'|'m6g.12xlarge'|'m6g.16xlarge'|'m6gd.metal'|'m6gd.medium'|'m6gd.large'|'m6gd.xlarge'|'m6gd.2xlarge'|'m6gd.4xlarge'|'m6gd.8xlarge'|'m6gd.12xlarge'|'m6gd.16xlarge'|'mac1.metal',
            'LocationType': 'region'|'availability-zone'|'availability-zone-id',
            'Location': 'string'
        },
    ],

}

**Response Structure**

- _(dict) --_

    - **InstanceTypeOfferings** _(list) --_

        The instance types offered.

        - _(dict) --_

            The instance types offered.

            - **InstanceType** _(string) --_

                The instance type. For more information, see [Instance Types](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/instance-types.html) in the _Amazon Elastic Compute Cloud User Guide_ .

            - **LocationType** _(string) --_

                The location type.

            - **Location** _(string) --_

                The identifier for the location. This depends on the location type. For example, if the location type is region , the location is the Region code (for example, us-east-2 .)


_class_ EC2.Paginator.DescribeInstanceTypes

paginator = client.get_paginator('describe_instance_types')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_instance_types()](#EC2.Client.describe_instance_types "EC2.Client.describe_instance_types").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeInstanceTypes)

**Request Syntax**

response_iterator = paginator.paginate(
    DryRun=True|False,
    InstanceTypes=[
        't1.micro'|'t2.nano'|'t2.micro'|'t2.small'|'t2.medium'|'t2.large'|'t2.xlarge'|'t2.2xlarge'|'t3.nano'|'t3.micro'|'t3.small'|'t3.medium'|'t3.large'|'t3.xlarge'|'t3.2xlarge'|'t3a.nano'|'t3a.micro'|'t3a.small'|'t3a.medium'|'t3a.large'|'t3a.xlarge'|'t3a.2xlarge'|'t4g.nano'|'t4g.micro'|'t4g.small'|'t4g.medium'|'t4g.large'|'t4g.xlarge'|'t4g.2xlarge'|'m1.small'|'m1.medium'|'m1.large'|'m1.xlarge'|'m3.medium'|'m3.large'|'m3.xlarge'|'m3.2xlarge'|'m4.large'|'m4.xlarge'|'m4.2xlarge'|'m4.4xlarge'|'m4.10xlarge'|'m4.16xlarge'|'m2.xlarge'|'m2.2xlarge'|'m2.4xlarge'|'cr1.8xlarge'|'r3.large'|'r3.xlarge'|'r3.2xlarge'|'r3.4xlarge'|'r3.8xlarge'|'r4.large'|'r4.xlarge'|'r4.2xlarge'|'r4.4xlarge'|'r4.8xlarge'|'r4.16xlarge'|'r5.large'|'r5.xlarge'|'r5.2xlarge'|'r5.4xlarge'|'r5.8xlarge'|'r5.12xlarge'|'r5.16xlarge'|'r5.24xlarge'|'r5.metal'|'r5a.large'|'r5a.xlarge'|'r5a.2xlarge'|'r5a.4xlarge'|'r5a.8xlarge'|'r5a.12xlarge'|'r5a.16xlarge'|'r5a.24xlarge'|'r5b.large'|'r5b.xlarge'|'r5b.2xlarge'|'r5b.4xlarge'|'r5b.8xlarge'|'r5b.12xlarge'|'r5b.16xlarge'|'r5b.24xlarge'|'r5b.metal'|'r5d.large'|'r5d.xlarge'|'r5d.2xlarge'|'r5d.4xlarge'|'r5d.8xlarge'|'r5d.12xlarge'|'r5d.16xlarge'|'r5d.24xlarge'|'r5d.metal'|'r5ad.large'|'r5ad.xlarge'|'r5ad.2xlarge'|'r5ad.4xlarge'|'r5ad.8xlarge'|'r5ad.12xlarge'|'r5ad.16xlarge'|'r5ad.24xlarge'|'r6g.metal'|'r6g.medium'|'r6g.large'|'r6g.xlarge'|'r6g.2xlarge'|'r6g.4xlarge'|'r6g.8xlarge'|'r6g.12xlarge'|'r6g.16xlarge'|'r6gd.metal'|'r6gd.medium'|'r6gd.large'|'r6gd.xlarge'|'r6gd.2xlarge'|'r6gd.4xlarge'|'r6gd.8xlarge'|'r6gd.12xlarge'|'r6gd.16xlarge'|'x1.16xlarge'|'x1.32xlarge'|'x1e.xlarge'|'x1e.2xlarge'|'x1e.4xlarge'|'x1e.8xlarge'|'x1e.16xlarge'|'x1e.32xlarge'|'i2.xlarge'|'i2.2xlarge'|'i2.4xlarge'|'i2.8xlarge'|'i3.large'|'i3.xlarge'|'i3.2xlarge'|'i3.4xlarge'|'i3.8xlarge'|'i3.16xlarge'|'i3.metal'|'i3en.large'|'i3en.xlarge'|'i3en.2xlarge'|'i3en.3xlarge'|'i3en.6xlarge'|'i3en.12xlarge'|'i3en.24xlarge'|'i3en.metal'|'hi1.4xlarge'|'hs1.8xlarge'|'c1.medium'|'c1.xlarge'|'c3.large'|'c3.xlarge'|'c3.2xlarge'|'c3.4xlarge'|'c3.8xlarge'|'c4.large'|'c4.xlarge'|'c4.2xlarge'|'c4.4xlarge'|'c4.8xlarge'|'c5.large'|'c5.xlarge'|'c5.2xlarge'|'c5.4xlarge'|'c5.9xlarge'|'c5.12xlarge'|'c5.18xlarge'|'c5.24xlarge'|'c5.metal'|'c5a.large'|'c5a.xlarge'|'c5a.2xlarge'|'c5a.4xlarge'|'c5a.8xlarge'|'c5a.12xlarge'|'c5a.16xlarge'|'c5a.24xlarge'|'c5ad.large'|'c5ad.xlarge'|'c5ad.2xlarge'|'c5ad.4xlarge'|'c5ad.8xlarge'|'c5ad.12xlarge'|'c5ad.16xlarge'|'c5ad.24xlarge'|'c5d.large'|'c5d.xlarge'|'c5d.2xlarge'|'c5d.4xlarge'|'c5d.9xlarge'|'c5d.12xlarge'|'c5d.18xlarge'|'c5d.24xlarge'|'c5d.metal'|'c5n.large'|'c5n.xlarge'|'c5n.2xlarge'|'c5n.4xlarge'|'c5n.9xlarge'|'c5n.18xlarge'|'c5n.metal'|'c6g.metal'|'c6g.medium'|'c6g.large'|'c6g.xlarge'|'c6g.2xlarge'|'c6g.4xlarge'|'c6g.8xlarge'|'c6g.12xlarge'|'c6g.16xlarge'|'c6gd.metal'|'c6gd.medium'|'c6gd.large'|'c6gd.xlarge'|'c6gd.2xlarge'|'c6gd.4xlarge'|'c6gd.8xlarge'|'c6gd.12xlarge'|'c6gd.16xlarge'|'c6gn.medium'|'c6gn.large'|'c6gn.xlarge'|'c6gn.2xlarge'|'c6gn.4xlarge'|'c6gn.8xlarge'|'c6gn.12xlarge'|'c6gn.16xlarge'|'cc1.4xlarge'|'cc2.8xlarge'|'g2.2xlarge'|'g2.8xlarge'|'g3.4xlarge'|'g3.8xlarge'|'g3.16xlarge'|'g3s.xlarge'|'g4ad.4xlarge'|'g4ad.8xlarge'|'g4ad.16xlarge'|'g4dn.xlarge'|'g4dn.2xlarge'|'g4dn.4xlarge'|'g4dn.8xlarge'|'g4dn.12xlarge'|'g4dn.16xlarge'|'g4dn.metal'|'cg1.4xlarge'|'p2.xlarge'|'p2.8xlarge'|'p2.16xlarge'|'p3.2xlarge'|'p3.8xlarge'|'p3.16xlarge'|'p3dn.24xlarge'|'p4d.24xlarge'|'d2.xlarge'|'d2.2xlarge'|'d2.4xlarge'|'d2.8xlarge'|'d3.xlarge'|'d3.2xlarge'|'d3.4xlarge'|'d3.8xlarge'|'d3en.xlarge'|'d3en.2xlarge'|'d3en.4xlarge'|'d3en.6xlarge'|'d3en.8xlarge'|'d3en.12xlarge'|'f1.2xlarge'|'f1.4xlarge'|'f1.16xlarge'|'m5.large'|'m5.xlarge'|'m5.2xlarge'|'m5.4xlarge'|'m5.8xlarge'|'m5.12xlarge'|'m5.16xlarge'|'m5.24xlarge'|'m5.metal'|'m5a.large'|'m5a.xlarge'|'m5a.2xlarge'|'m5a.4xlarge'|'m5a.8xlarge'|'m5a.12xlarge'|'m5a.16xlarge'|'m5a.24xlarge'|'m5d.large'|'m5d.xlarge'|'m5d.2xlarge'|'m5d.4xlarge'|'m5d.8xlarge'|'m5d.12xlarge'|'m5d.16xlarge'|'m5d.24xlarge'|'m5d.metal'|'m5ad.large'|'m5ad.xlarge'|'m5ad.2xlarge'|'m5ad.4xlarge'|'m5ad.8xlarge'|'m5ad.12xlarge'|'m5ad.16xlarge'|'m5ad.24xlarge'|'m5zn.large'|'m5zn.xlarge'|'m5zn.2xlarge'|'m5zn.3xlarge'|'m5zn.6xlarge'|'m5zn.12xlarge'|'m5zn.metal'|'h1.2xlarge'|'h1.4xlarge'|'h1.8xlarge'|'h1.16xlarge'|'z1d.large'|'z1d.xlarge'|'z1d.2xlarge'|'z1d.3xlarge'|'z1d.6xlarge'|'z1d.12xlarge'|'z1d.metal'|'u-6tb1.metal'|'u-9tb1.metal'|'u-12tb1.metal'|'u-18tb1.metal'|'u-24tb1.metal'|'a1.medium'|'a1.large'|'a1.xlarge'|'a1.2xlarge'|'a1.4xlarge'|'a1.metal'|'m5dn.large'|'m5dn.xlarge'|'m5dn.2xlarge'|'m5dn.4xlarge'|'m5dn.8xlarge'|'m5dn.12xlarge'|'m5dn.16xlarge'|'m5dn.24xlarge'|'m5n.large'|'m5n.xlarge'|'m5n.2xlarge'|'m5n.4xlarge'|'m5n.8xlarge'|'m5n.12xlarge'|'m5n.16xlarge'|'m5n.24xlarge'|'r5dn.large'|'r5dn.xlarge'|'r5dn.2xlarge'|'r5dn.4xlarge'|'r5dn.8xlarge'|'r5dn.12xlarge'|'r5dn.16xlarge'|'r5dn.24xlarge'|'r5n.large'|'r5n.xlarge'|'r5n.2xlarge'|'r5n.4xlarge'|'r5n.8xlarge'|'r5n.12xlarge'|'r5n.16xlarge'|'r5n.24xlarge'|'inf1.xlarge'|'inf1.2xlarge'|'inf1.6xlarge'|'inf1.24xlarge'|'m6g.metal'|'m6g.medium'|'m6g.large'|'m6g.xlarge'|'m6g.2xlarge'|'m6g.4xlarge'|'m6g.8xlarge'|'m6g.12xlarge'|'m6g.16xlarge'|'m6gd.metal'|'m6gd.medium'|'m6gd.large'|'m6gd.xlarge'|'m6gd.2xlarge'|'m6gd.4xlarge'|'m6gd.8xlarge'|'m6gd.12xlarge'|'m6gd.16xlarge'|'mac1.metal',
    ],
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **InstanceTypes** (_list_) --

    The instance types. For more information, see [Instance Types](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/instance-types.html) in the _Amazon Elastic Compute Cloud User Guide_ .

    - _(string) --_
- **Filters** (_list_) --

    One or more filters. Filter names and values are case-sensitive.

    - auto-recovery-supported - Indicates whether auto recovery is supported (true | false ).
    - bare-metal - Indicates whether it is a bare metal instance type (true | false ).
    - burstable-performance-supported - Indicates whether it is a burstable performance instance type (true | false ).
    - current-generation - Indicates whether this instance type is the latest generation instance type of an instance family (true | false ).
    - ebs-info.ebs-optimized-info.baseline-bandwidth-in-mbps - The baseline bandwidth performance for an EBS-optimized instance type, in Mbps.
    - ebs-info.ebs-optimized-info.baseline-iops - The baseline input/output storage operations per second for an EBS-optimized instance type.
    - ebs-info.ebs-optimized-info.baseline-throughput-in-mbps - The baseline throughput performance for an EBS-optimized instance type, in MB/s.
    - ebs-info.ebs-optimized-info.maximum-bandwidth-in-mbps - The maximum bandwidth performance for an EBS-optimized instance type, in Mbps.
    - ebs-info.ebs-optimized-info.maximum-iops - The maximum input/output storage operations per second for an EBS-optimized instance type.
    - ebs-info.ebs-optimized-info.maximum-throughput-in-mbps - The maximum throughput performance for an EBS-optimized instance type, in MB/s.
    - ebs-info.ebs-optimized-support - Indicates whether the instance type is EBS-optimized (supported | unsupported | default ).
    - ebs-info.encryption-support - Indicates whether EBS encryption is supported (supported | unsupported ).
    - ebs-info.nvme-support - Indicates whether non-volatile memory express (NVMe) is supported for EBS volumes (required | supported | unsupported ).
    - free-tier-eligible - Indicates whether the instance type is eligible to use in the free tier (true | false ).
    - hibernation-supported - Indicates whether On-Demand hibernation is supported (true | false ).
    - hypervisor - The hypervisor (nitro | xen ).
    - instance-storage-info.disk.count - The number of local disks.
    - instance-storage-info.disk.size-in-gb - The storage size of each instance storage disk, in GB.
    - instance-storage-info.disk.type - The storage technology for the local instance storage disks (hdd | ssd ).
    - instance-storage-info.nvme-support - Indicates whether non-volatile memory express (NVMe) is supported for instance store (required | supported ) | unsupported ).
    - instance-storage-info.total-size-in-gb - The total amount of storage available from all local instance storage, in GB.
    - instance-storage-supported - Indicates whether the instance type has local instance storage (true | false ).
    - instance-type - The instance type (for example c5.2xlarge or c5*).
    - memory-info.size-in-mib - The memory size.
    - network-info.efa-supported - Indicates whether the instance type supports Elastic Fabric Adapter (EFA) (true | false ).
    - network-info.ena-support - Indicates whether Elastic Network Adapter (ENA) is supported or required (required | supported | unsupported ).
    - network-info.ipv4-addresses-per-interface - The maximum number of private IPv4 addresses per network interface.
    - network-info.ipv6-addresses-per-interface - The maximum number of private IPv6 addresses per network interface.
    - network-info.ipv6-supported - Indicates whether the instance type supports IPv6 (true | false ).
    - network-info.maximum-network-interfaces - The maximum number of network interfaces per instance.
    - network-info.network-performance - The network performance (for example, "25 Gigabit").
    - processor-info.supported-architecture - The CPU architecture (arm64 | i386 | x86_64 ).
    - processor-info.sustained-clock-speed-in-ghz - The CPU clock speed, in GHz.
    - supported-root-device-type - The root device type (ebs | instance-store ).
    - supported-usage-class - The usage class (on-demand | spot ).
    - supported-virtualization-type - The virtualization type (hvm | paravirtual ).
    - vcpu-info.default-cores - The default number of cores for the instance type.
    - vcpu-info.default-threads-per-core - The default number of threads per core for the instance type.
    - vcpu-info.default-vcpus - The default number of vCPUs for the instance type.
    - vcpu-info.valid-cores - The number of cores that can be configured for the instance type.
    - vcpu-info.valid-threads-per-core - The number of threads per core that can be configured for the instance type. For example, "1" or "1,2".

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'InstanceTypes': [
        {
            'InstanceType': 't1.micro'|'t2.nano'|'t2.micro'|'t2.small'|'t2.medium'|'t2.large'|'t2.xlarge'|'t2.2xlarge'|'t3.nano'|'t3.micro'|'t3.small'|'t3.medium'|'t3.large'|'t3.xlarge'|'t3.2xlarge'|'t3a.nano'|'t3a.micro'|'t3a.small'|'t3a.medium'|'t3a.large'|'t3a.xlarge'|'t3a.2xlarge'|'t4g.nano'|'t4g.micro'|'t4g.small'|'t4g.medium'|'t4g.large'|'t4g.xlarge'|'t4g.2xlarge'|'m1.small'|'m1.medium'|'m1.large'|'m1.xlarge'|'m3.medium'|'m3.large'|'m3.xlarge'|'m3.2xlarge'|'m4.large'|'m4.xlarge'|'m4.2xlarge'|'m4.4xlarge'|'m4.10xlarge'|'m4.16xlarge'|'m2.xlarge'|'m2.2xlarge'|'m2.4xlarge'|'cr1.8xlarge'|'r3.large'|'r3.xlarge'|'r3.2xlarge'|'r3.4xlarge'|'r3.8xlarge'|'r4.large'|'r4.xlarge'|'r4.2xlarge'|'r4.4xlarge'|'r4.8xlarge'|'r4.16xlarge'|'r5.large'|'r5.xlarge'|'r5.2xlarge'|'r5.4xlarge'|'r5.8xlarge'|'r5.12xlarge'|'r5.16xlarge'|'r5.24xlarge'|'r5.metal'|'r5a.large'|'r5a.xlarge'|'r5a.2xlarge'|'r5a.4xlarge'|'r5a.8xlarge'|'r5a.12xlarge'|'r5a.16xlarge'|'r5a.24xlarge'|'r5b.large'|'r5b.xlarge'|'r5b.2xlarge'|'r5b.4xlarge'|'r5b.8xlarge'|'r5b.12xlarge'|'r5b.16xlarge'|'r5b.24xlarge'|'r5b.metal'|'r5d.large'|'r5d.xlarge'|'r5d.2xlarge'|'r5d.4xlarge'|'r5d.8xlarge'|'r5d.12xlarge'|'r5d.16xlarge'|'r5d.24xlarge'|'r5d.metal'|'r5ad.large'|'r5ad.xlarge'|'r5ad.2xlarge'|'r5ad.4xlarge'|'r5ad.8xlarge'|'r5ad.12xlarge'|'r5ad.16xlarge'|'r5ad.24xlarge'|'r6g.metal'|'r6g.medium'|'r6g.large'|'r6g.xlarge'|'r6g.2xlarge'|'r6g.4xlarge'|'r6g.8xlarge'|'r6g.12xlarge'|'r6g.16xlarge'|'r6gd.metal'|'r6gd.medium'|'r6gd.large'|'r6gd.xlarge'|'r6gd.2xlarge'|'r6gd.4xlarge'|'r6gd.8xlarge'|'r6gd.12xlarge'|'r6gd.16xlarge'|'x1.16xlarge'|'x1.32xlarge'|'x1e.xlarge'|'x1e.2xlarge'|'x1e.4xlarge'|'x1e.8xlarge'|'x1e.16xlarge'|'x1e.32xlarge'|'i2.xlarge'|'i2.2xlarge'|'i2.4xlarge'|'i2.8xlarge'|'i3.large'|'i3.xlarge'|'i3.2xlarge'|'i3.4xlarge'|'i3.8xlarge'|'i3.16xlarge'|'i3.metal'|'i3en.large'|'i3en.xlarge'|'i3en.2xlarge'|'i3en.3xlarge'|'i3en.6xlarge'|'i3en.12xlarge'|'i3en.24xlarge'|'i3en.metal'|'hi1.4xlarge'|'hs1.8xlarge'|'c1.medium'|'c1.xlarge'|'c3.large'|'c3.xlarge'|'c3.2xlarge'|'c3.4xlarge'|'c3.8xlarge'|'c4.large'|'c4.xlarge'|'c4.2xlarge'|'c4.4xlarge'|'c4.8xlarge'|'c5.large'|'c5.xlarge'|'c5.2xlarge'|'c5.4xlarge'|'c5.9xlarge'|'c5.12xlarge'|'c5.18xlarge'|'c5.24xlarge'|'c5.metal'|'c5a.large'|'c5a.xlarge'|'c5a.2xlarge'|'c5a.4xlarge'|'c5a.8xlarge'|'c5a.12xlarge'|'c5a.16xlarge'|'c5a.24xlarge'|'c5ad.large'|'c5ad.xlarge'|'c5ad.2xlarge'|'c5ad.4xlarge'|'c5ad.8xlarge'|'c5ad.12xlarge'|'c5ad.16xlarge'|'c5ad.24xlarge'|'c5d.large'|'c5d.xlarge'|'c5d.2xlarge'|'c5d.4xlarge'|'c5d.9xlarge'|'c5d.12xlarge'|'c5d.18xlarge'|'c5d.24xlarge'|'c5d.metal'|'c5n.large'|'c5n.xlarge'|'c5n.2xlarge'|'c5n.4xlarge'|'c5n.9xlarge'|'c5n.18xlarge'|'c5n.metal'|'c6g.metal'|'c6g.medium'|'c6g.large'|'c6g.xlarge'|'c6g.2xlarge'|'c6g.4xlarge'|'c6g.8xlarge'|'c6g.12xlarge'|'c6g.16xlarge'|'c6gd.metal'|'c6gd.medium'|'c6gd.large'|'c6gd.xlarge'|'c6gd.2xlarge'|'c6gd.4xlarge'|'c6gd.8xlarge'|'c6gd.12xlarge'|'c6gd.16xlarge'|'c6gn.medium'|'c6gn.large'|'c6gn.xlarge'|'c6gn.2xlarge'|'c6gn.4xlarge'|'c6gn.8xlarge'|'c6gn.12xlarge'|'c6gn.16xlarge'|'cc1.4xlarge'|'cc2.8xlarge'|'g2.2xlarge'|'g2.8xlarge'|'g3.4xlarge'|'g3.8xlarge'|'g3.16xlarge'|'g3s.xlarge'|'g4ad.4xlarge'|'g4ad.8xlarge'|'g4ad.16xlarge'|'g4dn.xlarge'|'g4dn.2xlarge'|'g4dn.4xlarge'|'g4dn.8xlarge'|'g4dn.12xlarge'|'g4dn.16xlarge'|'g4dn.metal'|'cg1.4xlarge'|'p2.xlarge'|'p2.8xlarge'|'p2.16xlarge'|'p3.2xlarge'|'p3.8xlarge'|'p3.16xlarge'|'p3dn.24xlarge'|'p4d.24xlarge'|'d2.xlarge'|'d2.2xlarge'|'d2.4xlarge'|'d2.8xlarge'|'d3.xlarge'|'d3.2xlarge'|'d3.4xlarge'|'d3.8xlarge'|'d3en.xlarge'|'d3en.2xlarge'|'d3en.4xlarge'|'d3en.6xlarge'|'d3en.8xlarge'|'d3en.12xlarge'|'f1.2xlarge'|'f1.4xlarge'|'f1.16xlarge'|'m5.large'|'m5.xlarge'|'m5.2xlarge'|'m5.4xlarge'|'m5.8xlarge'|'m5.12xlarge'|'m5.16xlarge'|'m5.24xlarge'|'m5.metal'|'m5a.large'|'m5a.xlarge'|'m5a.2xlarge'|'m5a.4xlarge'|'m5a.8xlarge'|'m5a.12xlarge'|'m5a.16xlarge'|'m5a.24xlarge'|'m5d.large'|'m5d.xlarge'|'m5d.2xlarge'|'m5d.4xlarge'|'m5d.8xlarge'|'m5d.12xlarge'|'m5d.16xlarge'|'m5d.24xlarge'|'m5d.metal'|'m5ad.large'|'m5ad.xlarge'|'m5ad.2xlarge'|'m5ad.4xlarge'|'m5ad.8xlarge'|'m5ad.12xlarge'|'m5ad.16xlarge'|'m5ad.24xlarge'|'m5zn.large'|'m5zn.xlarge'|'m5zn.2xlarge'|'m5zn.3xlarge'|'m5zn.6xlarge'|'m5zn.12xlarge'|'m5zn.metal'|'h1.2xlarge'|'h1.4xlarge'|'h1.8xlarge'|'h1.16xlarge'|'z1d.large'|'z1d.xlarge'|'z1d.2xlarge'|'z1d.3xlarge'|'z1d.6xlarge'|'z1d.12xlarge'|'z1d.metal'|'u-6tb1.metal'|'u-9tb1.metal'|'u-12tb1.metal'|'u-18tb1.metal'|'u-24tb1.metal'|'a1.medium'|'a1.large'|'a1.xlarge'|'a1.2xlarge'|'a1.4xlarge'|'a1.metal'|'m5dn.large'|'m5dn.xlarge'|'m5dn.2xlarge'|'m5dn.4xlarge'|'m5dn.8xlarge'|'m5dn.12xlarge'|'m5dn.16xlarge'|'m5dn.24xlarge'|'m5n.large'|'m5n.xlarge'|'m5n.2xlarge'|'m5n.4xlarge'|'m5n.8xlarge'|'m5n.12xlarge'|'m5n.16xlarge'|'m5n.24xlarge'|'r5dn.large'|'r5dn.xlarge'|'r5dn.2xlarge'|'r5dn.4xlarge'|'r5dn.8xlarge'|'r5dn.12xlarge'|'r5dn.16xlarge'|'r5dn.24xlarge'|'r5n.large'|'r5n.xlarge'|'r5n.2xlarge'|'r5n.4xlarge'|'r5n.8xlarge'|'r5n.12xlarge'|'r5n.16xlarge'|'r5n.24xlarge'|'inf1.xlarge'|'inf1.2xlarge'|'inf1.6xlarge'|'inf1.24xlarge'|'m6g.metal'|'m6g.medium'|'m6g.large'|'m6g.xlarge'|'m6g.2xlarge'|'m6g.4xlarge'|'m6g.8xlarge'|'m6g.12xlarge'|'m6g.16xlarge'|'m6gd.metal'|'m6gd.medium'|'m6gd.large'|'m6gd.xlarge'|'m6gd.2xlarge'|'m6gd.4xlarge'|'m6gd.8xlarge'|'m6gd.12xlarge'|'m6gd.16xlarge'|'mac1.metal',
            'CurrentGeneration': True|False,
            'FreeTierEligible': True|False,
            'SupportedUsageClasses': [
                'spot'|'on-demand',
            ],
            'SupportedRootDeviceTypes': [
                'ebs'|'instance-store',
            ],
            'SupportedVirtualizationTypes': [
                'hvm'|'paravirtual',
            ],
            'BareMetal': True|False,
            'Hypervisor': 'nitro'|'xen',
            'ProcessorInfo': {
                'SupportedArchitectures': [
                    'i386'|'x86_64'|'arm64',
                ],
                'SustainedClockSpeedInGhz': 123.0
            },
            'VCpuInfo': {
                'DefaultVCpus': 123,
                'DefaultCores': 123,
                'DefaultThreadsPerCore': 123,
                'ValidCores': [
                    123,
                ],
                'ValidThreadsPerCore': [
                    123,
                ]
            },
            'MemoryInfo': {
                'SizeInMiB': 123
            },
            'InstanceStorageSupported': True|False,
            'InstanceStorageInfo': {
                'TotalSizeInGB': 123,
                'Disks': [
                    {
                        'SizeInGB': 123,
                        'Count': 123,
                        'Type': 'hdd'|'ssd'
                    },
                ],
                'NvmeSupport': 'unsupported'|'supported'|'required'
            },
            'EbsInfo': {
                'EbsOptimizedSupport': 'unsupported'|'supported'|'default',
                'EncryptionSupport': 'unsupported'|'supported',
                'EbsOptimizedInfo': {
                    'BaselineBandwidthInMbps': 123,
                    'BaselineThroughputInMBps': 123.0,
                    'BaselineIops': 123,
                    'MaximumBandwidthInMbps': 123,
                    'MaximumThroughputInMBps': 123.0,
                    'MaximumIops': 123
                },
                'NvmeSupport': 'unsupported'|'supported'|'required'
            },
            'NetworkInfo': {
                'NetworkPerformance': 'string',
                'MaximumNetworkInterfaces': 123,
                'MaximumNetworkCards': 123,
                'DefaultNetworkCardIndex': 123,
                'NetworkCards': [
                    {
                        'NetworkCardIndex': 123,
                        'NetworkPerformance': 'string',
                        'MaximumNetworkInterfaces': 123
                    },
                ],
                'Ipv4AddressesPerInterface': 123,
                'Ipv6AddressesPerInterface': 123,
                'Ipv6Supported': True|False,
                'EnaSupport': 'unsupported'|'supported'|'required',
                'EfaSupported': True|False
            },
            'GpuInfo': {
                'Gpus': [
                    {
                        'Name': 'string',
                        'Manufacturer': 'string',
                        'Count': 123,
                        'MemoryInfo': {
                            'SizeInMiB': 123
                        }
                    },
                ],
                'TotalGpuMemoryInMiB': 123
            },
            'FpgaInfo': {
                'Fpgas': [
                    {
                        'Name': 'string',
                        'Manufacturer': 'string',
                        'Count': 123,
                        'MemoryInfo': {
                            'SizeInMiB': 123
                        }
                    },
                ],
                'TotalFpgaMemoryInMiB': 123
            },
            'PlacementGroupInfo': {
                'SupportedStrategies': [
                    'cluster'|'partition'|'spread',
                ]
            },
            'InferenceAcceleratorInfo': {
                'Accelerators': [
                    {
                        'Count': 123,
                        'Name': 'string',
                        'Manufacturer': 'string'
                    },
                ]
            },
            'HibernationSupported': True|False,
            'BurstablePerformanceSupported': True|False,
            'DedicatedHostsSupported': True|False,
            'AutoRecoverySupported': True|False
        },
    ],

}

**Response Structure**

- _(dict) --_

    - **InstanceTypes** _(list) --_

        The instance type. For more information, see [Instance Types](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/instance-types.html) in the _Amazon Elastic Compute Cloud User Guide_ .

        - _(dict) --_

            Describes the instance type.

            - **InstanceType** _(string) --_

                The instance type. For more information, see [Instance Types](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/instance-types.html) in the _Amazon Elastic Compute Cloud User Guide_ .

            - **CurrentGeneration** _(boolean) --_

                Indicates whether the instance type is current generation.

            - **FreeTierEligible** _(boolean) --_

                Indicates whether the instance type is eligible for the free tier.

            - **SupportedUsageClasses** _(list) --_

                Indicates whether the instance type is offered for spot or On-Demand.

                - _(string) --_
            - **SupportedRootDeviceTypes** _(list) --_

                The supported root device types.

                - _(string) --_
            - **SupportedVirtualizationTypes** _(list) --_

                The supported virtualization types.

                - _(string) --_
            - **BareMetal** _(boolean) --_

                Indicates whether the instance is a bare metal instance type.

            - **Hypervisor** _(string) --_

                The hypervisor for the instance type.

            - **ProcessorInfo** _(dict) --_

                Describes the processor.

                - **SupportedArchitectures** _(list) --_

                    The architectures supported by the instance type.

                    - _(string) --_
                - **SustainedClockSpeedInGhz** _(float) --_

                    The speed of the processor, in GHz.

            - **VCpuInfo** _(dict) --_

                Describes the vCPU configurations for the instance type.

                - **DefaultVCpus** _(integer) --_

                    The default number of vCPUs for the instance type.

                - **DefaultCores** _(integer) --_

                    The default number of cores for the instance type.

                - **DefaultThreadsPerCore** _(integer) --_

                    The default number of threads per core for the instance type.

                - **ValidCores** _(list) --_

                    The valid number of cores that can be configured for the instance type.

                    - _(integer) --_
                - **ValidThreadsPerCore** _(list) --_

                    The valid number of threads per core that can be configured for the instance type.

                    - _(integer) --_
            - **MemoryInfo** _(dict) --_

                Describes the memory for the instance type.

                - **SizeInMiB** _(integer) --_

                    The size of the memory, in MiB.

            - **InstanceStorageSupported** _(boolean) --_

                Indicates whether instance storage is supported.

            - **InstanceStorageInfo** _(dict) --_

                Describes the instance storage for the instance type.

                - **TotalSizeInGB** _(integer) --_

                    The total size of the disks, in GB.

                - **Disks** _(list) --_

                    Describes the disks that are available for the instance type.

                    - _(dict) --_

                        Describes the disk.

                        - **SizeInGB** _(integer) --_

                            The size of the disk in GB.

                        - **Count** _(integer) --_

                            The number of disks with this configuration.

                        - **Type** _(string) --_

                            The type of disk.

                - **NvmeSupport** _(string) --_

                    Indicates whether non-volatile memory express (NVMe) is supported for instance store.

            - **EbsInfo** _(dict) --_

                Describes the Amazon EBS settings for the instance type.

                - **EbsOptimizedSupport** _(string) --_

                    Indicates whether the instance type is Amazon EBS-optimized. For more information, see [Amazon EBS-Optimized Instances](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EBSOptimized.html) in _Amazon EC2 User Guide for Linux Instances_ .

                - **EncryptionSupport** _(string) --_

                    Indicates whether Amazon EBS encryption is supported.

                - **EbsOptimizedInfo** _(dict) --_

                    Describes the optimized EBS performance for the instance type.

                    - **BaselineBandwidthInMbps** _(integer) --_

                        The baseline bandwidth performance for an EBS-optimized instance type, in Mbps.

                    - **BaselineThroughputInMBps** _(float) --_

                        The baseline throughput performance for an EBS-optimized instance type, in MB/s.

                    - **BaselineIops** _(integer) --_

                        The baseline input/output storage operations per seconds for an EBS-optimized instance type.

                    - **MaximumBandwidthInMbps** _(integer) --_

                        The maximum bandwidth performance for an EBS-optimized instance type, in Mbps.

                    - **MaximumThroughputInMBps** _(float) --_

                        The maximum throughput performance for an EBS-optimized instance type, in MB/s.

                    - **MaximumIops** _(integer) --_

                        The maximum input/output storage operations per second for an EBS-optimized instance type.

                - **NvmeSupport** _(string) --_

                    Indicates whether non-volatile memory express (NVMe) is supported.

            - **NetworkInfo** _(dict) --_

                Describes the network settings for the instance type.

                - **NetworkPerformance** _(string) --_

                    The network performance.

                - **MaximumNetworkInterfaces** _(integer) --_

                    The maximum number of network interfaces for the instance type.

                - **MaximumNetworkCards** _(integer) --_

                    The maximum number of physical network cards that can be allocated to the instance.

                - **DefaultNetworkCardIndex** _(integer) --_

                    The index of the default network card, starting at 0.

                - **NetworkCards** _(list) --_

                    Describes the network cards for the instance type.

                    - _(dict) --_

                        Describes the network card support of the instance type.

                        - **NetworkCardIndex** _(integer) --_

                            The index of the network card.

                        - **NetworkPerformance** _(string) --_

                            The network performance of the network card.

                        - **MaximumNetworkInterfaces** _(integer) --_

                            The maximum number of network interfaces for the network card.

                - **Ipv4AddressesPerInterface** _(integer) --_

                    The maximum number of IPv4 addresses per network interface.

                - **Ipv6AddressesPerInterface** _(integer) --_

                    The maximum number of IPv6 addresses per network interface.

                - **Ipv6Supported** _(boolean) --_

                    Indicates whether IPv6 is supported.

                - **EnaSupport** _(string) --_

                    Indicates whether Elastic Network Adapter (ENA) is supported.

                - **EfaSupported** _(boolean) --_

                    Indicates whether Elastic Fabric Adapter (EFA) is supported.

            - **GpuInfo** _(dict) --_

                Describes the GPU accelerator settings for the instance type.

                - **Gpus** _(list) --_

                    Describes the GPU accelerators for the instance type.

                    - _(dict) --_

                        Describes the GPU accelerators for the instance type.

                        - **Name** _(string) --_

                            The name of the GPU accelerator.

                        - **Manufacturer** _(string) --_

                            The manufacturer of the GPU accelerator.

                        - **Count** _(integer) --_

                            The number of GPUs for the instance type.

                        - **MemoryInfo** _(dict) --_

                            Describes the memory available to the GPU accelerator.

                            - **SizeInMiB** _(integer) --_

                                The size of the memory available to the GPU accelerator, in MiB.

                - **TotalGpuMemoryInMiB** _(integer) --_

                    The total size of the memory for the GPU accelerators for the instance type, in MiB.

            - **FpgaInfo** _(dict) --_

                Describes the FPGA accelerator settings for the instance type.

                - **Fpgas** _(list) --_

                    Describes the FPGAs for the instance type.

                    - _(dict) --_

                        Describes the FPGA accelerator for the instance type.

                        - **Name** _(string) --_

                            The name of the FPGA accelerator.

                        - **Manufacturer** _(string) --_

                            The manufacturer of the FPGA accelerator.

                        - **Count** _(integer) --_

                            The count of FPGA accelerators for the instance type.

                        - **MemoryInfo** _(dict) --_

                            Describes the memory for the FPGA accelerator for the instance type.

                            - **SizeInMiB** _(integer) --_

                                The size of the memory available to the FPGA accelerator, in MiB.

                - **TotalFpgaMemoryInMiB** _(integer) --_

                    The total memory of all FPGA accelerators for the instance type.

            - **PlacementGroupInfo** _(dict) --_

                Describes the placement group settings for the instance type.

                - **SupportedStrategies** _(list) --_

                    The supported placement group types.

                    - _(string) --_
            - **InferenceAcceleratorInfo** _(dict) --_

                Describes the Inference accelerator settings for the instance type.

                - **Accelerators** _(list) --_

                    Describes the Inference accelerators for the instance type.

                    - _(dict) --_

                        Describes the Inference accelerators for the instance type.

                        - **Count** _(integer) --_

                            The number of Inference accelerators for the instance type.

                        - **Name** _(string) --_

                            The name of the Inference accelerator.

                        - **Manufacturer** _(string) --_

                            The manufacturer of the Inference accelerator.

            - **HibernationSupported** _(boolean) --_

                Indicates whether On-Demand hibernation is supported.

            - **BurstablePerformanceSupported** _(boolean) --_

                Indicates whether the instance type is a burstable performance instance type.

            - **DedicatedHostsSupported** _(boolean) --_

                Indicates whether Dedicated Hosts are supported on the instance type.

            - **AutoRecoverySupported** _(boolean) --_

                Indicates whether auto recovery is supported.


_class_ EC2.Paginator.DescribeInstances

paginator = client.get_paginator('describe_instances')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_instances()](#EC2.Client.describe_instances "EC2.Client.describe_instances").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeInstances)

**Request Syntax**

response_iterator = paginator.paginate(
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    InstanceIds=[
        'string',
    ],
    DryRun=True|False,
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **Filters** (_list_) --

    The filters.

    - affinity - The affinity setting for an instance running on a Dedicated Host (default | host ).
    - architecture - The instance architecture (i386 | x86_64 | arm64 ).
    - availability-zone - The Availability Zone of the instance.
    - block-device-mapping.attach-time - The attach time for an EBS volume mapped to the instance, for example, 2010-09-15T17:15:20.000Z .
    - block-device-mapping.delete-on-termination - A Boolean that indicates whether the EBS volume is deleted on instance termination.
    - block-device-mapping.device-name - The device name specified in the block device mapping (for example, /dev/sdh or xvdh ).
    - block-device-mapping.status - The status for the EBS volume (attaching | attached | detaching | detached ).
    - block-device-mapping.volume-id - The volume ID of the EBS volume.
    - client-token - The idempotency token you provided when you launched the instance.
    - dns-name - The public DNS name of the instance.
    - group-id - The ID of the security group for the instance. EC2-Classic only.
    - group-name - The name of the security group for the instance. EC2-Classic only.
    - hibernation-options.configured - A Boolean that indicates whether the instance is enabled for hibernation. A value of true means that the instance is enabled for hibernation.
    - host-id - The ID of the Dedicated Host on which the instance is running, if applicable.
    - hypervisor - The hypervisor type of the instance (ovm | xen ). The value xen is used for both Xen and Nitro hypervisors.
    - iam-instance-profile.arn - The instance profile associated with the instance. Specified as an ARN.
    - image-id - The ID of the image used to launch the instance.
    - instance-id - The ID of the instance.
    - instance-lifecycle - Indicates whether this is a Spot Instance or a Scheduled Instance (spot | scheduled ).
    - instance-state-code - The state of the instance, as a 16-bit unsigned integer. The high byte is used for internal purposes and should be ignored. The low byte is set based on the state represented. The valid values are: 0 (pending), 16 (running), 32 (shutting-down), 48 (terminated), 64 (stopping), and 80 (stopped).
    - instance-state-name - The state of the instance (pending | running | shutting-down | terminated | stopping | stopped ).
    - instance-type - The type of instance (for example, t2.micro ).
    - instance.group-id - The ID of the security group for the instance.
    - instance.group-name - The name of the security group for the instance.
    - ip-address - The public IPv4 address of the instance.
    - kernel-id - The kernel ID.
    - key-name - The name of the key pair used when the instance was launched.
    - launch-index - When launching multiple instances, this is the index for the instance in the launch group (for example, 0, 1, 2, and so on).
    - launch-time - The time when the instance was launched.
    - metadata-options.http-tokens - The metadata request authorization state (optional | required )
    - metadata-options.http-put-response-hop-limit - The http metadata request put response hop limit (integer, possible values 1 to 64 )
    - metadata-options.http-endpoint - Enable or disable metadata access on http endpoint (enabled | disabled )
    - monitoring-state - Indicates whether detailed monitoring is enabled (disabled | enabled ).
    - network-interface.addresses.private-ip-address - The private IPv4 address associated with the network interface.
    - network-interface.addresses.primary - Specifies whether the IPv4 address of the network interface is the primary private IPv4 address.
    - network-interface.addresses.association.public-ip - The ID of the association of an Elastic IP address (IPv4) with a network interface.
    - network-interface.addresses.association.ip-owner-id - The owner ID of the private IPv4 address associated with the network interface.
    - network-interface.association.public-ip - The address of the Elastic IP address (IPv4) bound to the network interface.
    - network-interface.association.ip-owner-id - The owner of the Elastic IP address (IPv4) associated with the network interface.
    - network-interface.association.allocation-id - The allocation ID returned when you allocated the Elastic IP address (IPv4) for your network interface.
    - network-interface.association.association-id - The association ID returned when the network interface was associated with an IPv4 address.
    - network-interface.attachment.attachment-id - The ID of the interface attachment.
    - network-interface.attachment.instance-id - The ID of the instance to which the network interface is attached.
    - network-interface.attachment.instance-owner-id - The owner ID of the instance to which the network interface is attached.
    - network-interface.attachment.device-index - The device index to which the network interface is attached.
    - network-interface.attachment.status - The status of the attachment (attaching | attached | detaching | detached ).
    - network-interface.attachment.attach-time - The time that the network interface was attached to an instance.
    - network-interface.attachment.delete-on-termination - Specifies whether the attachment is deleted when an instance is terminated.
    - network-interface.availability-zone - The Availability Zone for the network interface.
    - network-interface.description - The description of the network interface.
    - network-interface.group-id - The ID of a security group associated with the network interface.
    - network-interface.group-name - The name of a security group associated with the network interface.
    - network-interface.ipv6-addresses.ipv6-address - The IPv6 address associated with the network interface.
    - network-interface.mac-address - The MAC address of the network interface.
    - network-interface.network-interface-id - The ID of the network interface.
    - network-interface.owner-id - The ID of the owner of the network interface.
    - network-interface.private-dns-name - The private DNS name of the network interface.
    - network-interface.requester-id - The requester ID for the network interface.
    - network-interface.requester-managed - Indicates whether the network interface is being managed by AWS.
    - network-interface.status - The status of the network interface (available ) | in-use ).
    - network-interface.source-dest-check - Whether the network interface performs source/destination checking. A value of true means that checking is enabled, and false means that checking is disabled. The value must be false for the network interface to perform network address translation (NAT) in your VPC.
    - network-interface.subnet-id - The ID of the subnet for the network interface.
    - network-interface.vpc-id - The ID of the VPC for the network interface.
    - owner-id - The AWS account ID of the instance owner.
    - placement-group-name - The name of the placement group for the instance.
    - placement-partition-number - The partition in which the instance is located.
    - platform - The platform. To list only Windows instances, use windows .
    - private-dns-name - The private IPv4 DNS name of the instance.
    - private-ip-address - The private IPv4 address of the instance.
    - product-code - The product code associated with the AMI used to launch the instance.
    - product-code.type - The type of product code (devpay | marketplace ).
    - ramdisk-id - The RAM disk ID.
    - reason - The reason for the current state of the instance (for example, shows "User Initiated [date]" when you stop or terminate the instance). Similar to the state-reason-code filter.
    - requester-id - The ID of the entity that launched the instance on your behalf (for example, AWS Management Console, Auto Scaling, and so on).
    - reservation-id - The ID of the instance's reservation. A reservation ID is created any time you launch an instance. A reservation ID has a one-to-one relationship with an instance launch request, but can be associated with more than one instance if you launch multiple instances using the same launch request. For example, if you launch one instance, you get one reservation ID. If you launch ten instances using the same launch request, you also get one reservation ID.
    - root-device-name - The device name of the root device volume (for example, /dev/sda1 ).
    - root-device-type - The type of the root device volume (ebs | instance-store ).
    - source-dest-check - Indicates whether the instance performs source/destination checking. A value of true means that checking is enabled, and false means that checking is disabled. The value must be false for the instance to perform network address translation (NAT) in your VPC.
    - spot-instance-request-id - The ID of the Spot Instance request.
    - state-reason-code - The reason code for the state change.
    - state-reason-message - A message that describes the state change.
    - subnet-id - The ID of the subnet for the instance.
    - tag :<key> - The key/value combination of a tag assigned to the resource. Use the tag key in the filter name and the tag value as the filter value. For example, to find all resources that have a tag with the key Owner and the value TeamA , specify tag:Owner for the filter name and TeamA for the filter value.
    - tag-key - The key of a tag assigned to the resource. Use this filter to find all resources that have a tag with a specific key, regardless of the tag value.
    - tenancy - The tenancy of an instance (dedicated | default | host ).
    - virtualization-type - The virtualization type of the instance (paravirtual | hvm ).
    - vpc-id - The ID of the VPC that the instance is running in.

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **InstanceIds** (_list_) --

    The instance IDs.

    Default: Describes all your instances.

    - _(string) --_
- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'Reservations': [
        {
            'Groups': [
                {
                    'GroupName': 'string',
                    'GroupId': 'string'
                },
            ],
            'Instances': [
                {
                    'AmiLaunchIndex': 123,
                    'ImageId': 'string',
                    'InstanceId': 'string',
                    'InstanceType': 't1.micro'|'t2.nano'|'t2.micro'|'t2.small'|'t2.medium'|'t2.large'|'t2.xlarge'|'t2.2xlarge'|'t3.nano'|'t3.micro'|'t3.small'|'t3.medium'|'t3.large'|'t3.xlarge'|'t3.2xlarge'|'t3a.nano'|'t3a.micro'|'t3a.small'|'t3a.medium'|'t3a.large'|'t3a.xlarge'|'t3a.2xlarge'|'t4g.nano'|'t4g.micro'|'t4g.small'|'t4g.medium'|'t4g.large'|'t4g.xlarge'|'t4g.2xlarge'|'m1.small'|'m1.medium'|'m1.large'|'m1.xlarge'|'m3.medium'|'m3.large'|'m3.xlarge'|'m3.2xlarge'|'m4.large'|'m4.xlarge'|'m4.2xlarge'|'m4.4xlarge'|'m4.10xlarge'|'m4.16xlarge'|'m2.xlarge'|'m2.2xlarge'|'m2.4xlarge'|'cr1.8xlarge'|'r3.large'|'r3.xlarge'|'r3.2xlarge'|'r3.4xlarge'|'r3.8xlarge'|'r4.large'|'r4.xlarge'|'r4.2xlarge'|'r4.4xlarge'|'r4.8xlarge'|'r4.16xlarge'|'r5.large'|'r5.xlarge'|'r5.2xlarge'|'r5.4xlarge'|'r5.8xlarge'|'r5.12xlarge'|'r5.16xlarge'|'r5.24xlarge'|'r5.metal'|'r5a.large'|'r5a.xlarge'|'r5a.2xlarge'|'r5a.4xlarge'|'r5a.8xlarge'|'r5a.12xlarge'|'r5a.16xlarge'|'r5a.24xlarge'|'r5b.large'|'r5b.xlarge'|'r5b.2xlarge'|'r5b.4xlarge'|'r5b.8xlarge'|'r5b.12xlarge'|'r5b.16xlarge'|'r5b.24xlarge'|'r5b.metal'|'r5d.large'|'r5d.xlarge'|'r5d.2xlarge'|'r5d.4xlarge'|'r5d.8xlarge'|'r5d.12xlarge'|'r5d.16xlarge'|'r5d.24xlarge'|'r5d.metal'|'r5ad.large'|'r5ad.xlarge'|'r5ad.2xlarge'|'r5ad.4xlarge'|'r5ad.8xlarge'|'r5ad.12xlarge'|'r5ad.16xlarge'|'r5ad.24xlarge'|'r6g.metal'|'r6g.medium'|'r6g.large'|'r6g.xlarge'|'r6g.2xlarge'|'r6g.4xlarge'|'r6g.8xlarge'|'r6g.12xlarge'|'r6g.16xlarge'|'r6gd.metal'|'r6gd.medium'|'r6gd.large'|'r6gd.xlarge'|'r6gd.2xlarge'|'r6gd.4xlarge'|'r6gd.8xlarge'|'r6gd.12xlarge'|'r6gd.16xlarge'|'x1.16xlarge'|'x1.32xlarge'|'x1e.xlarge'|'x1e.2xlarge'|'x1e.4xlarge'|'x1e.8xlarge'|'x1e.16xlarge'|'x1e.32xlarge'|'i2.xlarge'|'i2.2xlarge'|'i2.4xlarge'|'i2.8xlarge'|'i3.large'|'i3.xlarge'|'i3.2xlarge'|'i3.4xlarge'|'i3.8xlarge'|'i3.16xlarge'|'i3.metal'|'i3en.large'|'i3en.xlarge'|'i3en.2xlarge'|'i3en.3xlarge'|'i3en.6xlarge'|'i3en.12xlarge'|'i3en.24xlarge'|'i3en.metal'|'hi1.4xlarge'|'hs1.8xlarge'|'c1.medium'|'c1.xlarge'|'c3.large'|'c3.xlarge'|'c3.2xlarge'|'c3.4xlarge'|'c3.8xlarge'|'c4.large'|'c4.xlarge'|'c4.2xlarge'|'c4.4xlarge'|'c4.8xlarge'|'c5.large'|'c5.xlarge'|'c5.2xlarge'|'c5.4xlarge'|'c5.9xlarge'|'c5.12xlarge'|'c5.18xlarge'|'c5.24xlarge'|'c5.metal'|'c5a.large'|'c5a.xlarge'|'c5a.2xlarge'|'c5a.4xlarge'|'c5a.8xlarge'|'c5a.12xlarge'|'c5a.16xlarge'|'c5a.24xlarge'|'c5ad.large'|'c5ad.xlarge'|'c5ad.2xlarge'|'c5ad.4xlarge'|'c5ad.8xlarge'|'c5ad.12xlarge'|'c5ad.16xlarge'|'c5ad.24xlarge'|'c5d.large'|'c5d.xlarge'|'c5d.2xlarge'|'c5d.4xlarge'|'c5d.9xlarge'|'c5d.12xlarge'|'c5d.18xlarge'|'c5d.24xlarge'|'c5d.metal'|'c5n.large'|'c5n.xlarge'|'c5n.2xlarge'|'c5n.4xlarge'|'c5n.9xlarge'|'c5n.18xlarge'|'c5n.metal'|'c6g.metal'|'c6g.medium'|'c6g.large'|'c6g.xlarge'|'c6g.2xlarge'|'c6g.4xlarge'|'c6g.8xlarge'|'c6g.12xlarge'|'c6g.16xlarge'|'c6gd.metal'|'c6gd.medium'|'c6gd.large'|'c6gd.xlarge'|'c6gd.2xlarge'|'c6gd.4xlarge'|'c6gd.8xlarge'|'c6gd.12xlarge'|'c6gd.16xlarge'|'c6gn.medium'|'c6gn.large'|'c6gn.xlarge'|'c6gn.2xlarge'|'c6gn.4xlarge'|'c6gn.8xlarge'|'c6gn.12xlarge'|'c6gn.16xlarge'|'cc1.4xlarge'|'cc2.8xlarge'|'g2.2xlarge'|'g2.8xlarge'|'g3.4xlarge'|'g3.8xlarge'|'g3.16xlarge'|'g3s.xlarge'|'g4ad.4xlarge'|'g4ad.8xlarge'|'g4ad.16xlarge'|'g4dn.xlarge'|'g4dn.2xlarge'|'g4dn.4xlarge'|'g4dn.8xlarge'|'g4dn.12xlarge'|'g4dn.16xlarge'|'g4dn.metal'|'cg1.4xlarge'|'p2.xlarge'|'p2.8xlarge'|'p2.16xlarge'|'p3.2xlarge'|'p3.8xlarge'|'p3.16xlarge'|'p3dn.24xlarge'|'p4d.24xlarge'|'d2.xlarge'|'d2.2xlarge'|'d2.4xlarge'|'d2.8xlarge'|'d3.xlarge'|'d3.2xlarge'|'d3.4xlarge'|'d3.8xlarge'|'d3en.xlarge'|'d3en.2xlarge'|'d3en.4xlarge'|'d3en.6xlarge'|'d3en.8xlarge'|'d3en.12xlarge'|'f1.2xlarge'|'f1.4xlarge'|'f1.16xlarge'|'m5.large'|'m5.xlarge'|'m5.2xlarge'|'m5.4xlarge'|'m5.8xlarge'|'m5.12xlarge'|'m5.16xlarge'|'m5.24xlarge'|'m5.metal'|'m5a.large'|'m5a.xlarge'|'m5a.2xlarge'|'m5a.4xlarge'|'m5a.8xlarge'|'m5a.12xlarge'|'m5a.16xlarge'|'m5a.24xlarge'|'m5d.large'|'m5d.xlarge'|'m5d.2xlarge'|'m5d.4xlarge'|'m5d.8xlarge'|'m5d.12xlarge'|'m5d.16xlarge'|'m5d.24xlarge'|'m5d.metal'|'m5ad.large'|'m5ad.xlarge'|'m5ad.2xlarge'|'m5ad.4xlarge'|'m5ad.8xlarge'|'m5ad.12xlarge'|'m5ad.16xlarge'|'m5ad.24xlarge'|'m5zn.large'|'m5zn.xlarge'|'m5zn.2xlarge'|'m5zn.3xlarge'|'m5zn.6xlarge'|'m5zn.12xlarge'|'m5zn.metal'|'h1.2xlarge'|'h1.4xlarge'|'h1.8xlarge'|'h1.16xlarge'|'z1d.large'|'z1d.xlarge'|'z1d.2xlarge'|'z1d.3xlarge'|'z1d.6xlarge'|'z1d.12xlarge'|'z1d.metal'|'u-6tb1.metal'|'u-9tb1.metal'|'u-12tb1.metal'|'u-18tb1.metal'|'u-24tb1.metal'|'a1.medium'|'a1.large'|'a1.xlarge'|'a1.2xlarge'|'a1.4xlarge'|'a1.metal'|'m5dn.large'|'m5dn.xlarge'|'m5dn.2xlarge'|'m5dn.4xlarge'|'m5dn.8xlarge'|'m5dn.12xlarge'|'m5dn.16xlarge'|'m5dn.24xlarge'|'m5n.large'|'m5n.xlarge'|'m5n.2xlarge'|'m5n.4xlarge'|'m5n.8xlarge'|'m5n.12xlarge'|'m5n.16xlarge'|'m5n.24xlarge'|'r5dn.large'|'r5dn.xlarge'|'r5dn.2xlarge'|'r5dn.4xlarge'|'r5dn.8xlarge'|'r5dn.12xlarge'|'r5dn.16xlarge'|'r5dn.24xlarge'|'r5n.large'|'r5n.xlarge'|'r5n.2xlarge'|'r5n.4xlarge'|'r5n.8xlarge'|'r5n.12xlarge'|'r5n.16xlarge'|'r5n.24xlarge'|'inf1.xlarge'|'inf1.2xlarge'|'inf1.6xlarge'|'inf1.24xlarge'|'m6g.metal'|'m6g.medium'|'m6g.large'|'m6g.xlarge'|'m6g.2xlarge'|'m6g.4xlarge'|'m6g.8xlarge'|'m6g.12xlarge'|'m6g.16xlarge'|'m6gd.metal'|'m6gd.medium'|'m6gd.large'|'m6gd.xlarge'|'m6gd.2xlarge'|'m6gd.4xlarge'|'m6gd.8xlarge'|'m6gd.12xlarge'|'m6gd.16xlarge'|'mac1.metal',
                    'KernelId': 'string',
                    'KeyName': 'string',
                    'LaunchTime': datetime(2015, 1, 1),
                    'Monitoring': {
                        'State': 'disabled'|'disabling'|'enabled'|'pending'
                    },
                    'Placement': {
                        'AvailabilityZone': 'string',
                        'Affinity': 'string',
                        'GroupName': 'string',
                        'PartitionNumber': 123,
                        'HostId': 'string',
                        'Tenancy': 'default'|'dedicated'|'host',
                        'SpreadDomain': 'string',
                        'HostResourceGroupArn': 'string'
                    },
                    'Platform': 'Windows',
                    'PrivateDnsName': 'string',
                    'PrivateIpAddress': 'string',
                    'ProductCodes': [
                        {
                            'ProductCodeId': 'string',
                            'ProductCodeType': 'devpay'|'marketplace'
                        },
                    ],
                    'PublicDnsName': 'string',
                    'PublicIpAddress': 'string',
                    'RamdiskId': 'string',
                    'State': {
                        'Code': 123,
                        'Name': 'pending'|'running'|'shutting-down'|'terminated'|'stopping'|'stopped'
                    },
                    'StateTransitionReason': 'string',
                    'SubnetId': 'string',
                    'VpcId': 'string',
                    'Architecture': 'i386'|'x86_64'|'arm64',
                    'BlockDeviceMappings': [
                        {
                            'DeviceName': 'string',
                            'Ebs': {
                                'AttachTime': datetime(2015, 1, 1),
                                'DeleteOnTermination': True|False,
                                'Status': 'attaching'|'attached'|'detaching'|'detached',
                                'VolumeId': 'string'
                            }
                        },
                    ],
                    'ClientToken': 'string',
                    'EbsOptimized': True|False,
                    'EnaSupport': True|False,
                    'Hypervisor': 'ovm'|'xen',
                    'IamInstanceProfile': {
                        'Arn': 'string',
                        'Id': 'string'
                    },
                    'InstanceLifecycle': 'spot'|'scheduled',
                    'ElasticGpuAssociations': [
                        {
                            'ElasticGpuId': 'string',
                            'ElasticGpuAssociationId': 'string',
                            'ElasticGpuAssociationState': 'string',
                            'ElasticGpuAssociationTime': 'string'
                        },
                    ],
                    'ElasticInferenceAcceleratorAssociations': [
                        {
                            'ElasticInferenceAcceleratorArn': 'string',
                            'ElasticInferenceAcceleratorAssociationId': 'string',
                            'ElasticInferenceAcceleratorAssociationState': 'string',
                            'ElasticInferenceAcceleratorAssociationTime': datetime(2015, 1, 1)
                        },
                    ],
                    'NetworkInterfaces': [
                        {
                            'Association': {
                                'CarrierIp': 'string',
                                'IpOwnerId': 'string',
                                'PublicDnsName': 'string',
                                'PublicIp': 'string'
                            },
                            'Attachment': {
                                'AttachTime': datetime(2015, 1, 1),
                                'AttachmentId': 'string',
                                'DeleteOnTermination': True|False,
                                'DeviceIndex': 123,
                                'Status': 'attaching'|'attached'|'detaching'|'detached',
                                'NetworkCardIndex': 123
                            },
                            'Description': 'string',
                            'Groups': [
                                {
                                    'GroupName': 'string',
                                    'GroupId': 'string'
                                },
                            ],
                            'Ipv6Addresses': [
                                {
                                    'Ipv6Address': 'string'
                                },
                            ],
                            'MacAddress': 'string',
                            'NetworkInterfaceId': 'string',
                            'OwnerId': 'string',
                            'PrivateDnsName': 'string',
                            'PrivateIpAddress': 'string',
                            'PrivateIpAddresses': [
                                {
                                    'Association': {
                                        'CarrierIp': 'string',
                                        'IpOwnerId': 'string',
                                        'PublicDnsName': 'string',
                                        'PublicIp': 'string'
                                    },
                                    'Primary': True|False,
                                    'PrivateDnsName': 'string',
                                    'PrivateIpAddress': 'string'
                                },
                            ],
                            'SourceDestCheck': True|False,
                            'Status': 'available'|'associated'|'attaching'|'in-use'|'detaching',
                            'SubnetId': 'string',
                            'VpcId': 'string',
                            'InterfaceType': 'string'
                        },
                    ],
                    'OutpostArn': 'string',
                    'RootDeviceName': 'string',
                    'RootDeviceType': 'ebs'|'instance-store',
                    'SecurityGroups': [
                        {
                            'GroupName': 'string',
                            'GroupId': 'string'
                        },
                    ],
                    'SourceDestCheck': True|False,
                    'SpotInstanceRequestId': 'string',
                    'SriovNetSupport': 'string',
                    'StateReason': {
                        'Code': 'string',
                        'Message': 'string'
                    },
                    'Tags': [
                        {
                            'Key': 'string',
                            'Value': 'string'
                        },
                    ],
                    'VirtualizationType': 'hvm'|'paravirtual',
                    'CpuOptions': {
                        'CoreCount': 123,
                        'ThreadsPerCore': 123
                    },
                    'CapacityReservationId': 'string',
                    'CapacityReservationSpecification': {
                        'CapacityReservationPreference': 'open'|'none',
                        'CapacityReservationTarget': {
                            'CapacityReservationId': 'string',
                            'CapacityReservationResourceGroupArn': 'string'
                        }
                    },
                    'HibernationOptions': {
                        'Configured': True|False
                    },
                    'Licenses': [
                        {
                            'LicenseConfigurationArn': 'string'
                        },
                    ],
                    'MetadataOptions': {
                        'State': 'pending'|'applied',
                        'HttpTokens': 'optional'|'required',
                        'HttpPutResponseHopLimit': 123,
                        'HttpEndpoint': 'disabled'|'enabled'
                    },
                    'EnclaveOptions': {
                        'Enabled': True|False
                    }
                },
            ],
            'OwnerId': 'string',
            'RequesterId': 'string',
            'ReservationId': 'string'
        },
    ],

}

**Response Structure**

- _(dict) --_

    - **Reservations** _(list) --_

        Information about the reservations.

        - _(dict) --_

            Describes a launch request for one or more instances, and includes owner, requester, and security group information that applies to all instances in the launch request.

            - **Groups** _(list) --_

                [EC2-Classic only] The security groups.

                - _(dict) --_

                    Describes a security group.

                    - **GroupName** _(string) --_

                        The name of the security group.

                    - **GroupId** _(string) --_

                        The ID of the security group.

            - **Instances** _(list) --_

                The instances.

                - _(dict) --_

                    Describes an instance.

                    - **AmiLaunchIndex** _(integer) --_

                        The AMI launch index, which can be used to find this instance in the launch group.

                    - **ImageId** _(string) --_

                        The ID of the AMI used to launch the instance.

                    - **InstanceId** _(string) --_

                        The ID of the instance.

                    - **InstanceType** _(string) --_

                        The instance type.

                    - **KernelId** _(string) --_

                        The kernel associated with this instance, if applicable.

                    - **KeyName** _(string) --_

                        The name of the key pair, if this instance was launched with an associated key pair.

                    - **LaunchTime** _(datetime) --_

                        The time the instance was launched.

                    - **Monitoring** _(dict) --_

                        The monitoring for the instance.

                        - **State** _(string) --_

                            Indicates whether detailed monitoring is enabled. Otherwise, basic monitoring is enabled.

                    - **Placement** _(dict) --_

                        The location where the instance launched, if applicable.

                        - **AvailabilityZone** _(string) --_

                            The Availability Zone of the instance.

                            If not specified, an Availability Zone will be automatically chosen for you based on the load balancing criteria for the Region.

                            This parameter is not supported by [CreateFleet](https://docs.aws.amazon.com/AWSEC2/latest/APIReference/API_CreateFleet) .

                        - **Affinity** _(string) --_

                            The affinity setting for the instance on the Dedicated Host. This parameter is not supported for the [ImportInstance](https://docs.aws.amazon.com/AWSEC2/latest/APIReference/API_ImportInstance.html) command.

                            This parameter is not supported by [CreateFleet](https://docs.aws.amazon.com/AWSEC2/latest/APIReference/API_CreateFleet) .

                        - **GroupName** _(string) --_

                            The name of the placement group the instance is in.

                        - **PartitionNumber** _(integer) --_

                            The number of the partition the instance is in. Valid only if the placement group strategy is set to partition .

                            This parameter is not supported by [CreateFleet](https://docs.aws.amazon.com/AWSEC2/latest/APIReference/API_CreateFleet) .

                        - **HostId** _(string) --_

                            The ID of the Dedicated Host on which the instance resides. This parameter is not supported for the [ImportInstance](https://docs.aws.amazon.com/AWSEC2/latest/APIReference/API_ImportInstance.html) command.

                            This parameter is not supported by [CreateFleet](https://docs.aws.amazon.com/AWSEC2/latest/APIReference/API_CreateFleet) .

                        - **Tenancy** _(string) --_

                            The tenancy of the instance (if the instance is running in a VPC). An instance with a tenancy of dedicated runs on single-tenant hardware. The host tenancy is not supported for the [ImportInstance](https://docs.aws.amazon.com/AWSEC2/latest/APIReference/API_ImportInstance.html) command.

                            This parameter is not supported by [CreateFleet](https://docs.aws.amazon.com/AWSEC2/latest/APIReference/API_CreateFleet) .

                        - **SpreadDomain** _(string) --_

                            Reserved for future use.

                            This parameter is not supported by [CreateFleet](https://docs.aws.amazon.com/AWSEC2/latest/APIReference/API_CreateFleet) .

                        - **HostResourceGroupArn** _(string) --_

                            The ARN of the host resource group in which to launch the instances. If you specify a host resource group ARN, omit the **Tenancy** parameter or set it to host .

                            This parameter is not supported by [CreateFleet](https://docs.aws.amazon.com/AWSEC2/latest/APIReference/API_CreateFleet) .

                    - **Platform** _(string) --_

                        The value is Windows for Windows instances; otherwise blank.

                    - **PrivateDnsName** _(string) --_

                        (IPv4 only) The private DNS hostname name assigned to the instance. This DNS hostname can only be used inside the Amazon EC2 network. This name is not available until the instance enters the running state.

                        [EC2-VPC] The Amazon-provided DNS server resolves Amazon-provided private DNS hostnames if you've enabled DNS resolution and DNS hostnames in your VPC. If you are not using the Amazon-provided DNS server in your VPC, your custom domain name servers must resolve the hostname as appropriate.

                    - **PrivateIpAddress** _(string) --_

                        The private IPv4 address assigned to the instance.

                    - **ProductCodes** _(list) --_

                        The product codes attached to this instance, if applicable.

                        - _(dict) --_

                            Describes a product code.

                            - **ProductCodeId** _(string) --_

                                The product code.

                            - **ProductCodeType** _(string) --_

                                The type of product code.

                    - **PublicDnsName** _(string) --_

                        (IPv4 only) The public DNS name assigned to the instance. This name is not available until the instance enters the running state. For EC2-VPC, this name is only available if you've enabled DNS hostnames for your VPC.

                    - **PublicIpAddress** _(string) --_

                        The public IPv4 address, or the Carrier IP address assigned to the instance, if applicable.

                        A Carrier IP address only applies to an instance launched in a subnet associated with a Wavelength Zone.

                    - **RamdiskId** _(string) --_

                        The RAM disk associated with this instance, if applicable.

                    - **State** _(dict) --_

                        The current state of the instance.

                        - **Code** _(integer) --_

                            The state of the instance as a 16-bit unsigned integer.

                            The high byte is all of the bits between 2^8 and (2^16)-1, which equals decimal values between 256 and 65,535. These numerical values are used for internal purposes and should be ignored.

                            The low byte is all of the bits between 2^0 and (2^8)-1, which equals decimal values between 0 and 255.

                            The valid values for instance-state-code will all be in the range of the low byte and they are:

                            - 0 : pending
                            - 16 : running
                            - 32 : shutting-down
                            - 48 : terminated
                            - 64 : stopping
                            - 80 : stopped

                            You can ignore the high byte value by zeroing out all of the bits above 2^8 or 256 in decimal.

                        - **Name** _(string) --_

                            The current state of the instance.

                    - **StateTransitionReason** _(string) --_

                        The reason for the most recent state transition. This might be an empty string.

                    - **SubnetId** _(string) --_

                        [EC2-VPC] The ID of the subnet in which the instance is running.

                    - **VpcId** _(string) --_

                        [EC2-VPC] The ID of the VPC in which the instance is running.

                    - **Architecture** _(string) --_

                        The architecture of the image.

                    - **BlockDeviceMappings** _(list) --_

                        Any block device mapping entries for the instance.

                        - _(dict) --_

                            Describes a block device mapping.

                            - **DeviceName** _(string) --_

                                The device name (for example, /dev/sdh or xvdh ).

                            - **Ebs** _(dict) --_

                                Parameters used to automatically set up EBS volumes when the instance is launched.

                                - **AttachTime** _(datetime) --_

                                    The time stamp when the attachment initiated.

                                - **DeleteOnTermination** _(boolean) --_

                                    Indicates whether the volume is deleted on instance termination.

                                - **Status** _(string) --_

                                    The attachment state.

                                - **VolumeId** _(string) --_

                                    The ID of the EBS volume.

                    - **ClientToken** _(string) --_

                        The idempotency token you provided when you launched the instance, if applicable.

                    - **EbsOptimized** _(boolean) --_

                        Indicates whether the instance is optimized for Amazon EBS I/O. This optimization provides dedicated throughput to Amazon EBS and an optimized configuration stack to provide optimal I/O performance. This optimization isn't available with all instance types. Additional usage charges apply when using an EBS Optimized instance.

                    - **EnaSupport** _(boolean) --_

                        Specifies whether enhanced networking with ENA is enabled.

                    - **Hypervisor** _(string) --_

                        The hypervisor type of the instance. The value xen is used for both Xen and Nitro hypervisors.

                    - **IamInstanceProfile** _(dict) --_

                        The IAM instance profile associated with the instance, if applicable.

                        - **Arn** _(string) --_

                            The Amazon Resource Name (ARN) of the instance profile.

                        - **Id** _(string) --_

                            The ID of the instance profile.

                    - **InstanceLifecycle** _(string) --_

                        Indicates whether this is a Spot Instance or a Scheduled Instance.

                    - **ElasticGpuAssociations** _(list) --_

                        The Elastic GPU associated with the instance.

                        - _(dict) --_

                            Describes the association between an instance and an Elastic Graphics accelerator.

                            - **ElasticGpuId** _(string) --_

                                The ID of the Elastic Graphics accelerator.

                            - **ElasticGpuAssociationId** _(string) --_

                                The ID of the association.

                            - **ElasticGpuAssociationState** _(string) --_

                                The state of the association between the instance and the Elastic Graphics accelerator.

                            - **ElasticGpuAssociationTime** _(string) --_

                                The time the Elastic Graphics accelerator was associated with the instance.

                    - **ElasticInferenceAcceleratorAssociations** _(list) --_

                        The elastic inference accelerator associated with the instance.

                        - _(dict) --_

                            Describes the association between an instance and an elastic inference accelerator.

                            - **ElasticInferenceAcceleratorArn** _(string) --_

                                The Amazon Resource Name (ARN) of the elastic inference accelerator.

                            - **ElasticInferenceAcceleratorAssociationId** _(string) --_

                                The ID of the association.

                            - **ElasticInferenceAcceleratorAssociationState** _(string) --_

                                The state of the elastic inference accelerator.

                            - **ElasticInferenceAcceleratorAssociationTime** _(datetime) --_

                                The time at which the elastic inference accelerator is associated with an instance.

                    - **NetworkInterfaces** _(list) --_

                        [EC2-VPC] The network interfaces for the instance.

                        - _(dict) --_

                            Describes a network interface.

                            - **Association** _(dict) --_

                                The association information for an Elastic IPv4 associated with the network interface.

                                - **CarrierIp** _(string) --_

                                    The carrier IP address associated with the network interface.

                                - **IpOwnerId** _(string) --_

                                    The ID of the owner of the Elastic IP address.

                                - **PublicDnsName** _(string) --_

                                    The public DNS name.

                                - **PublicIp** _(string) --_

                                    The public IP address or Elastic IP address bound to the network interface.

                            - **Attachment** _(dict) --_

                                The network interface attachment.

                                - **AttachTime** _(datetime) --_

                                    The time stamp when the attachment initiated.

                                - **AttachmentId** _(string) --_

                                    The ID of the network interface attachment.

                                - **DeleteOnTermination** _(boolean) --_

                                    Indicates whether the network interface is deleted when the instance is terminated.

                                - **DeviceIndex** _(integer) --_

                                    The index of the device on the instance for the network interface attachment.

                                - **Status** _(string) --_

                                    The attachment state.

                                - **NetworkCardIndex** _(integer) --_

                                    The index of the network card.

                            - **Description** _(string) --_

                                The description.

                            - **Groups** _(list) --_

                                One or more security groups.

                                - _(dict) --_

                                    Describes a security group.

                                    - **GroupName** _(string) --_

                                        The name of the security group.

                                    - **GroupId** _(string) --_

                                        The ID of the security group.

                            - **Ipv6Addresses** _(list) --_

                                One or more IPv6 addresses associated with the network interface.

                                - _(dict) --_

                                    Describes an IPv6 address.

                                    - **Ipv6Address** _(string) --_

                                        The IPv6 address.

                            - **MacAddress** _(string) --_

                                The MAC address.

                            - **NetworkInterfaceId** _(string) --_

                                The ID of the network interface.

                            - **OwnerId** _(string) --_

                                The ID of the AWS account that created the network interface.

                            - **PrivateDnsName** _(string) --_

                                The private DNS name.

                            - **PrivateIpAddress** _(string) --_

                                The IPv4 address of the network interface within the subnet.

                            - **PrivateIpAddresses** _(list) --_

                                One or more private IPv4 addresses associated with the network interface.

                                - _(dict) --_

                                    Describes a private IPv4 address.

                                    - **Association** _(dict) --_

                                        The association information for an Elastic IP address for the network interface.

                                        - **CarrierIp** _(string) --_

                                            The carrier IP address associated with the network interface.

                                        - **IpOwnerId** _(string) --_

                                            The ID of the owner of the Elastic IP address.

                                        - **PublicDnsName** _(string) --_

                                            The public DNS name.

                                        - **PublicIp** _(string) --_

                                            The public IP address or Elastic IP address bound to the network interface.

                                    - **Primary** _(boolean) --_

                                        Indicates whether this IPv4 address is the primary private IP address of the network interface.

                                    - **PrivateDnsName** _(string) --_

                                        The private IPv4 DNS name.

                                    - **PrivateIpAddress** _(string) --_

                                        The private IPv4 address of the network interface.

                            - **SourceDestCheck** _(boolean) --_

                                Indicates whether to validate network traffic to or from this network interface.

                            - **Status** _(string) --_

                                The status of the network interface.

                            - **SubnetId** _(string) --_

                                The ID of the subnet.

                            - **VpcId** _(string) --_

                                The ID of the VPC.

                            - **InterfaceType** _(string) --_

                                Describes the type of network interface.

                                Valid values: interface | efa

                    - **OutpostArn** _(string) --_

                        The Amazon Resource Name (ARN) of the Outpost.

                    - **RootDeviceName** _(string) --_

                        The device name of the root device volume (for example, /dev/sda1 ).

                    - **RootDeviceType** _(string) --_

                        The root device type used by the AMI. The AMI can use an EBS volume or an instance store volume.

                    - **SecurityGroups** _(list) --_

                        The security groups for the instance.

                        - _(dict) --_

                            Describes a security group.

                            - **GroupName** _(string) --_

                                The name of the security group.

                            - **GroupId** _(string) --_

                                The ID of the security group.

                    - **SourceDestCheck** _(boolean) --_

                        Specifies whether to enable an instance launched in a VPC to perform NAT. This controls whether source/destination checking is enabled on the instance. A value of true means that checking is enabled, and false means that checking is disabled. The value must be false for the instance to perform NAT. For more information, see [NAT Instances](https://docs.aws.amazon.com/AmazonVPC/latest/UserGuide/VPC_NAT_Instance.html) in the _Amazon Virtual Private Cloud User Guide_ .

                    - **SpotInstanceRequestId** _(string) --_

                        If the request is a Spot Instance request, the ID of the request.

                    - **SriovNetSupport** _(string) --_

                        Specifies whether enhanced networking with the Intel 82599 Virtual Function interface is enabled.

                    - **StateReason** _(dict) --_

                        The reason for the most recent state transition.

                        - **Code** _(string) --_

                            The reason code for the state change.

                        - **Message** _(string) --_

                            The message for the state change.

                            - Server.InsufficientInstanceCapacity : There was insufficient capacity available to satisfy the launch request.
                            - Server.InternalError : An internal error caused the instance to terminate during launch.
                            - Server.ScheduledStop : The instance was stopped due to a scheduled retirement.
                            - Server.SpotInstanceShutdown : The instance was stopped because the number of Spot requests with a maximum price equal to or higher than the Spot price exceeded available capacity or because of an increase in the Spot price.
                            - Server.SpotInstanceTermination : The instance was terminated because the number of Spot requests with a maximum price equal to or higher than the Spot price exceeded available capacity or because of an increase in the Spot price.
                            - Client.InstanceInitiatedShutdown : The instance was shut down using the shutdown -h command from the instance.
                            - Client.InstanceTerminated : The instance was terminated or rebooted during AMI creation.
                            - Client.InternalError : A client error caused the instance to terminate during launch.
                            - Client.InvalidSnapshot.NotFound : The specified snapshot was not found.
                            - Client.UserInitiatedHibernate : Hibernation was initiated on the instance.
                            - Client.UserInitiatedShutdown : The instance was shut down using the Amazon EC2 API.
                            - Client.VolumeLimitExceeded : The limit on the number of EBS volumes or total storage was exceeded. Decrease usage or request an increase in your account limits.
                    - **Tags** _(list) --_

                        Any tags assigned to the instance.

                        - _(dict) --_

                            Describes a tag.

                            - **Key** _(string) --_

                                The key of the tag.

                                Constraints: Tag keys are case-sensitive and accept a maximum of 127 Unicode characters. May not begin with aws: .

                            - **Value** _(string) --_

                                The value of the tag.

                                Constraints: Tag values are case-sensitive and accept a maximum of 255 Unicode characters.

                    - **VirtualizationType** _(string) --_

                        The virtualization type of the instance.

                    - **CpuOptions** _(dict) --_

                        The CPU options for the instance.

                        - **CoreCount** _(integer) --_

                            The number of CPU cores for the instance.

                        - **ThreadsPerCore** _(integer) --_

                            The number of threads per CPU core.

                    - **CapacityReservationId** _(string) --_

                        The ID of the Capacity Reservation.

                    - **CapacityReservationSpecification** _(dict) --_

                        Information about the Capacity Reservation targeting option.

                        - **CapacityReservationPreference** _(string) --_

                            Describes the instance's Capacity Reservation preferences. Possible preferences include:

                            - open - The instance can run in any open Capacity Reservation that has matching attributes (instance type, platform, Availability Zone).
                            - none - The instance avoids running in a Capacity Reservation even if one is available. The instance runs in On-Demand capacity.
                        - **CapacityReservationTarget** _(dict) --_

                            Information about the targeted Capacity Reservation or Capacity Reservation group.

                            - **CapacityReservationId** _(string) --_

                                The ID of the targeted Capacity Reservation.

                            - **CapacityReservationResourceGroupArn** _(string) --_

                                The ARN of the targeted Capacity Reservation group.

                    - **HibernationOptions** _(dict) --_

                        Indicates whether the instance is enabled for hibernation.

                        - **Configured** _(boolean) --_

                            If this parameter is set to true , your instance is enabled for hibernation; otherwise, it is not enabled for hibernation.

                    - **Licenses** _(list) --_

                        The license configurations.

                        - _(dict) --_

                            Describes a license configuration.

                            - **LicenseConfigurationArn** _(string) --_

                                The Amazon Resource Name (ARN) of the license configuration.

                    - **MetadataOptions** _(dict) --_

                        The metadata options for the instance.

                        - **State** _(string) --_

                            The state of the metadata option changes.

                            > pending - The metadata options are being updated and the instance is not ready to process metadata traffic with the new selection.
                            >
                            > applied - The metadata options have been successfully applied on the instance.

                        - **HttpTokens** _(string) --_

                            The state of token usage for your instance metadata requests. If the parameter is not specified in the request, the default state is optional .

                            If the state is optional , you can choose to retrieve instance metadata with or without a signed token header on your request. If you retrieve the IAM role credentials without a token, the version 1.0 role credentials are returned. If you retrieve the IAM role credentials using a valid signed token, the version 2.0 role credentials are returned.

                            If the state is required , you must send a signed token header with any instance metadata retrieval requests. In this state, retrieving the IAM role credential always returns the version 2.0 credentials; the version 1.0 credentials are not available.

                        - **HttpPutResponseHopLimit** _(integer) --_

                            The desired HTTP PUT response hop limit for instance metadata requests. The larger the number, the further instance metadata requests can travel.

                            Default: 1

                            Possible values: Integers from 1 to 64

                        - **HttpEndpoint** _(string) --_

                            This parameter enables or disables the HTTP metadata endpoint on your instances. If the parameter is not specified, the default state is enabled .

                            Note

                            If you specify a value of disabled , you will not be able to access your instance metadata.

                    - **EnclaveOptions** _(dict) --_

                        Indicates whether the instance is enabled for AWS Nitro Enclaves.

                        - **Enabled** _(boolean) --_

                            If this parameter is set to true , the instance is enabled for AWS Nitro Enclaves; otherwise, it is not enabled for AWS Nitro Enclaves.

            - **OwnerId** _(string) --_

                The ID of the AWS account that owns the reservation.

            - **RequesterId** _(string) --_

                The ID of the requester that launched the instances on your behalf (for example, AWS Management Console or Auto Scaling).

            - **ReservationId** _(string) --_

                The ID of the reservation.


_class_ EC2.Paginator.DescribeInternetGateways

paginator = client.get_paginator('describe_internet_gateways')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_internet_gateways()](#EC2.Client.describe_internet_gateways "EC2.Client.describe_internet_gateways").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeInternetGateways)

**Request Syntax**

response_iterator = paginator.paginate(
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    DryRun=True|False,
    InternetGatewayIds=[
        'string',
    ],
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **Filters** (_list_) --

    One or more filters.

    - attachment.state - The current state of the attachment between the gateway and the VPC (available ). Present only if a VPC is attached.
    - attachment.vpc-id - The ID of an attached VPC.
    - internet-gateway-id - The ID of the Internet gateway.
    - owner-id - The ID of the AWS account that owns the internet gateway.
    - tag :<key> - The key/value combination of a tag assigned to the resource. Use the tag key in the filter name and the tag value as the filter value. For example, to find all resources that have a tag with the key Owner and the value TeamA , specify tag:Owner for the filter name and TeamA for the filter value.
    - tag-key - The key of a tag assigned to the resource. Use this filter to find all resources assigned a tag with a specific key, regardless of the tag value.

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **InternetGatewayIds** (_list_) --

    One or more internet gateway IDs.

    Default: Describes all your internet gateways.

    - _(string) --_
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'InternetGateways': [
        {
            'Attachments': [
                {
                    'State': 'attaching'|'attached'|'detaching'|'detached',
                    'VpcId': 'string'
                },
            ],
            'InternetGatewayId': 'string',
            'OwnerId': 'string',
            'Tags': [
                {
                    'Key': 'string',
                    'Value': 'string'
                },
            ]
        },
    ],

}

**Response Structure**

- _(dict) --_

    - **InternetGateways** _(list) --_

        Information about one or more internet gateways.

        - _(dict) --_

            Describes an internet gateway.

            - **Attachments** _(list) --_

                Any VPCs attached to the internet gateway.

                - _(dict) --_

                    Describes the attachment of a VPC to an internet gateway or an egress-only internet gateway.

                    - **State** _(string) --_

                        The current state of the attachment. For an internet gateway, the state is available when attached to a VPC; otherwise, this value is not returned.

                    - **VpcId** _(string) --_

                        The ID of the VPC.

            - **InternetGatewayId** _(string) --_

                The ID of the internet gateway.

            - **OwnerId** _(string) --_

                The ID of the AWS account that owns the internet gateway.

            - **Tags** _(list) --_

                Any tags assigned to the internet gateway.

                - _(dict) --_

                    Describes a tag.

                    - **Key** _(string) --_

                        The key of the tag.

                        Constraints: Tag keys are case-sensitive and accept a maximum of 127 Unicode characters. May not begin with aws: .

                    - **Value** _(string) --_

                        The value of the tag.

                        Constraints: Tag values are case-sensitive and accept a maximum of 255 Unicode characters.


_class_ EC2.Paginator.DescribeIpv6Pools

paginator = client.get_paginator('describe_ipv6_pools')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_ipv6_pools()](#EC2.Client.describe_ipv6_pools "EC2.Client.describe_ipv6_pools").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeIpv6Pools)

**Request Syntax**

response_iterator = paginator.paginate(
    PoolIds=[
        'string',
    ],
    DryRun=True|False,
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **PoolIds** (_list_) --

    The IDs of the IPv6 address pools.

    - _(string) --_
- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **Filters** (_list_) --

    One or more filters.

    - tag :<key> - The key/value combination of a tag assigned to the resource. Use the tag key in the filter name and the tag value as the filter value. For example, to find all resources that have a tag with the key Owner and the value TeamA , specify tag:Owner for the filter name and TeamA for the filter value.
    - tag-key - The key of a tag assigned to the resource. Use this filter to find all resources assigned a tag with a specific key, regardless of the tag value.

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'Ipv6Pools': [
        {
            'PoolId': 'string',
            'Description': 'string',
            'PoolCidrBlocks': [
                {
                    'Cidr': 'string'
                },
            ],
            'Tags': [
                {
                    'Key': 'string',
                    'Value': 'string'
                },
            ]
        },
    ],

}

**Response Structure**

- _(dict) --_

    - **Ipv6Pools** _(list) --_

        Information about the IPv6 address pools.

        - _(dict) --_

            Describes an IPv6 address pool.

            - **PoolId** _(string) --_

                The ID of the address pool.

            - **Description** _(string) --_

                The description for the address pool.

            - **PoolCidrBlocks** _(list) --_

                The CIDR blocks for the address pool.

                - _(dict) --_

                    Describes a CIDR block for an address pool.

                    - **Cidr** _(string) --_

                        The CIDR block.

            - **Tags** _(list) --_

                Any tags for the address pool.

                - _(dict) --_

                    Describes a tag.

                    - **Key** _(string) --_

                        The key of the tag.

                        Constraints: Tag keys are case-sensitive and accept a maximum of 127 Unicode characters. May not begin with aws: .

                    - **Value** _(string) --_

                        The value of the tag.

                        Constraints: Tag values are case-sensitive and accept a maximum of 255 Unicode characters.


_class_ EC2.Paginator.DescribeLaunchTemplateVersions

paginator = client.get_paginator('describe_launch_template_versions')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_launch_template_versions()](#EC2.Client.describe_launch_template_versions "EC2.Client.describe_launch_template_versions").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeLaunchTemplateVersions)

**Request Syntax**

response_iterator = paginator.paginate(
    DryRun=True|False,
    LaunchTemplateId='string',
    LaunchTemplateName='string',
    Versions=[
        'string',
    ],
    MinVersion='string',
    MaxVersion='string',
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **LaunchTemplateId** (_string_) -- The ID of the launch template. To describe one or more versions of a specified launch template, you must specify either the launch template ID or the launch template name in the request. To describe all the latest or default launch template versions in your account, you must omit this parameter.
- **LaunchTemplateName** (_string_) -- The name of the launch template. To describe one or more versions of a specified launch template, you must specify either the launch template ID or the launch template name in the request. To describe all the latest or default launch template versions in your account, you must omit this parameter.
- **Versions** (_list_) --

    One or more versions of the launch template. Valid values depend on whether you are describing a specified launch template (by ID or name) or all launch templates in your account.

    To describe one or more versions of a specified launch template, valid values are $Latest , $Default , and numbers.

    To describe all launch templates in your account that are defined as the latest version, the valid value is $Latest . To describe all launch templates in your account that are defined as the default version, the valid value is $Default . You can specify $Latest and $Default in the same call. You cannot specify numbers.

    - _(string) --_
- **MinVersion** (_string_) -- The version number after which to describe launch template versions.
- **MaxVersion** (_string_) -- The version number up to which to describe launch template versions.
- **Filters** (_list_) --

    One or more filters.

    - create-time - The time the launch template version was created.
    - ebs-optimized - A boolean that indicates whether the instance is optimized for Amazon EBS I/O.
    - iam-instance-profile - The ARN of the IAM instance profile.
    - image-id - The ID of the AMI.
    - instance-type - The instance type.
    - is-default-version - A boolean that indicates whether the launch template version is the default version.
    - kernel-id - The kernel ID.
    - ram-disk-id - The RAM disk ID.

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'LaunchTemplateVersions': [
        {
            'LaunchTemplateId': 'string',
            'LaunchTemplateName': 'string',
            'VersionNumber': 123,
            'VersionDescription': 'string',
            'CreateTime': datetime(2015, 1, 1),
            'CreatedBy': 'string',
            'DefaultVersion': True|False,
            'LaunchTemplateData': {
                'KernelId': 'string',
                'EbsOptimized': True|False,
                'IamInstanceProfile': {
                    'Arn': 'string',
                    'Name': 'string'
                },
                'BlockDeviceMappings': [
                    {
                        'DeviceName': 'string',
                        'VirtualName': 'string',
                        'Ebs': {
                            'Encrypted': True|False,
                            'DeleteOnTermination': True|False,
                            'Iops': 123,
                            'KmsKeyId': 'string',
                            'SnapshotId': 'string',
                            'VolumeSize': 123,
                            'VolumeType': 'standard'|'io1'|'io2'|'gp2'|'sc1'|'st1'|'gp3',
                            'Throughput': 123
                        },
                        'NoDevice': 'string'
                    },
                ],
                'NetworkInterfaces': [
                    {
                        'AssociateCarrierIpAddress': True|False,
                        'AssociatePublicIpAddress': True|False,
                        'DeleteOnTermination': True|False,
                        'Description': 'string',
                        'DeviceIndex': 123,
                        'Groups': [
                            'string',
                        ],
                        'InterfaceType': 'string',
                        'Ipv6AddressCount': 123,
                        'Ipv6Addresses': [
                            {
                                'Ipv6Address': 'string'
                            },
                        ],
                        'NetworkInterfaceId': 'string',
                        'PrivateIpAddress': 'string',
                        'PrivateIpAddresses': [
                            {
                                'Primary': True|False,
                                'PrivateIpAddress': 'string'
                            },
                        ],
                        'SecondaryPrivateIpAddressCount': 123,
                        'SubnetId': 'string',
                        'NetworkCardIndex': 123
                    },
                ],
                'ImageId': 'string',
                'InstanceType': 't1.micro'|'t2.nano'|'t2.micro'|'t2.small'|'t2.medium'|'t2.large'|'t2.xlarge'|'t2.2xlarge'|'t3.nano'|'t3.micro'|'t3.small'|'t3.medium'|'t3.large'|'t3.xlarge'|'t3.2xlarge'|'t3a.nano'|'t3a.micro'|'t3a.small'|'t3a.medium'|'t3a.large'|'t3a.xlarge'|'t3a.2xlarge'|'t4g.nano'|'t4g.micro'|'t4g.small'|'t4g.medium'|'t4g.large'|'t4g.xlarge'|'t4g.2xlarge'|'m1.small'|'m1.medium'|'m1.large'|'m1.xlarge'|'m3.medium'|'m3.large'|'m3.xlarge'|'m3.2xlarge'|'m4.large'|'m4.xlarge'|'m4.2xlarge'|'m4.4xlarge'|'m4.10xlarge'|'m4.16xlarge'|'m2.xlarge'|'m2.2xlarge'|'m2.4xlarge'|'cr1.8xlarge'|'r3.large'|'r3.xlarge'|'r3.2xlarge'|'r3.4xlarge'|'r3.8xlarge'|'r4.large'|'r4.xlarge'|'r4.2xlarge'|'r4.4xlarge'|'r4.8xlarge'|'r4.16xlarge'|'r5.large'|'r5.xlarge'|'r5.2xlarge'|'r5.4xlarge'|'r5.8xlarge'|'r5.12xlarge'|'r5.16xlarge'|'r5.24xlarge'|'r5.metal'|'r5a.large'|'r5a.xlarge'|'r5a.2xlarge'|'r5a.4xlarge'|'r5a.8xlarge'|'r5a.12xlarge'|'r5a.16xlarge'|'r5a.24xlarge'|'r5b.large'|'r5b.xlarge'|'r5b.2xlarge'|'r5b.4xlarge'|'r5b.8xlarge'|'r5b.12xlarge'|'r5b.16xlarge'|'r5b.24xlarge'|'r5b.metal'|'r5d.large'|'r5d.xlarge'|'r5d.2xlarge'|'r5d.4xlarge'|'r5d.8xlarge'|'r5d.12xlarge'|'r5d.16xlarge'|'r5d.24xlarge'|'r5d.metal'|'r5ad.large'|'r5ad.xlarge'|'r5ad.2xlarge'|'r5ad.4xlarge'|'r5ad.8xlarge'|'r5ad.12xlarge'|'r5ad.16xlarge'|'r5ad.24xlarge'|'r6g.metal'|'r6g.medium'|'r6g.large'|'r6g.xlarge'|'r6g.2xlarge'|'r6g.4xlarge'|'r6g.8xlarge'|'r6g.12xlarge'|'r6g.16xlarge'|'r6gd.metal'|'r6gd.medium'|'r6gd.large'|'r6gd.xlarge'|'r6gd.2xlarge'|'r6gd.4xlarge'|'r6gd.8xlarge'|'r6gd.12xlarge'|'r6gd.16xlarge'|'x1.16xlarge'|'x1.32xlarge'|'x1e.xlarge'|'x1e.2xlarge'|'x1e.4xlarge'|'x1e.8xlarge'|'x1e.16xlarge'|'x1e.32xlarge'|'i2.xlarge'|'i2.2xlarge'|'i2.4xlarge'|'i2.8xlarge'|'i3.large'|'i3.xlarge'|'i3.2xlarge'|'i3.4xlarge'|'i3.8xlarge'|'i3.16xlarge'|'i3.metal'|'i3en.large'|'i3en.xlarge'|'i3en.2xlarge'|'i3en.3xlarge'|'i3en.6xlarge'|'i3en.12xlarge'|'i3en.24xlarge'|'i3en.metal'|'hi1.4xlarge'|'hs1.8xlarge'|'c1.medium'|'c1.xlarge'|'c3.large'|'c3.xlarge'|'c3.2xlarge'|'c3.4xlarge'|'c3.8xlarge'|'c4.large'|'c4.xlarge'|'c4.2xlarge'|'c4.4xlarge'|'c4.8xlarge'|'c5.large'|'c5.xlarge'|'c5.2xlarge'|'c5.4xlarge'|'c5.9xlarge'|'c5.12xlarge'|'c5.18xlarge'|'c5.24xlarge'|'c5.metal'|'c5a.large'|'c5a.xlarge'|'c5a.2xlarge'|'c5a.4xlarge'|'c5a.8xlarge'|'c5a.12xlarge'|'c5a.16xlarge'|'c5a.24xlarge'|'c5ad.large'|'c5ad.xlarge'|'c5ad.2xlarge'|'c5ad.4xlarge'|'c5ad.8xlarge'|'c5ad.12xlarge'|'c5ad.16xlarge'|'c5ad.24xlarge'|'c5d.large'|'c5d.xlarge'|'c5d.2xlarge'|'c5d.4xlarge'|'c5d.9xlarge'|'c5d.12xlarge'|'c5d.18xlarge'|'c5d.24xlarge'|'c5d.metal'|'c5n.large'|'c5n.xlarge'|'c5n.2xlarge'|'c5n.4xlarge'|'c5n.9xlarge'|'c5n.18xlarge'|'c5n.metal'|'c6g.metal'|'c6g.medium'|'c6g.large'|'c6g.xlarge'|'c6g.2xlarge'|'c6g.4xlarge'|'c6g.8xlarge'|'c6g.12xlarge'|'c6g.16xlarge'|'c6gd.metal'|'c6gd.medium'|'c6gd.large'|'c6gd.xlarge'|'c6gd.2xlarge'|'c6gd.4xlarge'|'c6gd.8xlarge'|'c6gd.12xlarge'|'c6gd.16xlarge'|'c6gn.medium'|'c6gn.large'|'c6gn.xlarge'|'c6gn.2xlarge'|'c6gn.4xlarge'|'c6gn.8xlarge'|'c6gn.12xlarge'|'c6gn.16xlarge'|'cc1.4xlarge'|'cc2.8xlarge'|'g2.2xlarge'|'g2.8xlarge'|'g3.4xlarge'|'g3.8xlarge'|'g3.16xlarge'|'g3s.xlarge'|'g4ad.4xlarge'|'g4ad.8xlarge'|'g4ad.16xlarge'|'g4dn.xlarge'|'g4dn.2xlarge'|'g4dn.4xlarge'|'g4dn.8xlarge'|'g4dn.12xlarge'|'g4dn.16xlarge'|'g4dn.metal'|'cg1.4xlarge'|'p2.xlarge'|'p2.8xlarge'|'p2.16xlarge'|'p3.2xlarge'|'p3.8xlarge'|'p3.16xlarge'|'p3dn.24xlarge'|'p4d.24xlarge'|'d2.xlarge'|'d2.2xlarge'|'d2.4xlarge'|'d2.8xlarge'|'d3.xlarge'|'d3.2xlarge'|'d3.4xlarge'|'d3.8xlarge'|'d3en.xlarge'|'d3en.2xlarge'|'d3en.4xlarge'|'d3en.6xlarge'|'d3en.8xlarge'|'d3en.12xlarge'|'f1.2xlarge'|'f1.4xlarge'|'f1.16xlarge'|'m5.large'|'m5.xlarge'|'m5.2xlarge'|'m5.4xlarge'|'m5.8xlarge'|'m5.12xlarge'|'m5.16xlarge'|'m5.24xlarge'|'m5.metal'|'m5a.large'|'m5a.xlarge'|'m5a.2xlarge'|'m5a.4xlarge'|'m5a.8xlarge'|'m5a.12xlarge'|'m5a.16xlarge'|'m5a.24xlarge'|'m5d.large'|'m5d.xlarge'|'m5d.2xlarge'|'m5d.4xlarge'|'m5d.8xlarge'|'m5d.12xlarge'|'m5d.16xlarge'|'m5d.24xlarge'|'m5d.metal'|'m5ad.large'|'m5ad.xlarge'|'m5ad.2xlarge'|'m5ad.4xlarge'|'m5ad.8xlarge'|'m5ad.12xlarge'|'m5ad.16xlarge'|'m5ad.24xlarge'|'m5zn.large'|'m5zn.xlarge'|'m5zn.2xlarge'|'m5zn.3xlarge'|'m5zn.6xlarge'|'m5zn.12xlarge'|'m5zn.metal'|'h1.2xlarge'|'h1.4xlarge'|'h1.8xlarge'|'h1.16xlarge'|'z1d.large'|'z1d.xlarge'|'z1d.2xlarge'|'z1d.3xlarge'|'z1d.6xlarge'|'z1d.12xlarge'|'z1d.metal'|'u-6tb1.metal'|'u-9tb1.metal'|'u-12tb1.metal'|'u-18tb1.metal'|'u-24tb1.metal'|'a1.medium'|'a1.large'|'a1.xlarge'|'a1.2xlarge'|'a1.4xlarge'|'a1.metal'|'m5dn.large'|'m5dn.xlarge'|'m5dn.2xlarge'|'m5dn.4xlarge'|'m5dn.8xlarge'|'m5dn.12xlarge'|'m5dn.16xlarge'|'m5dn.24xlarge'|'m5n.large'|'m5n.xlarge'|'m5n.2xlarge'|'m5n.4xlarge'|'m5n.8xlarge'|'m5n.12xlarge'|'m5n.16xlarge'|'m5n.24xlarge'|'r5dn.large'|'r5dn.xlarge'|'r5dn.2xlarge'|'r5dn.4xlarge'|'r5dn.8xlarge'|'r5dn.12xlarge'|'r5dn.16xlarge'|'r5dn.24xlarge'|'r5n.large'|'r5n.xlarge'|'r5n.2xlarge'|'r5n.4xlarge'|'r5n.8xlarge'|'r5n.12xlarge'|'r5n.16xlarge'|'r5n.24xlarge'|'inf1.xlarge'|'inf1.2xlarge'|'inf1.6xlarge'|'inf1.24xlarge'|'m6g.metal'|'m6g.medium'|'m6g.large'|'m6g.xlarge'|'m6g.2xlarge'|'m6g.4xlarge'|'m6g.8xlarge'|'m6g.12xlarge'|'m6g.16xlarge'|'m6gd.metal'|'m6gd.medium'|'m6gd.large'|'m6gd.xlarge'|'m6gd.2xlarge'|'m6gd.4xlarge'|'m6gd.8xlarge'|'m6gd.12xlarge'|'m6gd.16xlarge'|'mac1.metal',
                'KeyName': 'string',
                'Monitoring': {
                    'Enabled': True|False
                },
                'Placement': {
                    'AvailabilityZone': 'string',
                    'Affinity': 'string',
                    'GroupName': 'string',
                    'HostId': 'string',
                    'Tenancy': 'default'|'dedicated'|'host',
                    'SpreadDomain': 'string',
                    'HostResourceGroupArn': 'string',
                    'PartitionNumber': 123
                },
                'RamDiskId': 'string',
                'DisableApiTermination': True|False,
                'InstanceInitiatedShutdownBehavior': 'stop'|'terminate',
                'UserData': 'string',
                'TagSpecifications': [
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
                'ElasticGpuSpecifications': [
                    {
                        'Type': 'string'
                    },
                ],
                'ElasticInferenceAccelerators': [
                    {
                        'Type': 'string',
                        'Count': 123
                    },
                ],
                'SecurityGroupIds': [
                    'string',
                ],
                'SecurityGroups': [
                    'string',
                ],
                'InstanceMarketOptions': {
                    'MarketType': 'spot',
                    'SpotOptions': {
                        'MaxPrice': 'string',
                        'SpotInstanceType': 'one-time'|'persistent',
                        'BlockDurationMinutes': 123,
                        'ValidUntil': datetime(2015, 1, 1),
                        'InstanceInterruptionBehavior': 'hibernate'|'stop'|'terminate'
                    }
                },
                'CreditSpecification': {
                    'CpuCredits': 'string'
                },
                'CpuOptions': {
                    'CoreCount': 123,
                    'ThreadsPerCore': 123
                },
                'CapacityReservationSpecification': {
                    'CapacityReservationPreference': 'open'|'none',
                    'CapacityReservationTarget': {
                        'CapacityReservationId': 'string',
                        'CapacityReservationResourceGroupArn': 'string'
                    }
                },
                'LicenseSpecifications': [
                    {
                        'LicenseConfigurationArn': 'string'
                    },
                ],
                'HibernationOptions': {
                    'Configured': True|False
                },
                'MetadataOptions': {
                    'State': 'pending'|'applied',
                    'HttpTokens': 'optional'|'required',
                    'HttpPutResponseHopLimit': 123,
                    'HttpEndpoint': 'disabled'|'enabled'
                },
                'EnclaveOptions': {
                    'Enabled': True|False
                }
            }
        },
    ],

}

**Response Structure**

- _(dict) --_

    - **LaunchTemplateVersions** _(list) --_

        Information about the launch template versions.

        - _(dict) --_

            Describes a launch template version.

            - **LaunchTemplateId** _(string) --_

                The ID of the launch template.

            - **LaunchTemplateName** _(string) --_

                The name of the launch template.

            - **VersionNumber** _(integer) --_

                The version number.

            - **VersionDescription** _(string) --_

                The description for the version.

            - **CreateTime** _(datetime) --_

                The time the version was created.

            - **CreatedBy** _(string) --_

                The principal that created the version.

            - **DefaultVersion** _(boolean) --_

                Indicates whether the version is the default version.

            - **LaunchTemplateData** _(dict) --_

                Information about the launch template.

                - **KernelId** _(string) --_

                    The ID of the kernel, if applicable.

                - **EbsOptimized** _(boolean) --_

                    Indicates whether the instance is optimized for Amazon EBS I/O.

                - **IamInstanceProfile** _(dict) --_

                    The IAM instance profile.

                    - **Arn** _(string) --_

                        The Amazon Resource Name (ARN) of the instance profile.

                    - **Name** _(string) --_

                        The name of the instance profile.

                - **BlockDeviceMappings** _(list) --_

                    The block device mappings.

                    - _(dict) --_

                        Describes a block device mapping.

                        - **DeviceName** _(string) --_

                            The device name.

                        - **VirtualName** _(string) --_

                            The virtual device name (ephemeralN).

                        - **Ebs** _(dict) --_

                            Information about the block device for an EBS volume.

                            - **Encrypted** _(boolean) --_

                                Indicates whether the EBS volume is encrypted.

                            - **DeleteOnTermination** _(boolean) --_

                                Indicates whether the EBS volume is deleted on instance termination.

                            - **Iops** _(integer) --_

                                The number of I/O operations per second (IOPS) that the volume supports.

                            - **KmsKeyId** _(string) --_

                                The ARN of the AWS Key Management Service (AWS KMS) CMK used for encryption.

                            - **SnapshotId** _(string) --_

                                The ID of the snapshot.

                            - **VolumeSize** _(integer) --_

                                The size of the volume, in GiB.

                            - **VolumeType** _(string) --_

                                The volume type.

                            - **Throughput** _(integer) --_

                                The throughput that the volume supports, in MiB/s.

                        - **NoDevice** _(string) --_

                            Suppresses the specified device included in the block device mapping of the AMI.

                - **NetworkInterfaces** _(list) --_

                    The network interfaces.

                    - _(dict) --_

                        Describes a network interface.

                        - **AssociateCarrierIpAddress** _(boolean) --_

                            Indicates whether to associate a Carrier IP address with eth0 for a new network interface.

                            Use this option when you launch an instance in a Wavelength Zone and want to associate a Carrier IP address with the network interface. For more information about Carrier IP addresses, see [Carrier IP addresses](https://docs.aws.amazon.com/wavelength/latest/developerguide/how-wavelengths-work.html#provider-owned-ip) in the _AWS Wavelength Developer Guide_ .

                        - **AssociatePublicIpAddress** _(boolean) --_

                            Indicates whether to associate a public IPv4 address with eth0 for a new network interface.

                        - **DeleteOnTermination** _(boolean) --_

                            Indicates whether the network interface is deleted when the instance is terminated.

                        - **Description** _(string) --_

                            A description for the network interface.

                        - **DeviceIndex** _(integer) --_

                            The device index for the network interface attachment.

                        - **Groups** _(list) --_

                            The IDs of one or more security groups.

                            - _(string) --_
                        - **InterfaceType** _(string) --_

                            The type of network interface.

                        - **Ipv6AddressCount** _(integer) --_

                            The number of IPv6 addresses for the network interface.

                        - **Ipv6Addresses** _(list) --_

                            The IPv6 addresses for the network interface.

                            - _(dict) --_

                                Describes an IPv6 address.

                                - **Ipv6Address** _(string) --_

                                    The IPv6 address.

                        - **NetworkInterfaceId** _(string) --_

                            The ID of the network interface.

                        - **PrivateIpAddress** _(string) --_

                            The primary private IPv4 address of the network interface.

                        - **PrivateIpAddresses** _(list) --_

                            One or more private IPv4 addresses.

                            - _(dict) --_

                                Describes a secondary private IPv4 address for a network interface.

                                - **Primary** _(boolean) --_

                                    Indicates whether the private IPv4 address is the primary private IPv4 address. Only one IPv4 address can be designated as primary.

                                - **PrivateIpAddress** _(string) --_

                                    The private IPv4 addresses.

                        - **SecondaryPrivateIpAddressCount** _(integer) --_

                            The number of secondary private IPv4 addresses for the network interface.

                        - **SubnetId** _(string) --_

                            The ID of the subnet for the network interface.

                        - **NetworkCardIndex** _(integer) --_

                            The index of the network card.

                - **ImageId** _(string) --_

                    The ID of the AMI that was used to launch the instance.

                - **InstanceType** _(string) --_

                    The instance type.

                - **KeyName** _(string) --_

                    The name of the key pair.

                - **Monitoring** _(dict) --_

                    The monitoring for the instance.

                    - **Enabled** _(boolean) --_

                        Indicates whether detailed monitoring is enabled. Otherwise, basic monitoring is enabled.

                - **Placement** _(dict) --_

                    The placement of the instance.

                    - **AvailabilityZone** _(string) --_

                        The Availability Zone of the instance.

                    - **Affinity** _(string) --_

                        The affinity setting for the instance on the Dedicated Host.

                    - **GroupName** _(string) --_

                        The name of the placement group for the instance.

                    - **HostId** _(string) --_

                        The ID of the Dedicated Host for the instance.

                    - **Tenancy** _(string) --_

                        The tenancy of the instance (if the instance is running in a VPC). An instance with a tenancy of dedicated runs on single-tenant hardware.

                    - **SpreadDomain** _(string) --_

                        Reserved for future use.

                    - **HostResourceGroupArn** _(string) --_

                        The ARN of the host resource group in which to launch the instances.

                    - **PartitionNumber** _(integer) --_

                        The number of the partition the instance should launch in. Valid only if the placement group strategy is set to partition .

                - **RamDiskId** _(string) --_

                    The ID of the RAM disk, if applicable.

                - **DisableApiTermination** _(boolean) --_

                    If set to true , indicates that the instance cannot be terminated using the Amazon EC2 console, command line tool, or API.

                - **InstanceInitiatedShutdownBehavior** _(string) --_

                    Indicates whether an instance stops or terminates when you initiate shutdown from the instance (using the operating system command for system shutdown).

                - **UserData** _(string) --_

                    The user data for the instance.

                - **TagSpecifications** _(list) --_

                    The tags.

                    - _(dict) --_

                        The tag specification for the launch template.

                        - **ResourceType** _(string) --_

                            The type of resource.

                        - **Tags** _(list) --_

                            The tags for the resource.

                            - _(dict) --_

                                Describes a tag.

                                - **Key** _(string) --_

                                    The key of the tag.

                                    Constraints: Tag keys are case-sensitive and accept a maximum of 127 Unicode characters. May not begin with aws: .

                                - **Value** _(string) --_

                                    The value of the tag.

                                    Constraints: Tag values are case-sensitive and accept a maximum of 255 Unicode characters.

                - **ElasticGpuSpecifications** _(list) --_

                    The elastic GPU specification.

                    - _(dict) --_

                        Describes an elastic GPU.

                        - **Type** _(string) --_

                            The elastic GPU type.

                - **ElasticInferenceAccelerators** _(list) --_

                    The elastic inference accelerator for the instance.

                    - _(dict) --_

                        Describes an elastic inference accelerator.

                        - **Type** _(string) --_

                            The type of elastic inference accelerator. The possible values are eia1.medium, eia1.large, and eia1.xlarge.

                        - **Count** _(integer) --_

                            The number of elastic inference accelerators to attach to the instance.

                            Default: 1

                - **SecurityGroupIds** _(list) --_

                    The security group IDs.

                    - _(string) --_
                - **SecurityGroups** _(list) --_

                    The security group names.

                    - _(string) --_
                - **InstanceMarketOptions** _(dict) --_

                    The market (purchasing) option for the instances.

                    - **MarketType** _(string) --_

                        The market type.

                    - **SpotOptions** _(dict) --_

                        The options for Spot Instances.

                        - **MaxPrice** _(string) --_

                            The maximum hourly price you're willing to pay for the Spot Instances.

                        - **SpotInstanceType** _(string) --_

                            The Spot Instance request type.

                        - **BlockDurationMinutes** _(integer) --_

                            The required duration for the Spot Instances (also known as Spot blocks), in minutes. This value must be a multiple of 60 (60, 120, 180, 240, 300, or 360).

                        - **ValidUntil** _(datetime) --_

                            The end date of the request. For a one-time request, the request remains active until all instances launch, the request is canceled, or this date is reached. If the request is persistent, it remains active until it is canceled or this date and time is reached.

                        - **InstanceInterruptionBehavior** _(string) --_

                            The behavior when a Spot Instance is interrupted.

                - **CreditSpecification** _(dict) --_

                    The credit option for CPU usage of the instance.

                    - **CpuCredits** _(string) --_

                        The credit option for CPU usage of a T2, T3, or T3a instance. Valid values are standard and unlimited .

                - **CpuOptions** _(dict) --_

                    The CPU options for the instance. For more information, see [Optimizing CPU Options](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/instance-optimize-cpu.html) in the _Amazon Elastic Compute Cloud User Guide_ .

                    - **CoreCount** _(integer) --_

                        The number of CPU cores for the instance.

                    - **ThreadsPerCore** _(integer) --_

                        The number of threads per CPU core.

                - **CapacityReservationSpecification** _(dict) --_

                    Information about the Capacity Reservation targeting option.

                    - **CapacityReservationPreference** _(string) --_

                        Indicates the instance's Capacity Reservation preferences. Possible preferences include:

                        - open - The instance can run in any open Capacity Reservation that has matching attributes (instance type, platform, Availability Zone).
                        - none - The instance avoids running in a Capacity Reservation even if one is available. The instance runs in On-Demand capacity.
                    - **CapacityReservationTarget** _(dict) --_

                        Information about the target Capacity Reservation or Capacity Reservation group.

                        - **CapacityReservationId** _(string) --_

                            The ID of the targeted Capacity Reservation.

                        - **CapacityReservationResourceGroupArn** _(string) --_

                            The ARN of the targeted Capacity Reservation group.

                - **LicenseSpecifications** _(list) --_

                    The license configurations.

                    - _(dict) --_

                        Describes a license configuration.

                        - **LicenseConfigurationArn** _(string) --_

                            The Amazon Resource Name (ARN) of the license configuration.

                - **HibernationOptions** _(dict) --_

                    Indicates whether an instance is configured for hibernation. For more information, see [Hibernate Your Instance](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/Hibernate.html) in the _Amazon Elastic Compute Cloud User Guide_ .

                    - **Configured** _(boolean) --_

                        If this parameter is set to true , the instance is enabled for hibernation; otherwise, it is not enabled for hibernation.

                - **MetadataOptions** _(dict) --_

                    The metadata options for the instance. For more information, see [Instance Metadata and User Data](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-instance-metadata.html) in the _Amazon Elastic Compute Cloud User Guide_ .

                    - **State** _(string) --_

                        The state of the metadata option changes.

                        > pending - The metadata options are being updated and the instance is not ready to process metadata traffic with the new selection.
                        >
                        > applied - The metadata options have been successfully applied on the instance.

                    - **HttpTokens** _(string) --_

                        The state of token usage for your instance metadata requests. If the parameter is not specified in the request, the default state is optional .

                        If the state is optional , you can choose to retrieve instance metadata with or without a signed token header on your request. If you retrieve the IAM role credentials without a token, the version 1.0 role credentials are returned. If you retrieve the IAM role credentials using a valid signed token, the version 2.0 role credentials are returned.

                        If the state is required , you must send a signed token header with any instance metadata retrieval requests. In this state, retrieving the IAM role credentials always returns the version 2.0 credentials; the version 1.0 credentials are not available.

                    - **HttpPutResponseHopLimit** _(integer) --_

                        The desired HTTP PUT response hop limit for instance metadata requests. The larger the number, the further instance metadata requests can travel.

                        Default: 1

                        Possible values: Integers from 1 to 64

                    - **HttpEndpoint** _(string) --_

                        This parameter enables or disables the HTTP metadata endpoint on your instances. If the parameter is not specified, the default state is enabled .

                        Note

                        If you specify a value of disabled , you will not be able to access your instance metadata.

                - **EnclaveOptions** _(dict) --_

                    Indicates whether the instance is enabled for AWS Nitro Enclaves.

                    - **Enabled** _(boolean) --_

                        If this parameter is set to true , the instance is enabled for AWS Nitro Enclaves; otherwise, it is not enabled for AWS Nitro Enclaves.


_class_ EC2.Paginator.DescribeLaunchTemplates

paginator = client.get_paginator('describe_launch_templates')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_launch_templates()](#EC2.Client.describe_launch_templates "EC2.Client.describe_launch_templates").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeLaunchTemplates)

**Request Syntax**

response_iterator = paginator.paginate(
    DryRun=True|False,
    LaunchTemplateIds=[
        'string',
    ],
    LaunchTemplateNames=[
        'string',
    ],
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **LaunchTemplateIds** (_list_) --

    One or more launch template IDs.

    - _(string) --_
- **LaunchTemplateNames** (_list_) --

    One or more launch template names.

    - _(string) --_
- **Filters** (_list_) --

    One or more filters.

    - create-time - The time the launch template was created.
    - launch-template-name - The name of the launch template.
    - tag :<key> - The key/value combination of a tag assigned to the resource. Use the tag key in the filter name and the tag value as the filter value. For example, to find all resources that have a tag with the key Owner and the value TeamA , specify tag:Owner for the filter name and TeamA for the filter value.
    - tag-key - The key of a tag assigned to the resource. Use this filter to find all resources assigned a tag with a specific key, regardless of the tag value.

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'LaunchTemplates': [
        {
            'LaunchTemplateId': 'string',
            'LaunchTemplateName': 'string',
            'CreateTime': datetime(2015, 1, 1),
            'CreatedBy': 'string',
            'DefaultVersionNumber': 123,
            'LatestVersionNumber': 123,
            'Tags': [
                {
                    'Key': 'string',
                    'Value': 'string'
                },
            ]
        },
    ],

}

**Response Structure**

- _(dict) --_

    - **LaunchTemplates** _(list) --_

        Information about the launch templates.

        - _(dict) --_

            Describes a launch template.

            - **LaunchTemplateId** _(string) --_

                The ID of the launch template.

            - **LaunchTemplateName** _(string) --_

                The name of the launch template.

            - **CreateTime** _(datetime) --_

                The time launch template was created.

            - **CreatedBy** _(string) --_

                The principal that created the launch template.

            - **DefaultVersionNumber** _(integer) --_

                The version number of the default version of the launch template.

            - **LatestVersionNumber** _(integer) --_

                The version number of the latest version of the launch template.

            - **Tags** _(list) --_

                The tags for the launch template.

                - _(dict) --_

                    Describes a tag.

                    - **Key** _(string) --_

                        The key of the tag.

                        Constraints: Tag keys are case-sensitive and accept a maximum of 127 Unicode characters. May not begin with aws: .

                    - **Value** _(string) --_

                        The value of the tag.

                        Constraints: Tag values are case-sensitive and accept a maximum of 255 Unicode characters.


_class_ EC2.Paginator.DescribeLocalGatewayRouteTableVirtualInterfaceGroupAssociationsDescribeLocalGatewayRouteTableVirtualInterfaceGroupAssociations "Permalink to this definition")

paginator = client.get_paginator('describe_local_gateway_route_table_virtual_interface_group_associations')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_local_gateway_route_table_virtual_interface_group_associations()](#EC2.Client.describe_local_gateway_route_table_virtual_interface_group_associations "EC2.Client.describe_local_gateway_route_table_virtual_interface_group_associations").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeLocalGatewayRouteTableVirtualInterfaceGroupAssociations)

**Request Syntax**

response_iterator = paginator.paginate(
    LocalGatewayRouteTableVirtualInterfaceGroupAssociationIds=[
        'string',
    ],
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    DryRun=True|False,
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **LocalGatewayRouteTableVirtualInterfaceGroupAssociationIds** (_list_) --

    The IDs of the associations.

    - _(string) --_
- **Filters** (_list_) --

    One or more filters.

    - local-gateway-id - The ID of a local gateway.
    - local-gateway-route-table-id - The ID of the local gateway route table.
    - local-gateway-route-table-virtual-interface-group-association-id - The ID of the association.
    - local-gateway-route-table-virtual-interface-group-id - The ID of the virtual interface group.
    - state - The state of the association.

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'LocalGatewayRouteTableVirtualInterfaceGroupAssociations': [
        {
            'LocalGatewayRouteTableVirtualInterfaceGroupAssociationId': 'string',
            'LocalGatewayVirtualInterfaceGroupId': 'string',
            'LocalGatewayId': 'string',
            'LocalGatewayRouteTableId': 'string',
            'LocalGatewayRouteTableArn': 'string',
            'OwnerId': 'string',
            'State': 'string',
            'Tags': [
                {
                    'Key': 'string',
                    'Value': 'string'
                },
            ]
        },
    ],

}

**Response Structure**

- _(dict) --_

    - **LocalGatewayRouteTableVirtualInterfaceGroupAssociations** _(list) --_

        Information about the associations.

        - _(dict) --_

            Describes an association between a local gateway route table and a virtual interface group.

            - **LocalGatewayRouteTableVirtualInterfaceGroupAssociationId** _(string) --_

                The ID of the association.

            - **LocalGatewayVirtualInterfaceGroupId** _(string) --_

                The ID of the virtual interface group.

            - **LocalGatewayId** _(string) --_

                The ID of the local gateway.

            - **LocalGatewayRouteTableId** _(string) --_

                The ID of the local gateway route table.

            - **LocalGatewayRouteTableArn** _(string) --_

                The Amazon Resource Name (ARN) of the local gateway route table for the virtual interface group.

            - **OwnerId** _(string) --_

                The AWS account ID that owns the local gateway virtual interface group association.

            - **State** _(string) --_

                The state of the association.

            - **Tags** _(list) --_

                The tags assigned to the association.

                - _(dict) --_

                    Describes a tag.

                    - **Key** _(string) --_

                        The key of the tag.

                        Constraints: Tag keys are case-sensitive and accept a maximum of 127 Unicode characters. May not begin with aws: .

                    - **Value** _(string) --_

                        The value of the tag.

                        Constraints: Tag values are case-sensitive and accept a maximum of 255 Unicode characters.


_class_ EC2.Paginator.DescribeLocalGatewayRouteTableVpcAssociationsthis definition")

paginator = client.get_paginator('describe_local_gateway_route_table_vpc_associations')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_local_gateway_route_table_vpc_associations()](#EC2.Client.describe_local_gateway_route_table_vpc_associations "EC2.Client.describe_local_gateway_route_table_vpc_associations").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeLocalGatewayRouteTableVpcAssociations)

**Request Syntax**

response_iterator = paginator.paginate(
    LocalGatewayRouteTableVpcAssociationIds=[
        'string',
    ],
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    DryRun=True|False,
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **LocalGatewayRouteTableVpcAssociationIds** (_list_) --

    The IDs of the associations.

    - _(string) --_
- **Filters** (_list_) --

    One or more filters.

    - local-gateway-id - The ID of a local gateway.
    - local-gateway-route-table-id - The ID of the local gateway route table.
    - local-gateway-route-table-vpc-association-id - The ID of the association.
    - state - The state of the association.
    - vpc-id - The ID of the VPC.

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'LocalGatewayRouteTableVpcAssociations': [
        {
            'LocalGatewayRouteTableVpcAssociationId': 'string',
            'LocalGatewayRouteTableId': 'string',
            'LocalGatewayRouteTableArn': 'string',
            'LocalGatewayId': 'string',
            'VpcId': 'string',
            'OwnerId': 'string',
            'State': 'string',
            'Tags': [
                {
                    'Key': 'string',
                    'Value': 'string'
                },
            ]
        },
    ],

}

**Response Structure**

- _(dict) --_

    - **LocalGatewayRouteTableVpcAssociations** _(list) --_

        Information about the associations.

        - _(dict) --_

            Describes an association between a local gateway route table and a VPC.

            - **LocalGatewayRouteTableVpcAssociationId** _(string) --_

                The ID of the association.

            - **LocalGatewayRouteTableId** _(string) --_

                The ID of the local gateway route table.

            - **LocalGatewayRouteTableArn** _(string) --_

                The Amazon Resource Name (ARN) of the local gateway route table for the association.

            - **LocalGatewayId** _(string) --_

                The ID of the local gateway.

            - **VpcId** _(string) --_

                The ID of the VPC.

            - **OwnerId** _(string) --_

                The AWS account ID that owns the local gateway route table for the association.

            - **State** _(string) --_

                The state of the association.

            - **Tags** _(list) --_

                The tags assigned to the association.

                - _(dict) --_

                    Describes a tag.

                    - **Key** _(string) --_

                        The key of the tag.

                        Constraints: Tag keys are case-sensitive and accept a maximum of 127 Unicode characters. May not begin with aws: .

                    - **Value** _(string) --_

                        The value of the tag.

                        Constraints: Tag values are case-sensitive and accept a maximum of 255 Unicode characters.


_class_ EC2.Paginator.DescribeLocalGatewayRouteTables

paginator = client.get_paginator('describe_local_gateway_route_tables')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_local_gateway_route_tables()](#EC2.Client.describe_local_gateway_route_tables "EC2.Client.describe_local_gateway_route_tables").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeLocalGatewayRouteTables)

**Request Syntax**

response_iterator = paginator.paginate(
    LocalGatewayRouteTableIds=[
        'string',
    ],
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    DryRun=True|False,
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **LocalGatewayRouteTableIds** (_list_) --

    The IDs of the local gateway route tables.

    - _(string) --_
- **Filters** (_list_) --

    One or more filters.

    - local-gateway-id - The ID of a local gateway.
    - local-gateway-route-table-id - The ID of a local gateway route table.
    - outpost-arn - The Amazon Resource Name (ARN) of the Outpost.
    - state - The state of the local gateway route table.

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'LocalGatewayRouteTables': [
        {
            'LocalGatewayRouteTableId': 'string',
            'LocalGatewayRouteTableArn': 'string',
            'LocalGatewayId': 'string',
            'OutpostArn': 'string',
            'OwnerId': 'string',
            'State': 'string',
            'Tags': [
                {
                    'Key': 'string',
                    'Value': 'string'
                },
            ]
        },
    ],

}

**Response Structure**

- _(dict) --_

    - **LocalGatewayRouteTables** _(list) --_

        Information about the local gateway route tables.

        - _(dict) --_

            Describes a local gateway route table.

            - **LocalGatewayRouteTableId** _(string) --_

                The ID of the local gateway route table.

            - **LocalGatewayRouteTableArn** _(string) --_

                The Amazon Resource Name (ARN) of the local gateway route table.

            - **LocalGatewayId** _(string) --_

                The ID of the local gateway.

            - **OutpostArn** _(string) --_

                The Amazon Resource Name (ARN) of the Outpost.

            - **OwnerId** _(string) --_

                The AWS account ID that owns the local gateway route table.

            - **State** _(string) --_

                The state of the local gateway route table.

            - **Tags** _(list) --_

                The tags assigned to the local gateway route table.

                - _(dict) --_

                    Describes a tag.

                    - **Key** _(string) --_

                        The key of the tag.

                        Constraints: Tag keys are case-sensitive and accept a maximum of 127 Unicode characters. May not begin with aws: .

                    - **Value** _(string) --_

                        The value of the tag.

                        Constraints: Tag values are case-sensitive and accept a maximum of 255 Unicode characters.


_class_ EC2.Paginator.DescribeLocalGatewayVirtualInterfaceGroupsdefinition")

paginator = client.get_paginator('describe_local_gateway_virtual_interface_groups')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_local_gateway_virtual_interface_groups()](#EC2.Client.describe_local_gateway_virtual_interface_groups "EC2.Client.describe_local_gateway_virtual_interface_groups").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeLocalGatewayVirtualInterfaceGroups)

**Request Syntax**

response_iterator = paginator.paginate(
    LocalGatewayVirtualInterfaceGroupIds=[
        'string',
    ],
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    DryRun=True|False,
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **LocalGatewayVirtualInterfaceGroupIds** (_list_) --

    The IDs of the virtual interface groups.

    - _(string) --_
- **Filters** (_list_) --

    One or more filters.

    - local-gateway-id - The ID of a local gateway.
    - local-gateway-virtual-interface-id - The ID of the virtual interface.
    - local-gateway-virtual-interface-group-id - The ID of the virtual interface group.

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'LocalGatewayVirtualInterfaceGroups': [
        {
            'LocalGatewayVirtualInterfaceGroupId': 'string',
            'LocalGatewayVirtualInterfaceIds': [
                'string',
            ],
            'LocalGatewayId': 'string',
            'OwnerId': 'string',
            'Tags': [
                {
                    'Key': 'string',
                    'Value': 'string'
                },
            ]
        },
    ],

}

**Response Structure**

- _(dict) --_

    - **LocalGatewayVirtualInterfaceGroups** _(list) --_

        The virtual interface groups.

        - _(dict) --_

            Describes a local gateway virtual interface group.

            - **LocalGatewayVirtualInterfaceGroupId** _(string) --_

                The ID of the virtual interface group.

            - **LocalGatewayVirtualInterfaceIds** _(list) --_

                The IDs of the virtual interfaces.

                - _(string) --_
            - **LocalGatewayId** _(string) --_

                The ID of the local gateway.

            - **OwnerId** _(string) --_

                The AWS account ID that owns the local gateway virtual interface group.

            - **Tags** _(list) --_

                The tags assigned to the virtual interface group.

                - _(dict) --_

                    Describes a tag.

                    - **Key** _(string) --_

                        The key of the tag.

                        Constraints: Tag keys are case-sensitive and accept a maximum of 127 Unicode characters. May not begin with aws: .

                    - **Value** _(string) --_

                        The value of the tag.

                        Constraints: Tag values are case-sensitive and accept a maximum of 255 Unicode characters.


_class_ EC2.Paginator.DescribeLocalGatewayVirtualInterfaces

paginator = client.get_paginator('describe_local_gateway_virtual_interfaces')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_local_gateway_virtual_interfaces()](#EC2.Client.describe_local_gateway_virtual_interfaces "EC2.Client.describe_local_gateway_virtual_interfaces").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeLocalGatewayVirtualInterfaces)

**Request Syntax**

response_iterator = paginator.paginate(
    LocalGatewayVirtualInterfaceIds=[
        'string',
    ],
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    DryRun=True|False,
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **LocalGatewayVirtualInterfaceIds** (_list_) --

    The IDs of the virtual interfaces.

    - _(string) --_
- **Filters** (_list_) --

    One or more filters.

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'LocalGatewayVirtualInterfaces': [
        {
            'LocalGatewayVirtualInterfaceId': 'string',
            'LocalGatewayId': 'string',
            'Vlan': 123,
            'LocalAddress': 'string',
            'PeerAddress': 'string',
            'LocalBgpAsn': 123,
            'PeerBgpAsn': 123,
            'OwnerId': 'string',
            'Tags': [
                {
                    'Key': 'string',
                    'Value': 'string'
                },
            ]
        },
    ],

}

**Response Structure**

- _(dict) --_

    - **LocalGatewayVirtualInterfaces** _(list) --_

        Information about the virtual interfaces.

        - _(dict) --_

            Describes a local gateway virtual interface.

            - **LocalGatewayVirtualInterfaceId** _(string) --_

                The ID of the virtual interface.

            - **LocalGatewayId** _(string) --_

                The ID of the local gateway.

            - **Vlan** _(integer) --_

                The ID of the VLAN.

            - **LocalAddress** _(string) --_

                The local address.

            - **PeerAddress** _(string) --_

                The peer address.

            - **LocalBgpAsn** _(integer) --_

                The Border Gateway Protocol (BGP) Autonomous System Number (ASN) of the local gateway.

            - **PeerBgpAsn** _(integer) --_

                The peer BGP ASN.

            - **OwnerId** _(string) --_

                The AWS account ID that owns the local gateway virtual interface.

            - **Tags** _(list) --_

                The tags assigned to the virtual interface.

                - _(dict) --_

                    Describes a tag.

                    - **Key** _(string) --_

                        The key of the tag.

                        Constraints: Tag keys are case-sensitive and accept a maximum of 127 Unicode characters. May not begin with aws: .

                    - **Value** _(string) --_

                        The value of the tag.

                        Constraints: Tag values are case-sensitive and accept a maximum of 255 Unicode characters.


_class_ EC2.Paginator.DescribeLocalGateways

paginator = client.get_paginator('describe_local_gateways')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_local_gateways()](#EC2.Client.describe_local_gateways "EC2.Client.describe_local_gateways").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeLocalGateways)

**Request Syntax**

response_iterator = paginator.paginate(
    LocalGatewayIds=[
        'string',
    ],
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    DryRun=True|False,
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **LocalGatewayIds** (_list_) --

    One or more filters.

    - local-gateway-id - The ID of a local gateway.
    - local-gateway-route-table-id - The ID of the local gateway route table.
    - local-gateway-route-table-virtual-interface-group-association-id - The ID of the association.
    - local-gateway-route-table-virtual-interface-group-id - The ID of the virtual interface group.
    - outpost-arn - The Amazon Resource Name (ARN) of the Outpost.
    - state - The state of the association.

    - _(string) --_
- **Filters** (_list_) --

    One or more filters.

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'LocalGateways': [
        {
            'LocalGatewayId': 'string',
            'OutpostArn': 'string',
            'OwnerId': 'string',
            'State': 'string',
            'Tags': [
                {
                    'Key': 'string',
                    'Value': 'string'
                },
            ]
        },
    ],

}

**Response Structure**

- _(dict) --_

    - **LocalGateways** _(list) --_

        Information about the local gateways.

        - _(dict) --_

            Describes a local gateway.

            - **LocalGatewayId** _(string) --_

                The ID of the local gateway.

            - **OutpostArn** _(string) --_

                The Amazon Resource Name (ARN) of the Outpost.

            - **OwnerId** _(string) --_

                The AWS account ID that owns the local gateway.

            - **State** _(string) --_

                The state of the local gateway.

            - **Tags** _(list) --_

                The tags assigned to the local gateway.

                - _(dict) --_

                    Describes a tag.

                    - **Key** _(string) --_

                        The key of the tag.

                        Constraints: Tag keys are case-sensitive and accept a maximum of 127 Unicode characters. May not begin with aws: .

                    - **Value** _(string) --_

                        The value of the tag.

                        Constraints: Tag values are case-sensitive and accept a maximum of 255 Unicode characters.


_class_ EC2.Paginator.DescribeManagedPrefixLists

paginator = client.get_paginator('describe_managed_prefix_lists')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_managed_prefix_lists()](#EC2.Client.describe_managed_prefix_lists "EC2.Client.describe_managed_prefix_lists").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeManagedPrefixLists)

**Request Syntax**

response_iterator = paginator.paginate(
    DryRun=True|False,
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    PrefixListIds=[
        'string',
    ],
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **Filters** (_list_) --

    One or more filters.

    - owner-id - The ID of the prefix list owner.
    - prefix-list-id - The ID of the prefix list.
    - prefix-list-name - The name of the prefix list.

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **PrefixListIds** (_list_) --

    One or more prefix list IDs.

    - _(string) --_
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'PrefixLists': [
        {
            'PrefixListId': 'string',
            'AddressFamily': 'string',
            'State': 'create-in-progress'|'create-complete'|'create-failed'|'modify-in-progress'|'modify-complete'|'modify-failed'|'restore-in-progress'|'restore-complete'|'restore-failed'|'delete-in-progress'|'delete-complete'|'delete-failed',
            'StateMessage': 'string',
            'PrefixListArn': 'string',
            'PrefixListName': 'string',
            'MaxEntries': 123,
            'Version': 123,
            'Tags': [
                {
                    'Key': 'string',
                    'Value': 'string'
                },
            ],
            'OwnerId': 'string'
        },
    ]
}

**Response Structure**

- _(dict) --_

    - **PrefixLists** _(list) --_

        Information about the prefix lists.

        - _(dict) --_

            Describes a managed prefix list.

            - **PrefixListId** _(string) --_

                The ID of the prefix list.

            - **AddressFamily** _(string) --_

                The IP address version.

            - **State** _(string) --_

                The state of the prefix list.

            - **StateMessage** _(string) --_

                The state message.

            - **PrefixListArn** _(string) --_

                The Amazon Resource Name (ARN) for the prefix list.

            - **PrefixListName** _(string) --_

                The name of the prefix list.

            - **MaxEntries** _(integer) --_

                The maximum number of entries for the prefix list.

            - **Version** _(integer) --_

                The version of the prefix list.

            - **Tags** _(list) --_

                The tags for the prefix list.

                - _(dict) --_

                    Describes a tag.

                    - **Key** _(string) --_

                        The key of the tag.

                        Constraints: Tag keys are case-sensitive and accept a maximum of 127 Unicode characters. May not begin with aws: .

                    - **Value** _(string) --_

                        The value of the tag.

                        Constraints: Tag values are case-sensitive and accept a maximum of 255 Unicode characters.

            - **OwnerId** _(string) --_

                The ID of the owner of the prefix list.


_class_ EC2.Paginator.DescribeMovingAddresses

paginator = client.get_paginator('describe_moving_addresses')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_moving_addresses()](#EC2.Client.describe_moving_addresses "EC2.Client.describe_moving_addresses").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeMovingAddresses)

**Request Syntax**

response_iterator = paginator.paginate(
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    DryRun=True|False,
    PublicIps=[
        'string',
    ],
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **Filters** (_list_) --

    One or more filters.

    - moving-status - The status of the Elastic IP address (MovingToVpc | RestoringToClassic ).

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **PublicIps** (_list_) --

    One or more Elastic IP addresses.

    - _(string) --_
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'MovingAddressStatuses': [
        {
            'MoveStatus': 'movingToVpc'|'restoringToClassic',
            'PublicIp': 'string'
        },
    ],

}

**Response Structure**

- _(dict) --_

    - **MovingAddressStatuses** _(list) --_

        The status for each Elastic IP address.

        - _(dict) --_

            Describes the status of a moving Elastic IP address.

            - **MoveStatus** _(string) --_

                The status of the Elastic IP address that's being moved to the EC2-VPC platform, or restored to the EC2-Classic platform.

            - **PublicIp** _(string) --_

                The Elastic IP address.


_class_ EC2.Paginator.DescribeNatGateways

paginator = client.get_paginator('describe_nat_gateways')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_nat_gateways()](#EC2.Client.describe_nat_gateways "EC2.Client.describe_nat_gateways").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeNatGateways)

**Request Syntax**

response_iterator = paginator.paginate(
    DryRun=True|False,
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    NatGatewayIds=[
        'string',
    ],
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **Filters** (_list_) --

    One or more filters.

    - nat-gateway-id - The ID of the NAT gateway.
    - state - The state of the NAT gateway (pending | failed | available | deleting | deleted ).
    - subnet-id - The ID of the subnet in which the NAT gateway resides.
    - tag :<key> - The key/value combination of a tag assigned to the resource. Use the tag key in the filter name and the tag value as the filter value. For example, to find all resources that have a tag with the key Owner and the value TeamA , specify tag:Owner for the filter name and TeamA for the filter value.
    - tag-key - The key of a tag assigned to the resource. Use this filter to find all resources assigned a tag with a specific key, regardless of the tag value.
    - vpc-id - The ID of the VPC in which the NAT gateway resides.

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **NatGatewayIds** (_list_) --

    One or more NAT gateway IDs.

    - _(string) --_
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'NatGateways': [
        {
            'CreateTime': datetime(2015, 1, 1),
            'DeleteTime': datetime(2015, 1, 1),
            'FailureCode': 'string',
            'FailureMessage': 'string',
            'NatGatewayAddresses': [
                {
                    'AllocationId': 'string',
                    'NetworkInterfaceId': 'string',
                    'PrivateIp': 'string',
                    'PublicIp': 'string'
                },
            ],
            'NatGatewayId': 'string',
            'ProvisionedBandwidth': {
                'ProvisionTime': datetime(2015, 1, 1),
                'Provisioned': 'string',
                'RequestTime': datetime(2015, 1, 1),
                'Requested': 'string',
                'Status': 'string'
            },
            'State': 'pending'|'failed'|'available'|'deleting'|'deleted',
            'SubnetId': 'string',
            'VpcId': 'string',
            'Tags': [
                {
                    'Key': 'string',
                    'Value': 'string'
                },
            ]
        },
    ],

}

**Response Structure**

- _(dict) --_

    - **NatGateways** _(list) --_

        Information about the NAT gateways.

        - _(dict) --_

            Describes a NAT gateway.

            - **CreateTime** _(datetime) --_

                The date and time the NAT gateway was created.

            - **DeleteTime** _(datetime) --_

                The date and time the NAT gateway was deleted, if applicable.

            - **FailureCode** _(string) --_

                If the NAT gateway could not be created, specifies the error code for the failure. (InsufficientFreeAddressesInSubnet | Gateway.NotAttached | InvalidAllocationID.NotFound | Resource.AlreadyAssociated | InternalError | InvalidSubnetID.NotFound )

            - **FailureMessage** _(string) --_

                If the NAT gateway could not be created, specifies the error message for the failure, that corresponds to the error code.

                - For InsufficientFreeAddressesInSubnet: "Subnet has insufficient free addresses to create this NAT gateway"
                - For Gateway.NotAttached: "Network vpc-xxxxxxxx has no Internet gateway attached"
                - For InvalidAllocationID.NotFound: "Elastic IP address eipalloc-xxxxxxxx could not be associated with this NAT gateway"
                - For Resource.AlreadyAssociated: "Elastic IP address eipalloc-xxxxxxxx is already associated"
                - For InternalError: "Network interface eni-xxxxxxxx, created and used internally by this NAT gateway is in an invalid state. Please try again."
                - For InvalidSubnetID.NotFound: "The specified subnet subnet-xxxxxxxx does not exist or could not be found."
            - **NatGatewayAddresses** _(list) --_

                Information about the IP addresses and network interface associated with the NAT gateway.

                - _(dict) --_

                    Describes the IP addresses and network interface associated with a NAT gateway.

                    - **AllocationId** _(string) --_

                        The allocation ID of the Elastic IP address that's associated with the NAT gateway.

                    - **NetworkInterfaceId** _(string) --_

                        The ID of the network interface associated with the NAT gateway.

                    - **PrivateIp** _(string) --_

                        The private IP address associated with the Elastic IP address.

                    - **PublicIp** _(string) --_

                        The Elastic IP address associated with the NAT gateway.

            - **NatGatewayId** _(string) --_

                The ID of the NAT gateway.

            - **ProvisionedBandwidth** _(dict) --_

                Reserved. If you need to sustain traffic greater than the [documented limits](https://docs.aws.amazon.com/vpc/latest/userguide/vpc-nat-gateway.html) , contact us through the [Support Center](https://console.aws.amazon.com/support/home?) .

                - **ProvisionTime** _(datetime) --_

                    Reserved. If you need to sustain traffic greater than the [documented limits](https://docs.aws.amazon.com/vpc/latest/userguide/vpc-nat-gateway.html) , contact us through the [Support Center](https://console.aws.amazon.com/support/home?) .

                - **Provisioned** _(string) --_

                    Reserved. If you need to sustain traffic greater than the [documented limits](https://docs.aws.amazon.com/vpc/latest/userguide/vpc-nat-gateway.html) , contact us through the [Support Center](https://console.aws.amazon.com/support/home?) .

                - **RequestTime** _(datetime) --_

                    Reserved. If you need to sustain traffic greater than the [documented limits](https://docs.aws.amazon.com/vpc/latest/userguide/vpc-nat-gateway.html) , contact us through the [Support Center](https://console.aws.amazon.com/support/home?) .

                - **Requested** _(string) --_

                    Reserved. If you need to sustain traffic greater than the [documented limits](https://docs.aws.amazon.com/vpc/latest/userguide/vpc-nat-gateway.html) , contact us through the [Support Center](https://console.aws.amazon.com/support/home?) .

                - **Status** _(string) --_

                    Reserved. If you need to sustain traffic greater than the [documented limits](https://docs.aws.amazon.com/vpc/latest/userguide/vpc-nat-gateway.html) , contact us through the [Support Center](https://console.aws.amazon.com/support/home?) .

            - **State** _(string) --_

                The state of the NAT gateway.

                - pending : The NAT gateway is being created and is not ready to process traffic.
                - failed : The NAT gateway could not be created. Check the failureCode and failureMessage fields for the reason.
                - available : The NAT gateway is able to process traffic. This status remains until you delete the NAT gateway, and does not indicate the health of the NAT gateway.
                - deleting : The NAT gateway is in the process of being terminated and may still be processing traffic.
                - deleted : The NAT gateway has been terminated and is no longer processing traffic.
            - **SubnetId** _(string) --_

                The ID of the subnet in which the NAT gateway is located.

            - **VpcId** _(string) --_

                The ID of the VPC in which the NAT gateway is located.

            - **Tags** _(list) --_

                The tags for the NAT gateway.

                - _(dict) --_

                    Describes a tag.

                    - **Key** _(string) --_

                        The key of the tag.

                        Constraints: Tag keys are case-sensitive and accept a maximum of 127 Unicode characters. May not begin with aws: .

                    - **Value** _(string) --_

                        The value of the tag.

                        Constraints: Tag values are case-sensitive and accept a maximum of 255 Unicode characters.


_class_ EC2.Paginator.DescribeNetworkAcls

paginator = client.get_paginator('describe_network_acls')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_network_acls()](#EC2.Client.describe_network_acls "EC2.Client.describe_network_acls").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeNetworkAcls)

**Request Syntax**

response_iterator = paginator.paginate(
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    DryRun=True|False,
    NetworkAclIds=[
        'string',
    ],
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **Filters** (_list_) --

    One or more filters.

    - association.association-id - The ID of an association ID for the ACL.
    - association.network-acl-id - The ID of the network ACL involved in the association.
    - association.subnet-id - The ID of the subnet involved in the association.
    - default - Indicates whether the ACL is the default network ACL for the VPC.
    - entry.cidr - The IPv4 CIDR range specified in the entry.
    - entry.icmp.code - The ICMP code specified in the entry, if any.
    - entry.icmp.type - The ICMP type specified in the entry, if any.
    - entry.ipv6-cidr - The IPv6 CIDR range specified in the entry.
    - entry.port-range.from - The start of the port range specified in the entry.
    - entry.port-range.to - The end of the port range specified in the entry.
    - entry.protocol - The protocol specified in the entry (tcp | udp | icmp or a protocol number).
    - entry.rule-action - Allows or denies the matching traffic (allow | deny ).
    - entry.rule-number - The number of an entry (in other words, rule) in the set of ACL entries.
    - network-acl-id - The ID of the network ACL.
    - owner-id - The ID of the AWS account that owns the network ACL.
    - tag :<key> - The key/value combination of a tag assigned to the resource. Use the tag key in the filter name and the tag value as the filter value. For example, to find all resources that have a tag with the key Owner and the value TeamA , specify tag:Owner for the filter name and TeamA for the filter value.
    - tag-key - The key of a tag assigned to the resource. Use this filter to find all resources assigned a tag with a specific key, regardless of the tag value.
    - vpc-id - The ID of the VPC for the network ACL.

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **NetworkAclIds** (_list_) --

    One or more network ACL IDs.

    Default: Describes all your network ACLs.

    - _(string) --_
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'NetworkAcls': [
        {
            'Associations': [
                {
                    'NetworkAclAssociationId': 'string',
                    'NetworkAclId': 'string',
                    'SubnetId': 'string'
                },
            ],
            'Entries': [
                {
                    'CidrBlock': 'string',
                    'Egress': True|False,
                    'IcmpTypeCode': {
                        'Code': 123,
                        'Type': 123
                    },
                    'Ipv6CidrBlock': 'string',
                    'PortRange': {
                        'From': 123,
                        'To': 123
                    },
                    'Protocol': 'string',
                    'RuleAction': 'allow'|'deny',
                    'RuleNumber': 123
                },
            ],
            'IsDefault': True|False,
            'NetworkAclId': 'string',
            'Tags': [
                {
                    'Key': 'string',
                    'Value': 'string'
                },
            ],
            'VpcId': 'string',
            'OwnerId': 'string'
        },
    ],

}

**Response Structure**

- _(dict) --_

    - **NetworkAcls** _(list) --_

        Information about one or more network ACLs.

        - _(dict) --_

            Describes a network ACL.

            - **Associations** _(list) --_

                Any associations between the network ACL and one or more subnets

                - _(dict) --_

                    Describes an association between a network ACL and a subnet.

                    - **NetworkAclAssociationId** _(string) --_

                        The ID of the association between a network ACL and a subnet.

                    - **NetworkAclId** _(string) --_

                        The ID of the network ACL.

                    - **SubnetId** _(string) --_

                        The ID of the subnet.

            - **Entries** _(list) --_

                One or more entries (rules) in the network ACL.

                - _(dict) --_

                    Describes an entry in a network ACL.

                    - **CidrBlock** _(string) --_

                        The IPv4 network range to allow or deny, in CIDR notation.

                    - **Egress** _(boolean) --_

                        Indicates whether the rule is an egress rule (applied to traffic leaving the subnet).

                    - **IcmpTypeCode** _(dict) --_

                        ICMP protocol: The ICMP type and code.

                        - **Code** _(integer) --_

                            The ICMP code. A value of -1 means all codes for the specified ICMP type.

                        - **Type** _(integer) --_

                            The ICMP type. A value of -1 means all types.

                    - **Ipv6CidrBlock** _(string) --_

                        The IPv6 network range to allow or deny, in CIDR notation.

                    - **PortRange** _(dict) --_

                        TCP or UDP protocols: The range of ports the rule applies to.

                        - **From** _(integer) --_

                            The first port in the range.

                        - **To** _(integer) --_

                            The last port in the range.

                    - **Protocol** _(string) --_

                        The protocol number. A value of "-1" means all protocols.

                    - **RuleAction** _(string) --_

                        Indicates whether to allow or deny the traffic that matches the rule.

                    - **RuleNumber** _(integer) --_

                        The rule number for the entry. ACL entries are processed in ascending order by rule number.

            - **IsDefault** _(boolean) --_

                Indicates whether this is the default network ACL for the VPC.

            - **NetworkAclId** _(string) --_

                The ID of the network ACL.

            - **Tags** _(list) --_

                Any tags assigned to the network ACL.

                - _(dict) --_

                    Describes a tag.

                    - **Key** _(string) --_

                        The key of the tag.

                        Constraints: Tag keys are case-sensitive and accept a maximum of 127 Unicode characters. May not begin with aws: .

                    - **Value** _(string) --_

                        The value of the tag.

                        Constraints: Tag values are case-sensitive and accept a maximum of 255 Unicode characters.

            - **VpcId** _(string) --_

                The ID of the VPC for the network ACL.

            - **OwnerId** _(string) --_

                The ID of the AWS account that owns the network ACL.


_class_ EC2.Paginator.DescribeNetworkInsightsAnalyses

paginator = client.get_paginator('describe_network_insights_analyses')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_network_insights_analyses()](#EC2.Client.describe_network_insights_analyses "EC2.Client.describe_network_insights_analyses").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeNetworkInsightsAnalyses)

**Request Syntax**

response_iterator = paginator.paginate(
    NetworkInsightsAnalysisIds=[
        'string',
    ],
    NetworkInsightsPathId='string',
    AnalysisStartTime=datetime(2015, 1, 1),
    AnalysisEndTime=datetime(2015, 1, 1),
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    DryRun=True|False,
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **NetworkInsightsAnalysisIds** (_list_) --

    The ID of the network insights analyses. You must specify either analysis IDs or a path ID.

    - _(string) --_
- **NetworkInsightsPathId** (_string_) -- The ID of the path. You must specify either a path ID or analysis IDs.
- **AnalysisStartTime** (_datetime_) -- The time when the network insights analyses started.
- **AnalysisEndTime** (_datetime_) -- The time when the network insights analyses ended.
- **Filters** (_list_) --

    The filters. The following are possible values:

    - PathFound - A Boolean value that indicates whether a feasible path is found.
    - Status - The status of the analysis (running | succeeded | failed).

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'NetworkInsightsAnalyses': [
        {
            'NetworkInsightsAnalysisId': 'string',
            'NetworkInsightsAnalysisArn': 'string',
            'NetworkInsightsPathId': 'string',
            'FilterInArns': [
                'string',
            ],
            'StartDate': datetime(2015, 1, 1),
            'Status': 'running'|'succeeded'|'failed',
            'StatusMessage': 'string',
            'NetworkPathFound': True|False,
            'ForwardPathComponents': [
                {
                    'SequenceNumber': 123,
                    'AclRule': {
                        'Cidr': 'string',
                        'Egress': True|False,
                        'PortRange': {
                            'From': 123,
                            'To': 123
                        },
                        'Protocol': 'string',
                        'RuleAction': 'string',
                        'RuleNumber': 123
                    },
                    'Component': {
                        'Id': 'string',
                        'Arn': 'string'
                    },
                    'DestinationVpc': {
                        'Id': 'string',
                        'Arn': 'string'
                    },
                    'OutboundHeader': {
                        'DestinationAddresses': [
                            'string',
                        ],
                        'DestinationPortRanges': [
                            {
                                'From': 123,
                                'To': 123
                            },
                        ],
                        'Protocol': 'string',
                        'SourceAddresses': [
                            'string',
                        ],
                        'SourcePortRanges': [
                            {
                                'From': 123,
                                'To': 123
                            },
                        ]
                    },
                    'InboundHeader': {
                        'DestinationAddresses': [
                            'string',
                        ],
                        'DestinationPortRanges': [
                            {
                                'From': 123,
                                'To': 123
                            },
                        ],
                        'Protocol': 'string',
                        'SourceAddresses': [
                            'string',
                        ],
                        'SourcePortRanges': [
                            {
                                'From': 123,
                                'To': 123
                            },
                        ]
                    },
                    'RouteTableRoute': {
                        'DestinationCidr': 'string',
                        'DestinationPrefixListId': 'string',
                        'EgressOnlyInternetGatewayId': 'string',
                        'GatewayId': 'string',
                        'InstanceId': 'string',
                        'NatGatewayId': 'string',
                        'NetworkInterfaceId': 'string',
                        'Origin': 'string',
                        'TransitGatewayId': 'string',
                        'VpcPeeringConnectionId': 'string'
                    },
                    'SecurityGroupRule': {
                        'Cidr': 'string',
                        'Direction': 'string',
                        'SecurityGroupId': 'string',
                        'PortRange': {
                            'From': 123,
                            'To': 123
                        },
                        'PrefixListId': 'string',
                        'Protocol': 'string'
                    },
                    'SourceVpc': {
                        'Id': 'string',
                        'Arn': 'string'
                    },
                    'Subnet': {
                        'Id': 'string',
                        'Arn': 'string'
                    },
                    'Vpc': {
                        'Id': 'string',
                        'Arn': 'string'
                    }
                },
            ],
            'ReturnPathComponents': [
                {
                    'SequenceNumber': 123,
                    'AclRule': {
                        'Cidr': 'string',
                        'Egress': True|False,
                        'PortRange': {
                            'From': 123,
                            'To': 123
                        },
                        'Protocol': 'string',
                        'RuleAction': 'string',
                        'RuleNumber': 123
                    },
                    'Component': {
                        'Id': 'string',
                        'Arn': 'string'
                    },
                    'DestinationVpc': {
                        'Id': 'string',
                        'Arn': 'string'
                    },
                    'OutboundHeader': {
                        'DestinationAddresses': [
                            'string',
                        ],
                        'DestinationPortRanges': [
                            {
                                'From': 123,
                                'To': 123
                            },
                        ],
                        'Protocol': 'string',
                        'SourceAddresses': [
                            'string',
                        ],
                        'SourcePortRanges': [
                            {
                                'From': 123,
                                'To': 123
                            },
                        ]
                    },
                    'InboundHeader': {
                        'DestinationAddresses': [
                            'string',
                        ],
                        'DestinationPortRanges': [
                            {
                                'From': 123,
                                'To': 123
                            },
                        ],
                        'Protocol': 'string',
                        'SourceAddresses': [
                            'string',
                        ],
                        'SourcePortRanges': [
                            {
                                'From': 123,
                                'To': 123
                            },
                        ]
                    },
                    'RouteTableRoute': {
                        'DestinationCidr': 'string',
                        'DestinationPrefixListId': 'string',
                        'EgressOnlyInternetGatewayId': 'string',
                        'GatewayId': 'string',
                        'InstanceId': 'string',
                        'NatGatewayId': 'string',
                        'NetworkInterfaceId': 'string',
                        'Origin': 'string',
                        'TransitGatewayId': 'string',
                        'VpcPeeringConnectionId': 'string'
                    },
                    'SecurityGroupRule': {
                        'Cidr': 'string',
                        'Direction': 'string',
                        'SecurityGroupId': 'string',
                        'PortRange': {
                            'From': 123,
                            'To': 123
                        },
                        'PrefixListId': 'string',
                        'Protocol': 'string'
                    },
                    'SourceVpc': {
                        'Id': 'string',
                        'Arn': 'string'
                    },
                    'Subnet': {
                        'Id': 'string',
                        'Arn': 'string'
                    },
                    'Vpc': {
                        'Id': 'string',
                        'Arn': 'string'
                    }
                },
            ],
            'Explanations': [
                {
                    'Acl': {
                        'Id': 'string',
                        'Arn': 'string'
                    },
                    'AclRule': {
                        'Cidr': 'string',
                        'Egress': True|False,
                        'PortRange': {
                            'From': 123,
                            'To': 123
                        },
                        'Protocol': 'string',
                        'RuleAction': 'string',
                        'RuleNumber': 123
                    },
                    'Address': 'string',
                    'Addresses': [
                        'string',
                    ],
                    'AttachedTo': {
                        'Id': 'string',
                        'Arn': 'string'
                    },
                    'AvailabilityZones': [
                        'string',
                    ],
                    'Cidrs': [
                        'string',
                    ],
                    'Component': {
                        'Id': 'string',
                        'Arn': 'string'
                    },
                    'CustomerGateway': {
                        'Id': 'string',
                        'Arn': 'string'
                    },
                    'Destination': {
                        'Id': 'string',
                        'Arn': 'string'
                    },
                    'DestinationVpc': {
                        'Id': 'string',
                        'Arn': 'string'
                    },
                    'Direction': 'string',
                    'ExplanationCode': 'string',
                    'IngressRouteTable': {
                        'Id': 'string',
                        'Arn': 'string'
                    },
                    'InternetGateway': {
                        'Id': 'string',
                        'Arn': 'string'
                    },
                    'LoadBalancerArn': 'string',
                    'ClassicLoadBalancerListener': {
                        'LoadBalancerPort': 123,
                        'InstancePort': 123
                    },
                    'LoadBalancerListenerPort': 123,
                    'LoadBalancerTarget': {
                        'Address': 'string',
                        'AvailabilityZone': 'string',
                        'Instance': {
                            'Id': 'string',
                            'Arn': 'string'
                        },
                        'Port': 123
                    },
                    'LoadBalancerTargetGroup': {
                        'Id': 'string',
                        'Arn': 'string'
                    },
                    'LoadBalancerTargetGroups': [
                        {
                            'Id': 'string',
                            'Arn': 'string'
                        },
                    ],
                    'LoadBalancerTargetPort': 123,
                    'ElasticLoadBalancerListener': {
                        'Id': 'string',
                        'Arn': 'string'
                    },
                    'MissingComponent': 'string',
                    'NatGateway': {
                        'Id': 'string',
                        'Arn': 'string'
                    },
                    'NetworkInterface': {
                        'Id': 'string',
                        'Arn': 'string'
                    },
                    'PacketField': 'string',
                    'VpcPeeringConnection': {
                        'Id': 'string',
                        'Arn': 'string'
                    },
                    'Port': 123,
                    'PortRanges': [
                        {
                            'From': 123,
                            'To': 123
                        },
                    ],
                    'PrefixList': {
                        'Id': 'string',
                        'Arn': 'string'
                    },
                    'Protocols': [
                        'string',
                    ],
                    'RouteTableRoute': {
                        'DestinationCidr': 'string',
                        'DestinationPrefixListId': 'string',
                        'EgressOnlyInternetGatewayId': 'string',
                        'GatewayId': 'string',
                        'InstanceId': 'string',
                        'NatGatewayId': 'string',
                        'NetworkInterfaceId': 'string',
                        'Origin': 'string',
                        'TransitGatewayId': 'string',
                        'VpcPeeringConnectionId': 'string'
                    },
                    'RouteTable': {
                        'Id': 'string',
                        'Arn': 'string'
                    },
                    'SecurityGroup': {
                        'Id': 'string',
                        'Arn': 'string'
                    },
                    'SecurityGroupRule': {
                        'Cidr': 'string',
                        'Direction': 'string',
                        'SecurityGroupId': 'string',
                        'PortRange': {
                            'From': 123,
                            'To': 123
                        },
                        'PrefixListId': 'string',
                        'Protocol': 'string'
                    },
                    'SecurityGroups': [
                        {
                            'Id': 'string',
                            'Arn': 'string'
                        },
                    ],
                    'SourceVpc': {
                        'Id': 'string',
                        'Arn': 'string'
                    },
                    'State': 'string',
                    'Subnet': {
                        'Id': 'string',
                        'Arn': 'string'
                    },
                    'SubnetRouteTable': {
                        'Id': 'string',
                        'Arn': 'string'
                    },
                    'Vpc': {
                        'Id': 'string',
                        'Arn': 'string'
                    },
                    'VpcEndpoint': {
                        'Id': 'string',
                        'Arn': 'string'
                    },
                    'VpnConnection': {
                        'Id': 'string',
                        'Arn': 'string'
                    },
                    'VpnGateway': {
                        'Id': 'string',
                        'Arn': 'string'
                    }
                },
            ],
            'AlternatePathHints': [
                {
                    'ComponentId': 'string',
                    'ComponentArn': 'string'
                },
            ],
            'Tags': [
                {
                    'Key': 'string',
                    'Value': 'string'
                },
            ]
        },
    ],

}

**Response Structure**

- _(dict) --_

    - **NetworkInsightsAnalyses** _(list) --_

        Information about the network insights analyses.

        - _(dict) --_

            Describes a network insights analysis.

            - **NetworkInsightsAnalysisId** _(string) --_

                The ID of the network insights analysis.

            - **NetworkInsightsAnalysisArn** _(string) --_

                The Amazon Resource Name (ARN) of the network insights analysis.

            - **NetworkInsightsPathId** _(string) --_

                The ID of the path.

            - **FilterInArns** _(list) --_

                The Amazon Resource Names (ARN) of the AWS resources that the path must traverse.

                - _(string) --_
            - **StartDate** _(datetime) --_

                The time the analysis started.

            - **Status** _(string) --_

                The status of the network insights analysis.

            - **StatusMessage** _(string) --_

                The status message, if the status is failed .

            - **NetworkPathFound** _(boolean) --_

                Indicates whether the destination is reachable from the source.

            - **ForwardPathComponents** _(list) --_

                The components in the path from source to destination.

                - _(dict) --_

                    Describes a path component.

                    - **SequenceNumber** _(integer) --_

                        The sequence number.

                    - **AclRule** _(dict) --_

                        The network ACL rule.

                        - **Cidr** _(string) --_

                            The IPv4 address range, in CIDR notation.

                        - **Egress** _(boolean) --_

                            Indicates whether the rule is an outbound rule.

                        - **PortRange** _(dict) --_

                            The range of ports.

                            - **From** _(integer) --_

                                The first port in the range.

                            - **To** _(integer) --_

                                The last port in the range.

                        - **Protocol** _(string) --_

                            The protocol.

                        - **RuleAction** _(string) --_

                            Indicates whether to allow or deny traffic that matches the rule.

                        - **RuleNumber** _(integer) --_

                            The rule number.

                    - **Component** _(dict) --_

                        The component.

                        - **Id** _(string) --_

                            The ID of the component.

                        - **Arn** _(string) --_

                            The Amazon Resource Name (ARN) of the component.

                    - **DestinationVpc** _(dict) --_

                        The destination VPC.

                        - **Id** _(string) --_

                            The ID of the component.

                        - **Arn** _(string) --_

                            The Amazon Resource Name (ARN) of the component.

                    - **OutboundHeader** _(dict) --_

                        The outbound header.

                        - **DestinationAddresses** _(list) --_

                            The destination addresses.

                            - _(string) --_
                        - **DestinationPortRanges** _(list) --_

                            The destination port ranges.

                            - _(dict) --_

                                Describes a range of ports.

                                - **From** _(integer) --_

                                    The first port in the range.

                                - **To** _(integer) --_

                                    The last port in the range.

                        - **Protocol** _(string) --_

                            The protocol.

                        - **SourceAddresses** _(list) --_

                            The source addresses.

                            - _(string) --_
                        - **SourcePortRanges** _(list) --_

                            The source port ranges.

                            - _(dict) --_

                                Describes a range of ports.

                                - **From** _(integer) --_

                                    The first port in the range.

                                - **To** _(integer) --_

                                    The last port in the range.

                    - **InboundHeader** _(dict) --_

                        The inbound header.

                        - **DestinationAddresses** _(list) --_

                            The destination addresses.

                            - _(string) --_
                        - **DestinationPortRanges** _(list) --_

                            The destination port ranges.

                            - _(dict) --_

                                Describes a range of ports.

                                - **From** _(integer) --_

                                    The first port in the range.

                                - **To** _(integer) --_

                                    The last port in the range.

                        - **Protocol** _(string) --_

                            The protocol.

                        - **SourceAddresses** _(list) --_

                            The source addresses.

                            - _(string) --_
                        - **SourcePortRanges** _(list) --_

                            The source port ranges.

                            - _(dict) --_

                                Describes a range of ports.

                                - **From** _(integer) --_

                                    The first port in the range.

                                - **To** _(integer) --_

                                    The last port in the range.

                    - **RouteTableRoute** _(dict) --_

                        The route table route.

                        - **DestinationCidr** _(string) --_

                            The destination IPv4 address, in CIDR notation.

                        - **DestinationPrefixListId** _(string) --_

                            The prefix of the AWS service.

                        - **EgressOnlyInternetGatewayId** _(string) --_

                            The ID of an egress-only internet gateway.

                        - **GatewayId** _(string) --_

                            The ID of the gateway, such as an internet gateway or virtual private gateway.

                        - **InstanceId** _(string) --_

                            The ID of the instance, such as a NAT instance.

                        - **NatGatewayId** _(string) --_

                            The ID of a NAT gateway.

                        - **NetworkInterfaceId** _(string) --_

                            The ID of a network interface.

                        - **Origin** _(string) --_

                            Describes how the route was created. The following are possible values:

                            - CreateRouteTable - The route was automatically created when the route table was created.
                            - CreateRoute - The route was manually added to the route table.
                            - EnableVgwRoutePropagation - The route was propagated by route propagation.
                        - **TransitGatewayId** _(string) --_

                            The ID of a transit gateway.

                        - **VpcPeeringConnectionId** _(string) --_

                            The ID of a VPC peering connection.

                    - **SecurityGroupRule** _(dict) --_

                        The security group rule.

                        - **Cidr** _(string) --_

                            The IPv4 address range, in CIDR notation.

                        - **Direction** _(string) --_

                            The direction. The following are possible values:

                            - egress
                            - ingress
                        - **SecurityGroupId** _(string) --_

                            The security group ID.

                        - **PortRange** _(dict) --_

                            The port range.

                            - **From** _(integer) --_

                                The first port in the range.

                            - **To** _(integer) --_

                                The last port in the range.

                        - **PrefixListId** _(string) --_

                            The prefix list ID.

                        - **Protocol** _(string) --_

                            The protocol name.

                    - **SourceVpc** _(dict) --_

                        The source VPC.

                        - **Id** _(string) --_

                            The ID of the component.

                        - **Arn** _(string) --_

                            The Amazon Resource Name (ARN) of the component.

                    - **Subnet** _(dict) --_

                        The subnet.

                        - **Id** _(string) --_

                            The ID of the component.

                        - **Arn** _(string) --_

                            The Amazon Resource Name (ARN) of the component.

                    - **Vpc** _(dict) --_

                        The component VPC.

                        - **Id** _(string) --_

                            The ID of the component.

                        - **Arn** _(string) --_

                            The Amazon Resource Name (ARN) of the component.

            - **ReturnPathComponents** _(list) --_

                The components in the path from destination to source.

                - _(dict) --_

                    Describes a path component.

                    - **SequenceNumber** _(integer) --_

                        The sequence number.

                    - **AclRule** _(dict) --_

                        The network ACL rule.

                        - **Cidr** _(string) --_

                            The IPv4 address range, in CIDR notation.

                        - **Egress** _(boolean) --_

                            Indicates whether the rule is an outbound rule.

                        - **PortRange** _(dict) --_

                            The range of ports.

                            - **From** _(integer) --_

                                The first port in the range.

                            - **To** _(integer) --_

                                The last port in the range.

                        - **Protocol** _(string) --_

                            The protocol.

                        - **RuleAction** _(string) --_

                            Indicates whether to allow or deny traffic that matches the rule.

                        - **RuleNumber** _(integer) --_

                            The rule number.

                    - **Component** _(dict) --_

                        The component.

                        - **Id** _(string) --_

                            The ID of the component.

                        - **Arn** _(string) --_

                            The Amazon Resource Name (ARN) of the component.

                    - **DestinationVpc** _(dict) --_

                        The destination VPC.

                        - **Id** _(string) --_

                            The ID of the component.

                        - **Arn** _(string) --_

                            The Amazon Resource Name (ARN) of the component.

                    - **OutboundHeader** _(dict) --_

                        The outbound header.

                        - **DestinationAddresses** _(list) --_

                            The destination addresses.

                            - _(string) --_
                        - **DestinationPortRanges** _(list) --_

                            The destination port ranges.

                            - _(dict) --_

                                Describes a range of ports.

                                - **From** _(integer) --_

                                    The first port in the range.

                                - **To** _(integer) --_

                                    The last port in the range.

                        - **Protocol** _(string) --_

                            The protocol.

                        - **SourceAddresses** _(list) --_

                            The source addresses.

                            - _(string) --_
                        - **SourcePortRanges** _(list) --_

                            The source port ranges.

                            - _(dict) --_

                                Describes a range of ports.

                                - **From** _(integer) --_

                                    The first port in the range.

                                - **To** _(integer) --_

                                    The last port in the range.

                    - **InboundHeader** _(dict) --_

                        The inbound header.

                        - **DestinationAddresses** _(list) --_

                            The destination addresses.

                            - _(string) --_
                        - **DestinationPortRanges** _(list) --_

                            The destination port ranges.

                            - _(dict) --_

                                Describes a range of ports.

                                - **From** _(integer) --_

                                    The first port in the range.

                                - **To** _(integer) --_

                                    The last port in the range.

                        - **Protocol** _(string) --_

                            The protocol.

                        - **SourceAddresses** _(list) --_

                            The source addresses.

                            - _(string) --_
                        - **SourcePortRanges** _(list) --_

                            The source port ranges.

                            - _(dict) --_

                                Describes a range of ports.

                                - **From** _(integer) --_

                                    The first port in the range.

                                - **To** _(integer) --_

                                    The last port in the range.

                    - **RouteTableRoute** _(dict) --_

                        The route table route.

                        - **DestinationCidr** _(string) --_

                            The destination IPv4 address, in CIDR notation.

                        - **DestinationPrefixListId** _(string) --_

                            The prefix of the AWS service.

                        - **EgressOnlyInternetGatewayId** _(string) --_

                            The ID of an egress-only internet gateway.

                        - **GatewayId** _(string) --_

                            The ID of the gateway, such as an internet gateway or virtual private gateway.

                        - **InstanceId** _(string) --_

                            The ID of the instance, such as a NAT instance.

                        - **NatGatewayId** _(string) --_

                            The ID of a NAT gateway.

                        - **NetworkInterfaceId** _(string) --_

                            The ID of a network interface.

                        - **Origin** _(string) --_

                            Describes how the route was created. The following are possible values:

                            - CreateRouteTable - The route was automatically created when the route table was created.
                            - CreateRoute - The route was manually added to the route table.
                            - EnableVgwRoutePropagation - The route was propagated by route propagation.
                        - **TransitGatewayId** _(string) --_

                            The ID of a transit gateway.

                        - **VpcPeeringConnectionId** _(string) --_

                            The ID of a VPC peering connection.

                    - **SecurityGroupRule** _(dict) --_

                        The security group rule.

                        - **Cidr** _(string) --_

                            The IPv4 address range, in CIDR notation.

                        - **Direction** _(string) --_

                            The direction. The following are possible values:

                            - egress
                            - ingress
                        - **SecurityGroupId** _(string) --_

                            The security group ID.

                        - **PortRange** _(dict) --_

                            The port range.

                            - **From** _(integer) --_

                                The first port in the range.

                            - **To** _(integer) --_

                                The last port in the range.

                        - **PrefixListId** _(string) --_

                            The prefix list ID.

                        - **Protocol** _(string) --_

                            The protocol name.

                    - **SourceVpc** _(dict) --_

                        The source VPC.

                        - **Id** _(string) --_

                            The ID of the component.

                        - **Arn** _(string) --_

                            The Amazon Resource Name (ARN) of the component.

                    - **Subnet** _(dict) --_

                        The subnet.

                        - **Id** _(string) --_

                            The ID of the component.

                        - **Arn** _(string) --_

                            The Amazon Resource Name (ARN) of the component.

                    - **Vpc** _(dict) --_

                        The component VPC.

                        - **Id** _(string) --_

                            The ID of the component.

                        - **Arn** _(string) --_

                            The Amazon Resource Name (ARN) of the component.

            - **Explanations** _(list) --_

                The explanations. For more information, see [Reachability Analyzer explanation codes](https://docs.aws.amazon.com/vpc/latest/reachability/explanation-codes.html) .

                - _(dict) --_

                    Describes an explanation code for an unreachable path. For more information, see [Reachability Analyzer explanation codes](https://docs.aws.amazon.com/vpc/latest/reachability/explanation-codes.html) .

                    - **Acl** _(dict) --_

                        The network ACL.

                        - **Id** _(string) --_

                            The ID of the component.

                        - **Arn** _(string) --_

                            The Amazon Resource Name (ARN) of the component.

                    - **AclRule** _(dict) --_

                        The network ACL rule.

                        - **Cidr** _(string) --_

                            The IPv4 address range, in CIDR notation.

                        - **Egress** _(boolean) --_

                            Indicates whether the rule is an outbound rule.

                        - **PortRange** _(dict) --_

                            The range of ports.

                            - **From** _(integer) --_

                                The first port in the range.

                            - **To** _(integer) --_

                                The last port in the range.

                        - **Protocol** _(string) --_

                            The protocol.

                        - **RuleAction** _(string) --_

                            Indicates whether to allow or deny traffic that matches the rule.

                        - **RuleNumber** _(integer) --_

                            The rule number.

                    - **Address** _(string) --_

                        The IPv4 address, in CIDR notation.

                    - **Addresses** _(list) --_

                        The IPv4 addresses, in CIDR notation.

                        - _(string) --_
                    - **AttachedTo** _(dict) --_

                        The resource to which the component is attached.

                        - **Id** _(string) --_

                            The ID of the component.

                        - **Arn** _(string) --_

                            The Amazon Resource Name (ARN) of the component.

                    - **AvailabilityZones** _(list) --_

                        The Availability Zones.

                        - _(string) --_
                    - **Cidrs** _(list) --_

                        The CIDR ranges.

                        - _(string) --_
                    - **Component** _(dict) --_

                        The component.

                        - **Id** _(string) --_

                            The ID of the component.

                        - **Arn** _(string) --_

                            The Amazon Resource Name (ARN) of the component.

                    - **CustomerGateway** _(dict) --_

                        The customer gateway.

                        - **Id** _(string) --_

                            The ID of the component.

                        - **Arn** _(string) --_

                            The Amazon Resource Name (ARN) of the component.

                    - **Destination** _(dict) --_

                        The destination.

                        - **Id** _(string) --_

                            The ID of the component.

                        - **Arn** _(string) --_

                            The Amazon Resource Name (ARN) of the component.

                    - **DestinationVpc** _(dict) --_

                        The destination VPC.

                        - **Id** _(string) --_

                            The ID of the component.

                        - **Arn** _(string) --_

                            The Amazon Resource Name (ARN) of the component.

                    - **Direction** _(string) --_

                        The direction. The following are possible values:

                        - egress
                        - ingress
                    - **ExplanationCode** _(string) --_

                        The explanation code.

                    - **IngressRouteTable** _(dict) --_

                        The route table.

                        - **Id** _(string) --_

                            The ID of the component.

                        - **Arn** _(string) --_

                            The Amazon Resource Name (ARN) of the component.

                    - **InternetGateway** _(dict) --_

                        The internet gateway.

                        - **Id** _(string) --_

                            The ID of the component.

                        - **Arn** _(string) --_

                            The Amazon Resource Name (ARN) of the component.

                    - **LoadBalancerArn** _(string) --_

                        The Amazon Resource Name (ARN) of the load balancer.

                    - **ClassicLoadBalancerListener** _(dict) --_

                        The listener for a Classic Load Balancer.

                        - **LoadBalancerPort** _(integer) --_

                            The port on which the load balancer is listening.

                        - **InstancePort** _(integer) --_

                            [Classic Load Balancers] The back-end port for the listener.

                    - **LoadBalancerListenerPort** _(integer) --_

                        The listener port of the load balancer.

                    - **LoadBalancerTarget** _(dict) --_

                        The target.

                        - **Address** _(string) --_

                            The IP address.

                        - **AvailabilityZone** _(string) --_

                            The Availability Zone.

                        - **Instance** _(dict) --_

                            Information about the instance.

                            - **Id** _(string) --_

                                The ID of the component.

                            - **Arn** _(string) --_

                                The Amazon Resource Name (ARN) of the component.

                        - **Port** _(integer) --_

                            The port on which the target is listening.

                    - **LoadBalancerTargetGroup** _(dict) --_

                        The target group.

                        - **Id** _(string) --_

                            The ID of the component.

                        - **Arn** _(string) --_

                            The Amazon Resource Name (ARN) of the component.

                    - **LoadBalancerTargetGroups** _(list) --_

                        The target groups.

                        - _(dict) --_

                            Describes a path component.

                            - **Id** _(string) --_

                                The ID of the component.

                            - **Arn** _(string) --_

                                The Amazon Resource Name (ARN) of the component.

                    - **LoadBalancerTargetPort** _(integer) --_

                        The target port.

                    - **ElasticLoadBalancerListener** _(dict) --_

                        The load balancer listener.

                        - **Id** _(string) --_

                            The ID of the component.

                        - **Arn** _(string) --_

                            The Amazon Resource Name (ARN) of the component.

                    - **MissingComponent** _(string) --_

                        The missing component.

                    - **NatGateway** _(dict) --_

                        The NAT gateway.

                        - **Id** _(string) --_

                            The ID of the component.

                        - **Arn** _(string) --_

                            The Amazon Resource Name (ARN) of the component.

                    - **NetworkInterface** _(dict) --_

                        The network interface.

                        - **Id** _(string) --_

                            The ID of the component.

                        - **Arn** _(string) --_

                            The Amazon Resource Name (ARN) of the component.

                    - **PacketField** _(string) --_

                        The packet field.

                    - **VpcPeeringConnection** _(dict) --_

                        The VPC peering connection.

                        - **Id** _(string) --_

                            The ID of the component.

                        - **Arn** _(string) --_

                            The Amazon Resource Name (ARN) of the component.

                    - **Port** _(integer) --_

                        The port.

                    - **PortRanges** _(list) --_

                        The port ranges.

                        - _(dict) --_

                            Describes a range of ports.

                            - **From** _(integer) --_

                                The first port in the range.

                            - **To** _(integer) --_

                                The last port in the range.

                    - **PrefixList** _(dict) --_

                        The prefix list.

                        - **Id** _(string) --_

                            The ID of the component.

                        - **Arn** _(string) --_

                            The Amazon Resource Name (ARN) of the component.

                    - **Protocols** _(list) --_

                        The protocols.

                        - _(string) --_
                    - **RouteTableRoute** _(dict) --_

                        The route table route.

                        - **DestinationCidr** _(string) --_

                            The destination IPv4 address, in CIDR notation.

                        - **DestinationPrefixListId** _(string) --_

                            The prefix of the AWS service.

                        - **EgressOnlyInternetGatewayId** _(string) --_

                            The ID of an egress-only internet gateway.

                        - **GatewayId** _(string) --_

                            The ID of the gateway, such as an internet gateway or virtual private gateway.

                        - **InstanceId** _(string) --_

                            The ID of the instance, such as a NAT instance.

                        - **NatGatewayId** _(string) --_

                            The ID of a NAT gateway.

                        - **NetworkInterfaceId** _(string) --_

                            The ID of a network interface.

                        - **Origin** _(string) --_

                            Describes how the route was created. The following are possible values:

                            - CreateRouteTable - The route was automatically created when the route table was created.
                            - CreateRoute - The route was manually added to the route table.
                            - EnableVgwRoutePropagation - The route was propagated by route propagation.
                        - **TransitGatewayId** _(string) --_

                            The ID of a transit gateway.

                        - **VpcPeeringConnectionId** _(string) --_

                            The ID of a VPC peering connection.

                    - **RouteTable** _(dict) --_

                        The route table.

                        - **Id** _(string) --_

                            The ID of the component.

                        - **Arn** _(string) --_

                            The Amazon Resource Name (ARN) of the component.

                    - **SecurityGroup** _(dict) --_

                        The security group.

                        - **Id** _(string) --_

                            The ID of the component.

                        - **Arn** _(string) --_

                            The Amazon Resource Name (ARN) of the component.

                    - **SecurityGroupRule** _(dict) --_

                        The security group rule.

                        - **Cidr** _(string) --_

                            The IPv4 address range, in CIDR notation.

                        - **Direction** _(string) --_

                            The direction. The following are possible values:

                            - egress
                            - ingress
                        - **SecurityGroupId** _(string) --_

                            The security group ID.

                        - **PortRange** _(dict) --_

                            The port range.

                            - **From** _(integer) --_

                                The first port in the range.

                            - **To** _(integer) --_

                                The last port in the range.

                        - **PrefixListId** _(string) --_

                            The prefix list ID.

                        - **Protocol** _(string) --_

                            The protocol name.

                    - **SecurityGroups** _(list) --_

                        The security groups.

                        - _(dict) --_

                            Describes a path component.

                            - **Id** _(string) --_

                                The ID of the component.

                            - **Arn** _(string) --_

                                The Amazon Resource Name (ARN) of the component.

                    - **SourceVpc** _(dict) --_

                        The source VPC.

                        - **Id** _(string) --_

                            The ID of the component.

                        - **Arn** _(string) --_

                            The Amazon Resource Name (ARN) of the component.

                    - **State** _(string) --_

                        The state.

                    - **Subnet** _(dict) --_

                        The subnet.

                        - **Id** _(string) --_

                            The ID of the component.

                        - **Arn** _(string) --_

                            The Amazon Resource Name (ARN) of the component.

                    - **SubnetRouteTable** _(dict) --_

                        The route table for the subnet.

                        - **Id** _(string) --_

                            The ID of the component.

                        - **Arn** _(string) --_

                            The Amazon Resource Name (ARN) of the component.

                    - **Vpc** _(dict) --_

                        The component VPC.

                        - **Id** _(string) --_

                            The ID of the component.

                        - **Arn** _(string) --_

                            The Amazon Resource Name (ARN) of the component.

                    - **VpcEndpoint** _(dict) --_

                        The VPC endpoint.

                        - **Id** _(string) --_

                            The ID of the component.

                        - **Arn** _(string) --_

                            The Amazon Resource Name (ARN) of the component.

                    - **VpnConnection** _(dict) --_

                        The VPN connection.

                        - **Id** _(string) --_

                            The ID of the component.

                        - **Arn** _(string) --_

                            The Amazon Resource Name (ARN) of the component.

                    - **VpnGateway** _(dict) --_

                        The VPN gateway.

                        - **Id** _(string) --_

                            The ID of the component.

                        - **Arn** _(string) --_

                            The Amazon Resource Name (ARN) of the component.

            - **AlternatePathHints** _(list) --_

                Potential intermediate components.

                - _(dict) --_

                    Describes an potential intermediate component of a feasible path.

                    - **ComponentId** _(string) --_

                        The ID of the component.

                    - **ComponentArn** _(string) --_

                        The Amazon Resource Name (ARN) of the component.

            - **Tags** _(list) --_

                The tags.

                - _(dict) --_

                    Describes a tag.

                    - **Key** _(string) --_

                        The key of the tag.

                        Constraints: Tag keys are case-sensitive and accept a maximum of 127 Unicode characters. May not begin with aws: .

                    - **Value** _(string) --_

                        The value of the tag.

                        Constraints: Tag values are case-sensitive and accept a maximum of 255 Unicode characters.


_class_ EC2.Paginator.DescribeNetworkInsightsPaths

paginator = client.get_paginator('describe_network_insights_paths')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_network_insights_paths()](#EC2.Client.describe_network_insights_paths "EC2.Client.describe_network_insights_paths").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeNetworkInsightsPaths)

**Request Syntax**

response_iterator = paginator.paginate(
    NetworkInsightsPathIds=[
        'string',
    ],
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    DryRun=True|False,
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **NetworkInsightsPathIds** (_list_) --

    The IDs of the paths.

    - _(string) --_
- **Filters** (_list_) --

    The filters. The following are possible values:

    - Destination - The ID of the resource.
    - DestinationPort - The destination port.
    - Name - The path name.
    - Protocol - The protocol.
    - Source - The ID of the resource.

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'NetworkInsightsPaths': [
        {
            'NetworkInsightsPathId': 'string',
            'NetworkInsightsPathArn': 'string',
            'CreatedDate': datetime(2015, 1, 1),
            'Source': 'string',
            'Destination': 'string',
            'SourceIp': 'string',
            'DestinationIp': 'string',
            'Protocol': 'tcp'|'udp',
            'DestinationPort': 123,
            'Tags': [
                {
                    'Key': 'string',
                    'Value': 'string'
                },
            ]
        },
    ],

}

**Response Structure**

- _(dict) --_

    - **NetworkInsightsPaths** _(list) --_

        Information about the paths.

        - _(dict) --_

            Describes a path.

            - **NetworkInsightsPathId** _(string) --_

                The ID of the path.

            - **NetworkInsightsPathArn** _(string) --_

                The Amazon Resource Name (ARN) of the path.

            - **CreatedDate** _(datetime) --_

                The time stamp when the path was created.

            - **Source** _(string) --_

                The AWS resource that is the source of the path.

            - **Destination** _(string) --_

                The AWS resource that is the destination of the path.

            - **SourceIp** _(string) --_

                The IP address of the AWS resource that is the source of the path.

            - **DestinationIp** _(string) --_

                The IP address of the AWS resource that is the destination of the path.

            - **Protocol** _(string) --_

                The protocol.

            - **DestinationPort** _(integer) --_

                The destination port.

            - **Tags** _(list) --_

                The tags associated with the path.

                - _(dict) --_

                    Describes a tag.

                    - **Key** _(string) --_

                        The key of the tag.

                        Constraints: Tag keys are case-sensitive and accept a maximum of 127 Unicode characters. May not begin with aws: .

                    - **Value** _(string) --_

                        The value of the tag.

                        Constraints: Tag values are case-sensitive and accept a maximum of 255 Unicode characters.


_class_ EC2.Paginator.DescribeNetworkInterfacePermissions

paginator = client.get_paginator('describe_network_interface_permissions')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_network_interface_permissions()](#EC2.Client.describe_network_interface_permissions "EC2.Client.describe_network_interface_permissions").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeNetworkInterfacePermissions)

**Request Syntax**

response_iterator = paginator.paginate(
    NetworkInterfacePermissionIds=[
        'string',
    ],
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **NetworkInterfacePermissionIds** (_list_) --

    One or more network interface permission IDs.

    - _(string) --_
- **Filters** (_list_) --

    One or more filters.

    - network-interface-permission.network-interface-permission-id - The ID of the permission.
    - network-interface-permission.network-interface-id - The ID of the network interface.
    - network-interface-permission.aws-account-id - The AWS account ID.
    - network-interface-permission.aws-service - The AWS service.
    - network-interface-permission.permission - The type of permission (INSTANCE-ATTACH | EIP-ASSOCIATE ).

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'NetworkInterfacePermissions': [
        {
            'NetworkInterfacePermissionId': 'string',
            'NetworkInterfaceId': 'string',
            'AwsAccountId': 'string',
            'AwsService': 'string',
            'Permission': 'INSTANCE-ATTACH'|'EIP-ASSOCIATE',
            'PermissionState': {
                'State': 'pending'|'granted'|'revoking'|'revoked',
                'StatusMessage': 'string'
            }
        },
    ],

}

**Response Structure**

- _(dict) --_

    Contains the output for DescribeNetworkInterfacePermissions.

    - **NetworkInterfacePermissions** _(list) --_

        The network interface permissions.

        - _(dict) --_

            Describes a permission for a network interface.

            - **NetworkInterfacePermissionId** _(string) --_

                The ID of the network interface permission.

            - **NetworkInterfaceId** _(string) --_

                The ID of the network interface.

            - **AwsAccountId** _(string) --_

                The AWS account ID.

            - **AwsService** _(string) --_

                The AWS service.

            - **Permission** _(string) --_

                The type of permission.

            - **PermissionState** _(dict) --_

                Information about the state of the permission.

                - **State** _(string) --_

                    The state of the permission.

                - **StatusMessage** _(string) --_

                    A status message, if applicable.


_class_ EC2.Paginator.DescribeNetworkInterfaces

paginator = client.get_paginator('describe_network_interfaces')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_network_interfaces()](#EC2.Client.describe_network_interfaces "EC2.Client.describe_network_interfaces").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeNetworkInterfaces)

**Request Syntax**

response_iterator = paginator.paginate(
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    DryRun=True|False,
    NetworkInterfaceIds=[
        'string',
    ],
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **Filters** (_list_) --

    One or more filters.

    - addresses.private-ip-address - The private IPv4 addresses associated with the network interface.
    - addresses.primary - Whether the private IPv4 address is the primary IP address associated with the network interface.
    - addresses.association.public-ip - The association ID returned when the network interface was associated with the Elastic IP address (IPv4).
    - addresses.association.owner-id - The owner ID of the addresses associated with the network interface.
    - association.association-id - The association ID returned when the network interface was associated with an IPv4 address.
    - association.allocation-id - The allocation ID returned when you allocated the Elastic IP address (IPv4) for your network interface.
    - association.ip-owner-id - The owner of the Elastic IP address (IPv4) associated with the network interface.
    - association.public-ip - The address of the Elastic IP address (IPv4) bound to the network interface.
    - association.public-dns-name - The public DNS name for the network interface (IPv4).
    - attachment.attachment-id - The ID of the interface attachment.
    - attachment.attach-time - The time that the network interface was attached to an instance.
    - attachment.delete-on-termination - Indicates whether the attachment is deleted when an instance is terminated.
    - attachment.device-index - The device index to which the network interface is attached.
    - attachment.instance-id - The ID of the instance to which the network interface is attached.
    - attachment.instance-owner-id - The owner ID of the instance to which the network interface is attached.
    - attachment.status - The status of the attachment (attaching | attached | detaching | detached ).
    - availability-zone - The Availability Zone of the network interface.
    - description - The description of the network interface.
    - group-id - The ID of a security group associated with the network interface.
    - group-name - The name of a security group associated with the network interface.
    - ipv6-addresses.ipv6-address - An IPv6 address associated with the network interface.
    - mac-address - The MAC address of the network interface.
    - network-interface-id - The ID of the network interface.
    - owner-id - The AWS account ID of the network interface owner.
    - private-ip-address - The private IPv4 address or addresses of the network interface.
    - private-dns-name - The private DNS name of the network interface (IPv4).
    - requester-id - The ID of the entity that launched the instance on your behalf (for example, AWS Management Console, Auto Scaling, and so on).
    - requester-managed - Indicates whether the network interface is being managed by an AWS service (for example, AWS Management Console, Auto Scaling, and so on).
    - source-dest-check - Indicates whether the network interface performs source/destination checking. A value of true means checking is enabled, and false means checking is disabled. The value must be false for the network interface to perform network address translation (NAT) in your VPC.
    - status - The status of the network interface. If the network interface is not attached to an instance, the status is available ; if a network interface is attached to an instance the status is in-use .
    - subnet-id - The ID of the subnet for the network interface.
    - tag :<key> - The key/value combination of a tag assigned to the resource. Use the tag key in the filter name and the tag value as the filter value. For example, to find all resources that have a tag with the key Owner and the value TeamA , specify tag:Owner for the filter name and TeamA for the filter value.
    - tag-key - The key of a tag assigned to the resource. Use this filter to find all resources assigned a tag with a specific key, regardless of the tag value.
    - vpc-id - The ID of the VPC for the network interface.

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **NetworkInterfaceIds** (_list_) --

    One or more network interface IDs.

    Default: Describes all your network interfaces.

    - _(string) --_
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'NetworkInterfaces': [
        {
            'Association': {
                'AllocationId': 'string',
                'AssociationId': 'string',
                'IpOwnerId': 'string',
                'PublicDnsName': 'string',
                'PublicIp': 'string',
                'CustomerOwnedIp': 'string',
                'CarrierIp': 'string'
            },
            'Attachment': {
                'AttachTime': datetime(2015, 1, 1),
                'AttachmentId': 'string',
                'DeleteOnTermination': True|False,
                'DeviceIndex': 123,
                'NetworkCardIndex': 123,
                'InstanceId': 'string',
                'InstanceOwnerId': 'string',
                'Status': 'attaching'|'attached'|'detaching'|'detached'
            },
            'AvailabilityZone': 'string',
            'Description': 'string',
            'Groups': [
                {
                    'GroupName': 'string',
                    'GroupId': 'string'
                },
            ],
            'InterfaceType': 'interface'|'natGateway'|'efa',
            'Ipv6Addresses': [
                {
                    'Ipv6Address': 'string'
                },
            ],
            'MacAddress': 'string',
            'NetworkInterfaceId': 'string',
            'OutpostArn': 'string',
            'OwnerId': 'string',
            'PrivateDnsName': 'string',
            'PrivateIpAddress': 'string',
            'PrivateIpAddresses': [
                {
                    'Association': {
                        'AllocationId': 'string',
                        'AssociationId': 'string',
                        'IpOwnerId': 'string',
                        'PublicDnsName': 'string',
                        'PublicIp': 'string',
                        'CustomerOwnedIp': 'string',
                        'CarrierIp': 'string'
                    },
                    'Primary': True|False,
                    'PrivateDnsName': 'string',
                    'PrivateIpAddress': 'string'
                },
            ],
            'RequesterId': 'string',
            'RequesterManaged': True|False,
            'SourceDestCheck': True|False,
            'Status': 'available'|'associated'|'attaching'|'in-use'|'detaching',
            'SubnetId': 'string',
            'TagSet': [
                {
                    'Key': 'string',
                    'Value': 'string'
                },
            ],
            'VpcId': 'string'
        },
    ],

}

**Response Structure**

- _(dict) --_

    Contains the output of DescribeNetworkInterfaces.

    - **NetworkInterfaces** _(list) --_

        Information about one or more network interfaces.

        - _(dict) --_

            Describes a network interface.

            - **Association** _(dict) --_

                The association information for an Elastic IP address (IPv4) associated with the network interface.

                - **AllocationId** _(string) --_

                    The allocation ID.

                - **AssociationId** _(string) --_

                    The association ID.

                - **IpOwnerId** _(string) --_

                    The ID of the Elastic IP address owner.

                - **PublicDnsName** _(string) --_

                    The public DNS name.

                - **PublicIp** _(string) --_

                    The address of the Elastic IP address bound to the network interface.

                - **CustomerOwnedIp** _(string) --_

                    The customer-owned IP address associated with the network interface.

                - **CarrierIp** _(string) --_

                    The carrier IP address associated with the network interface.

                    This option is only available when the network interface is in a subnet which is associated with a Wavelength Zone.

            - **Attachment** _(dict) --_

                The network interface attachment.

                - **AttachTime** _(datetime) --_

                    The timestamp indicating when the attachment initiated.

                - **AttachmentId** _(string) --_

                    The ID of the network interface attachment.

                - **DeleteOnTermination** _(boolean) --_

                    Indicates whether the network interface is deleted when the instance is terminated.

                - **DeviceIndex** _(integer) --_

                    The device index of the network interface attachment on the instance.

                - **NetworkCardIndex** _(integer) --_

                    The index of the network card.

                - **InstanceId** _(string) --_

                    The ID of the instance.

                - **InstanceOwnerId** _(string) --_

                    The AWS account ID of the owner of the instance.

                - **Status** _(string) --_

                    The attachment state.

            - **AvailabilityZone** _(string) --_

                The Availability Zone.

            - **Description** _(string) --_

                A description.

            - **Groups** _(list) --_

                Any security groups for the network interface.

                - _(dict) --_

                    Describes a security group.

                    - **GroupName** _(string) --_

                        The name of the security group.

                    - **GroupId** _(string) --_

                        The ID of the security group.

            - **InterfaceType** _(string) --_

                The type of network interface.

            - **Ipv6Addresses** _(list) --_

                The IPv6 addresses associated with the network interface.

                - _(dict) --_

                    Describes an IPv6 address associated with a network interface.

                    - **Ipv6Address** _(string) --_

                        The IPv6 address.

            - **MacAddress** _(string) --_

                The MAC address.

            - **NetworkInterfaceId** _(string) --_

                The ID of the network interface.

            - **OutpostArn** _(string) --_

                The Amazon Resource Name (ARN) of the Outpost.

            - **OwnerId** _(string) --_

                The AWS account ID of the owner of the network interface.

            - **PrivateDnsName** _(string) --_

                The private DNS name.

            - **PrivateIpAddress** _(string) --_

                The IPv4 address of the network interface within the subnet.

            - **PrivateIpAddresses** _(list) --_

                The private IPv4 addresses associated with the network interface.

                - _(dict) --_

                    Describes the private IPv4 address of a network interface.

                    - **Association** _(dict) --_

                        The association information for an Elastic IP address (IPv4) associated with the network interface.

                        - **AllocationId** _(string) --_

                            The allocation ID.

                        - **AssociationId** _(string) --_

                            The association ID.

                        - **IpOwnerId** _(string) --_

                            The ID of the Elastic IP address owner.

                        - **PublicDnsName** _(string) --_

                            The public DNS name.

                        - **PublicIp** _(string) --_

                            The address of the Elastic IP address bound to the network interface.

                        - **CustomerOwnedIp** _(string) --_

                            The customer-owned IP address associated with the network interface.

                        - **CarrierIp** _(string) --_

                            The carrier IP address associated with the network interface.

                            This option is only available when the network interface is in a subnet which is associated with a Wavelength Zone.

                    - **Primary** _(boolean) --_

                        Indicates whether this IPv4 address is the primary private IPv4 address of the network interface.

                    - **PrivateDnsName** _(string) --_

                        The private DNS name.

                    - **PrivateIpAddress** _(string) --_

                        The private IPv4 address.

            - **RequesterId** _(string) --_

                The ID of the entity that launched the instance on your behalf (for example, AWS Management Console or Auto Scaling).

            - **RequesterManaged** _(boolean) --_

                Indicates whether the network interface is being managed by AWS.

            - **SourceDestCheck** _(boolean) --_

                Indicates whether traffic to or from the instance is validated.

            - **Status** _(string) --_

                The status of the network interface.

            - **SubnetId** _(string) --_

                The ID of the subnet.

            - **TagSet** _(list) --_

                Any tags assigned to the network interface.

                - _(dict) --_

                    Describes a tag.

                    - **Key** _(string) --_

                        The key of the tag.

                        Constraints: Tag keys are case-sensitive and accept a maximum of 127 Unicode characters. May not begin with aws: .

                    - **Value** _(string) --_

                        The value of the tag.

                        Constraints: Tag values are case-sensitive and accept a maximum of 255 Unicode characters.

            - **VpcId** _(string) --_

                The ID of the VPC.


_class_ EC2.Paginator.DescribePrefixLists

paginator = client.get_paginator('describe_prefix_lists')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_prefix_lists()](#EC2.Client.describe_prefix_lists "EC2.Client.describe_prefix_lists").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribePrefixLists)

**Request Syntax**

response_iterator = paginator.paginate(
    DryRun=True|False,
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    PrefixListIds=[
        'string',
    ],
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **Filters** (_list_) --

    One or more filters.

    - prefix-list-id : The ID of a prefix list.
    - prefix-list-name : The name of a prefix list.

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **PrefixListIds** (_list_) --

    One or more prefix list IDs.

    - _(string) --_
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'PrefixLists': [
        {
            'Cidrs': [
                'string',
            ],
            'PrefixListId': 'string',
            'PrefixListName': 'string'
        },
    ]
}

**Response Structure**

- _(dict) --_

    - **PrefixLists** _(list) --_

        All available prefix lists.

        - _(dict) --_

            Describes prefixes for AWS services.

            - **Cidrs** _(list) --_

                The IP address range of the AWS service.

                - _(string) --_
            - **PrefixListId** _(string) --_

                The ID of the prefix.

            - **PrefixListName** _(string) --_

                The name of the prefix.


_class_ EC2.Paginator.DescribePrincipalIdFormat

paginator = client.get_paginator('describe_principal_id_format')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_principal_id_format()](#EC2.Client.describe_principal_id_format "EC2.Client.describe_principal_id_format").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribePrincipalIdFormat)

**Request Syntax**

response_iterator = paginator.paginate(
    DryRun=True|False,
    Resources=[
        'string',
    ],
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **Resources** (_list_) --

    The type of resource: bundle | conversion-task | customer-gateway | dhcp-options | elastic-ip-allocation | elastic-ip-association | export-task | flow-log | image | import-task | instance | internet-gateway | network-acl | network-acl-association | network-interface | network-interface-attachment | prefix-list | reservation | route-table | route-table-association | security-group | snapshot | subnet | subnet-cidr-block-association | volume | vpc | vpc-cidr-block-association | vpc-endpoint | vpc-peering-connection | vpn-connection | vpn-gateway

    - _(string) --_
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'Principals': [
        {
            'Arn': 'string',
            'Statuses': [
                {
                    'Deadline': datetime(2015, 1, 1),
                    'Resource': 'string',
                    'UseLongIds': True|False
                },
            ]
        },
    ],

}

**Response Structure**

- _(dict) --_

    - **Principals** _(list) --_

        Information about the ID format settings for the ARN.

        - _(dict) --_

            PrincipalIdFormat description

            - **Arn** _(string) --_

                PrincipalIdFormatARN description

            - **Statuses** _(list) --_

                PrincipalIdFormatStatuses description

                - _(dict) --_

                    Describes the ID format for a resource.

                    - **Deadline** _(datetime) --_

                        The date in UTC at which you are permanently switched over to using longer IDs. If a deadline is not yet available for this resource type, this field is not returned.

                    - **Resource** _(string) --_

                        The type of resource.

                    - **UseLongIds** _(boolean) --_

                        Indicates whether longer IDs (17-character IDs) are enabled for the resource.


_class_ EC2.Paginator.DescribePublicIpv4Pools

paginator = client.get_paginator('describe_public_ipv4_pools')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_public_ipv4_pools()](#EC2.Client.describe_public_ipv4_pools "EC2.Client.describe_public_ipv4_pools").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribePublicIpv4Pools)

**Request Syntax**

response_iterator = paginator.paginate(
    PoolIds=[
        'string',
    ],
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **PoolIds** (_list_) --

    The IDs of the address pools.

    - _(string) --_
- **Filters** (_list_) --

    One or more filters.

    - tag :<key> - The key/value combination of a tag assigned to the resource. Use the tag key in the filter name and the tag value as the filter value. For example, to find all resources that have a tag with the key Owner and the value TeamA , specify tag:Owner for the filter name and TeamA for the filter value.
    - tag-key - The key of a tag assigned to the resource. Use this filter to find all resources assigned a tag with a specific key, regardless of the tag value.

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'PublicIpv4Pools': [
        {
            'PoolId': 'string',
            'Description': 'string',
            'PoolAddressRanges': [
                {
                    'FirstAddress': 'string',
                    'LastAddress': 'string',
                    'AddressCount': 123,
                    'AvailableAddressCount': 123
                },
            ],
            'TotalAddressCount': 123,
            'TotalAvailableAddressCount': 123,
            'NetworkBorderGroup': 'string',
            'Tags': [
                {
                    'Key': 'string',
                    'Value': 'string'
                },
            ]
        },
    ],

}

**Response Structure**

- _(dict) --_

    - **PublicIpv4Pools** _(list) --_

        Information about the address pools.

        - _(dict) --_

            Describes an IPv4 address pool.

            - **PoolId** _(string) --_

                The ID of the address pool.

            - **Description** _(string) --_

                A description of the address pool.

            - **PoolAddressRanges** _(list) --_

                The address ranges.

                - _(dict) --_

                    Describes an address range of an IPv4 address pool.

                    - **FirstAddress** _(string) --_

                        The first IP address in the range.

                    - **LastAddress** _(string) --_

                        The last IP address in the range.

                    - **AddressCount** _(integer) --_

                        The number of addresses in the range.

                    - **AvailableAddressCount** _(integer) --_

                        The number of available addresses in the range.

            - **TotalAddressCount** _(integer) --_

                The total number of addresses.

            - **TotalAvailableAddressCount** _(integer) --_

                The total number of available addresses.

            - **NetworkBorderGroup** _(string) --_

                The name of the location from which the address pool is advertised. A network border group is a unique set of Availability Zones or Local Zones from where AWS advertises public IP addresses.

            - **Tags** _(list) --_

                Any tags for the address pool.

                - _(dict) --_

                    Describes a tag.

                    - **Key** _(string) --_

                        The key of the tag.

                        Constraints: Tag keys are case-sensitive and accept a maximum of 127 Unicode characters. May not begin with aws: .

                    - **Value** _(string) --_

                        The value of the tag.

                        Constraints: Tag values are case-sensitive and accept a maximum of 255 Unicode characters.


_class_ EC2.Paginator.DescribeReservedInstancesModifications

paginator = client.get_paginator('describe_reserved_instances_modifications')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_reserved_instances_modifications()](#EC2.Client.describe_reserved_instances_modifications "EC2.Client.describe_reserved_instances_modifications").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeReservedInstancesModifications)

**Request Syntax**

response_iterator = paginator.paginate(
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    ReservedInstancesModificationIds=[
        'string',
    ],
    PaginationConfig={
        'MaxItems': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **Filters** (_list_) --

    One or more filters.

    - client-token - The idempotency token for the modification request.
    - create-date - The time when the modification request was created.
    - effective-date - The time when the modification becomes effective.
    - modification-result.reserved-instances-id - The ID for the Reserved Instances created as part of the modification request. This ID is only available when the status of the modification is fulfilled .
    - modification-result.target-configuration.availability-zone - The Availability Zone for the new Reserved Instances.
    - modification-result.target-configuration.instance-count - The number of new Reserved Instances.
    - modification-result.target-configuration.instance-type - The instance type of the new Reserved Instances.
    - modification-result.target-configuration.platform - The network platform of the new Reserved Instances (EC2-Classic | EC2-VPC ).
    - reserved-instances-id - The ID of the Reserved Instances modified.
    - reserved-instances-modification-id - The ID of the modification request.
    - status - The status of the Reserved Instances modification request (processing | fulfilled | failed ).
    - status-message - The reason for the status.
    - update-date - The time when the modification request was last updated.

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **ReservedInstancesModificationIds** (_list_) --

    IDs for the submitted modification request.

    - _(string) --_
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'ReservedInstancesModifications': [
        {
            'ClientToken': 'string',
            'CreateDate': datetime(2015, 1, 1),
            'EffectiveDate': datetime(2015, 1, 1),
            'ModificationResults': [
                {
                    'ReservedInstancesId': 'string',
                    'TargetConfiguration': {
                        'AvailabilityZone': 'string',
                        'InstanceCount': 123,
                        'InstanceType': 't1.micro'|'t2.nano'|'t2.micro'|'t2.small'|'t2.medium'|'t2.large'|'t2.xlarge'|'t2.2xlarge'|'t3.nano'|'t3.micro'|'t3.small'|'t3.medium'|'t3.large'|'t3.xlarge'|'t3.2xlarge'|'t3a.nano'|'t3a.micro'|'t3a.small'|'t3a.medium'|'t3a.large'|'t3a.xlarge'|'t3a.2xlarge'|'t4g.nano'|'t4g.micro'|'t4g.small'|'t4g.medium'|'t4g.large'|'t4g.xlarge'|'t4g.2xlarge'|'m1.small'|'m1.medium'|'m1.large'|'m1.xlarge'|'m3.medium'|'m3.large'|'m3.xlarge'|'m3.2xlarge'|'m4.large'|'m4.xlarge'|'m4.2xlarge'|'m4.4xlarge'|'m4.10xlarge'|'m4.16xlarge'|'m2.xlarge'|'m2.2xlarge'|'m2.4xlarge'|'cr1.8xlarge'|'r3.large'|'r3.xlarge'|'r3.2xlarge'|'r3.4xlarge'|'r3.8xlarge'|'r4.large'|'r4.xlarge'|'r4.2xlarge'|'r4.4xlarge'|'r4.8xlarge'|'r4.16xlarge'|'r5.large'|'r5.xlarge'|'r5.2xlarge'|'r5.4xlarge'|'r5.8xlarge'|'r5.12xlarge'|'r5.16xlarge'|'r5.24xlarge'|'r5.metal'|'r5a.large'|'r5a.xlarge'|'r5a.2xlarge'|'r5a.4xlarge'|'r5a.8xlarge'|'r5a.12xlarge'|'r5a.16xlarge'|'r5a.24xlarge'|'r5b.large'|'r5b.xlarge'|'r5b.2xlarge'|'r5b.4xlarge'|'r5b.8xlarge'|'r5b.12xlarge'|'r5b.16xlarge'|'r5b.24xlarge'|'r5b.metal'|'r5d.large'|'r5d.xlarge'|'r5d.2xlarge'|'r5d.4xlarge'|'r5d.8xlarge'|'r5d.12xlarge'|'r5d.16xlarge'|'r5d.24xlarge'|'r5d.metal'|'r5ad.large'|'r5ad.xlarge'|'r5ad.2xlarge'|'r5ad.4xlarge'|'r5ad.8xlarge'|'r5ad.12xlarge'|'r5ad.16xlarge'|'r5ad.24xlarge'|'r6g.metal'|'r6g.medium'|'r6g.large'|'r6g.xlarge'|'r6g.2xlarge'|'r6g.4xlarge'|'r6g.8xlarge'|'r6g.12xlarge'|'r6g.16xlarge'|'r6gd.metal'|'r6gd.medium'|'r6gd.large'|'r6gd.xlarge'|'r6gd.2xlarge'|'r6gd.4xlarge'|'r6gd.8xlarge'|'r6gd.12xlarge'|'r6gd.16xlarge'|'x1.16xlarge'|'x1.32xlarge'|'x1e.xlarge'|'x1e.2xlarge'|'x1e.4xlarge'|'x1e.8xlarge'|'x1e.16xlarge'|'x1e.32xlarge'|'i2.xlarge'|'i2.2xlarge'|'i2.4xlarge'|'i2.8xlarge'|'i3.large'|'i3.xlarge'|'i3.2xlarge'|'i3.4xlarge'|'i3.8xlarge'|'i3.16xlarge'|'i3.metal'|'i3en.large'|'i3en.xlarge'|'i3en.2xlarge'|'i3en.3xlarge'|'i3en.6xlarge'|'i3en.12xlarge'|'i3en.24xlarge'|'i3en.metal'|'hi1.4xlarge'|'hs1.8xlarge'|'c1.medium'|'c1.xlarge'|'c3.large'|'c3.xlarge'|'c3.2xlarge'|'c3.4xlarge'|'c3.8xlarge'|'c4.large'|'c4.xlarge'|'c4.2xlarge'|'c4.4xlarge'|'c4.8xlarge'|'c5.large'|'c5.xlarge'|'c5.2xlarge'|'c5.4xlarge'|'c5.9xlarge'|'c5.12xlarge'|'c5.18xlarge'|'c5.24xlarge'|'c5.metal'|'c5a.large'|'c5a.xlarge'|'c5a.2xlarge'|'c5a.4xlarge'|'c5a.8xlarge'|'c5a.12xlarge'|'c5a.16xlarge'|'c5a.24xlarge'|'c5ad.large'|'c5ad.xlarge'|'c5ad.2xlarge'|'c5ad.4xlarge'|'c5ad.8xlarge'|'c5ad.12xlarge'|'c5ad.16xlarge'|'c5ad.24xlarge'|'c5d.large'|'c5d.xlarge'|'c5d.2xlarge'|'c5d.4xlarge'|'c5d.9xlarge'|'c5d.12xlarge'|'c5d.18xlarge'|'c5d.24xlarge'|'c5d.metal'|'c5n.large'|'c5n.xlarge'|'c5n.2xlarge'|'c5n.4xlarge'|'c5n.9xlarge'|'c5n.18xlarge'|'c5n.metal'|'c6g.metal'|'c6g.medium'|'c6g.large'|'c6g.xlarge'|'c6g.2xlarge'|'c6g.4xlarge'|'c6g.8xlarge'|'c6g.12xlarge'|'c6g.16xlarge'|'c6gd.metal'|'c6gd.medium'|'c6gd.large'|'c6gd.xlarge'|'c6gd.2xlarge'|'c6gd.4xlarge'|'c6gd.8xlarge'|'c6gd.12xlarge'|'c6gd.16xlarge'|'c6gn.medium'|'c6gn.large'|'c6gn.xlarge'|'c6gn.2xlarge'|'c6gn.4xlarge'|'c6gn.8xlarge'|'c6gn.12xlarge'|'c6gn.16xlarge'|'cc1.4xlarge'|'cc2.8xlarge'|'g2.2xlarge'|'g2.8xlarge'|'g3.4xlarge'|'g3.8xlarge'|'g3.16xlarge'|'g3s.xlarge'|'g4ad.4xlarge'|'g4ad.8xlarge'|'g4ad.16xlarge'|'g4dn.xlarge'|'g4dn.2xlarge'|'g4dn.4xlarge'|'g4dn.8xlarge'|'g4dn.12xlarge'|'g4dn.16xlarge'|'g4dn.metal'|'cg1.4xlarge'|'p2.xlarge'|'p2.8xlarge'|'p2.16xlarge'|'p3.2xlarge'|'p3.8xlarge'|'p3.16xlarge'|'p3dn.24xlarge'|'p4d.24xlarge'|'d2.xlarge'|'d2.2xlarge'|'d2.4xlarge'|'d2.8xlarge'|'d3.xlarge'|'d3.2xlarge'|'d3.4xlarge'|'d3.8xlarge'|'d3en.xlarge'|'d3en.2xlarge'|'d3en.4xlarge'|'d3en.6xlarge'|'d3en.8xlarge'|'d3en.12xlarge'|'f1.2xlarge'|'f1.4xlarge'|'f1.16xlarge'|'m5.large'|'m5.xlarge'|'m5.2xlarge'|'m5.4xlarge'|'m5.8xlarge'|'m5.12xlarge'|'m5.16xlarge'|'m5.24xlarge'|'m5.metal'|'m5a.large'|'m5a.xlarge'|'m5a.2xlarge'|'m5a.4xlarge'|'m5a.8xlarge'|'m5a.12xlarge'|'m5a.16xlarge'|'m5a.24xlarge'|'m5d.large'|'m5d.xlarge'|'m5d.2xlarge'|'m5d.4xlarge'|'m5d.8xlarge'|'m5d.12xlarge'|'m5d.16xlarge'|'m5d.24xlarge'|'m5d.metal'|'m5ad.large'|'m5ad.xlarge'|'m5ad.2xlarge'|'m5ad.4xlarge'|'m5ad.8xlarge'|'m5ad.12xlarge'|'m5ad.16xlarge'|'m5ad.24xlarge'|'m5zn.large'|'m5zn.xlarge'|'m5zn.2xlarge'|'m5zn.3xlarge'|'m5zn.6xlarge'|'m5zn.12xlarge'|'m5zn.metal'|'h1.2xlarge'|'h1.4xlarge'|'h1.8xlarge'|'h1.16xlarge'|'z1d.large'|'z1d.xlarge'|'z1d.2xlarge'|'z1d.3xlarge'|'z1d.6xlarge'|'z1d.12xlarge'|'z1d.metal'|'u-6tb1.metal'|'u-9tb1.metal'|'u-12tb1.metal'|'u-18tb1.metal'|'u-24tb1.metal'|'a1.medium'|'a1.large'|'a1.xlarge'|'a1.2xlarge'|'a1.4xlarge'|'a1.metal'|'m5dn.large'|'m5dn.xlarge'|'m5dn.2xlarge'|'m5dn.4xlarge'|'m5dn.8xlarge'|'m5dn.12xlarge'|'m5dn.16xlarge'|'m5dn.24xlarge'|'m5n.large'|'m5n.xlarge'|'m5n.2xlarge'|'m5n.4xlarge'|'m5n.8xlarge'|'m5n.12xlarge'|'m5n.16xlarge'|'m5n.24xlarge'|'r5dn.large'|'r5dn.xlarge'|'r5dn.2xlarge'|'r5dn.4xlarge'|'r5dn.8xlarge'|'r5dn.12xlarge'|'r5dn.16xlarge'|'r5dn.24xlarge'|'r5n.large'|'r5n.xlarge'|'r5n.2xlarge'|'r5n.4xlarge'|'r5n.8xlarge'|'r5n.12xlarge'|'r5n.16xlarge'|'r5n.24xlarge'|'inf1.xlarge'|'inf1.2xlarge'|'inf1.6xlarge'|'inf1.24xlarge'|'m6g.metal'|'m6g.medium'|'m6g.large'|'m6g.xlarge'|'m6g.2xlarge'|'m6g.4xlarge'|'m6g.8xlarge'|'m6g.12xlarge'|'m6g.16xlarge'|'m6gd.metal'|'m6gd.medium'|'m6gd.large'|'m6gd.xlarge'|'m6gd.2xlarge'|'m6gd.4xlarge'|'m6gd.8xlarge'|'m6gd.12xlarge'|'m6gd.16xlarge'|'mac1.metal',
                        'Platform': 'string',
                        'Scope': 'Availability Zone'|'Region'
                    }
                },
            ],
            'ReservedInstancesIds': [
                {
                    'ReservedInstancesId': 'string'
                },
            ],
            'ReservedInstancesModificationId': 'string',
            'Status': 'string',
            'StatusMessage': 'string',
            'UpdateDate': datetime(2015, 1, 1)
        },
    ]
}

**Response Structure**

- _(dict) --_

    Contains the output of DescribeReservedInstancesModifications.

    - **ReservedInstancesModifications** _(list) --_

        The Reserved Instance modification information.

        - _(dict) --_

            Describes a Reserved Instance modification.

            - **ClientToken** _(string) --_

                A unique, case-sensitive key supplied by the client to ensure that the request is idempotent. For more information, see [Ensuring Idempotency](https://docs.aws.amazon.com/AWSEC2/latest/APIReference/Run_Instance_Idempotency.html) .

            - **CreateDate** _(datetime) --_

                The time when the modification request was created.

            - **EffectiveDate** _(datetime) --_

                The time for the modification to become effective.

            - **ModificationResults** _(list) --_

                Contains target configurations along with their corresponding new Reserved Instance IDs.

                - _(dict) --_

                    Describes the modification request/s.

                    - **ReservedInstancesId** _(string) --_

                        The ID for the Reserved Instances that were created as part of the modification request. This field is only available when the modification is fulfilled.

                    - **TargetConfiguration** _(dict) --_

                        The target Reserved Instances configurations supplied as part of the modification request.

                        - **AvailabilityZone** _(string) --_

                            The Availability Zone for the modified Reserved Instances.

                        - **InstanceCount** _(integer) --_

                            The number of modified Reserved Instances.

                            Note

                            This is a required field for a request.

                        - **InstanceType** _(string) --_

                            The instance type for the modified Reserved Instances.

                        - **Platform** _(string) --_

                            The network platform of the modified Reserved Instances, which is either EC2-Classic or EC2-VPC.

                        - **Scope** _(string) --_

                            Whether the Reserved Instance is applied to instances in a Region or instances in a specific Availability Zone.

            - **ReservedInstancesIds** _(list) --_

                The IDs of one or more Reserved Instances.

                - _(dict) --_

                    Describes the ID of a Reserved Instance.

                    - **ReservedInstancesId** _(string) --_

                        The ID of the Reserved Instance.

            - **ReservedInstancesModificationId** _(string) --_

                A unique ID for the Reserved Instance modification.

            - **Status** _(string) --_

                The status of the Reserved Instances modification request.

            - **StatusMessage** _(string) --_

                The reason for the status.

            - **UpdateDate** _(datetime) --_

                The time when the modification request was last updated.


_class_ EC2.Paginator.DescribeReservedInstancesOfferings

paginator = client.get_paginator('describe_reserved_instances_offerings')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_reserved_instances_offerings()](#EC2.Client.describe_reserved_instances_offerings "EC2.Client.describe_reserved_instances_offerings").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeReservedInstancesOfferings)

**Request Syntax**

response_iterator = paginator.paginate(
    AvailabilityZone='string',
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    IncludeMarketplace=True|False,
    InstanceType='t1.micro'|'t2.nano'|'t2.micro'|'t2.small'|'t2.medium'|'t2.large'|'t2.xlarge'|'t2.2xlarge'|'t3.nano'|'t3.micro'|'t3.small'|'t3.medium'|'t3.large'|'t3.xlarge'|'t3.2xlarge'|'t3a.nano'|'t3a.micro'|'t3a.small'|'t3a.medium'|'t3a.large'|'t3a.xlarge'|'t3a.2xlarge'|'t4g.nano'|'t4g.micro'|'t4g.small'|'t4g.medium'|'t4g.large'|'t4g.xlarge'|'t4g.2xlarge'|'m1.small'|'m1.medium'|'m1.large'|'m1.xlarge'|'m3.medium'|'m3.large'|'m3.xlarge'|'m3.2xlarge'|'m4.large'|'m4.xlarge'|'m4.2xlarge'|'m4.4xlarge'|'m4.10xlarge'|'m4.16xlarge'|'m2.xlarge'|'m2.2xlarge'|'m2.4xlarge'|'cr1.8xlarge'|'r3.large'|'r3.xlarge'|'r3.2xlarge'|'r3.4xlarge'|'r3.8xlarge'|'r4.large'|'r4.xlarge'|'r4.2xlarge'|'r4.4xlarge'|'r4.8xlarge'|'r4.16xlarge'|'r5.large'|'r5.xlarge'|'r5.2xlarge'|'r5.4xlarge'|'r5.8xlarge'|'r5.12xlarge'|'r5.16xlarge'|'r5.24xlarge'|'r5.metal'|'r5a.large'|'r5a.xlarge'|'r5a.2xlarge'|'r5a.4xlarge'|'r5a.8xlarge'|'r5a.12xlarge'|'r5a.16xlarge'|'r5a.24xlarge'|'r5b.large'|'r5b.xlarge'|'r5b.2xlarge'|'r5b.4xlarge'|'r5b.8xlarge'|'r5b.12xlarge'|'r5b.16xlarge'|'r5b.24xlarge'|'r5b.metal'|'r5d.large'|'r5d.xlarge'|'r5d.2xlarge'|'r5d.4xlarge'|'r5d.8xlarge'|'r5d.12xlarge'|'r5d.16xlarge'|'r5d.24xlarge'|'r5d.metal'|'r5ad.large'|'r5ad.xlarge'|'r5ad.2xlarge'|'r5ad.4xlarge'|'r5ad.8xlarge'|'r5ad.12xlarge'|'r5ad.16xlarge'|'r5ad.24xlarge'|'r6g.metal'|'r6g.medium'|'r6g.large'|'r6g.xlarge'|'r6g.2xlarge'|'r6g.4xlarge'|'r6g.8xlarge'|'r6g.12xlarge'|'r6g.16xlarge'|'r6gd.metal'|'r6gd.medium'|'r6gd.large'|'r6gd.xlarge'|'r6gd.2xlarge'|'r6gd.4xlarge'|'r6gd.8xlarge'|'r6gd.12xlarge'|'r6gd.16xlarge'|'x1.16xlarge'|'x1.32xlarge'|'x1e.xlarge'|'x1e.2xlarge'|'x1e.4xlarge'|'x1e.8xlarge'|'x1e.16xlarge'|'x1e.32xlarge'|'i2.xlarge'|'i2.2xlarge'|'i2.4xlarge'|'i2.8xlarge'|'i3.large'|'i3.xlarge'|'i3.2xlarge'|'i3.4xlarge'|'i3.8xlarge'|'i3.16xlarge'|'i3.metal'|'i3en.large'|'i3en.xlarge'|'i3en.2xlarge'|'i3en.3xlarge'|'i3en.6xlarge'|'i3en.12xlarge'|'i3en.24xlarge'|'i3en.metal'|'hi1.4xlarge'|'hs1.8xlarge'|'c1.medium'|'c1.xlarge'|'c3.large'|'c3.xlarge'|'c3.2xlarge'|'c3.4xlarge'|'c3.8xlarge'|'c4.large'|'c4.xlarge'|'c4.2xlarge'|'c4.4xlarge'|'c4.8xlarge'|'c5.large'|'c5.xlarge'|'c5.2xlarge'|'c5.4xlarge'|'c5.9xlarge'|'c5.12xlarge'|'c5.18xlarge'|'c5.24xlarge'|'c5.metal'|'c5a.large'|'c5a.xlarge'|'c5a.2xlarge'|'c5a.4xlarge'|'c5a.8xlarge'|'c5a.12xlarge'|'c5a.16xlarge'|'c5a.24xlarge'|'c5ad.large'|'c5ad.xlarge'|'c5ad.2xlarge'|'c5ad.4xlarge'|'c5ad.8xlarge'|'c5ad.12xlarge'|'c5ad.16xlarge'|'c5ad.24xlarge'|'c5d.large'|'c5d.xlarge'|'c5d.2xlarge'|'c5d.4xlarge'|'c5d.9xlarge'|'c5d.12xlarge'|'c5d.18xlarge'|'c5d.24xlarge'|'c5d.metal'|'c5n.large'|'c5n.xlarge'|'c5n.2xlarge'|'c5n.4xlarge'|'c5n.9xlarge'|'c5n.18xlarge'|'c5n.metal'|'c6g.metal'|'c6g.medium'|'c6g.large'|'c6g.xlarge'|'c6g.2xlarge'|'c6g.4xlarge'|'c6g.8xlarge'|'c6g.12xlarge'|'c6g.16xlarge'|'c6gd.metal'|'c6gd.medium'|'c6gd.large'|'c6gd.xlarge'|'c6gd.2xlarge'|'c6gd.4xlarge'|'c6gd.8xlarge'|'c6gd.12xlarge'|'c6gd.16xlarge'|'c6gn.medium'|'c6gn.large'|'c6gn.xlarge'|'c6gn.2xlarge'|'c6gn.4xlarge'|'c6gn.8xlarge'|'c6gn.12xlarge'|'c6gn.16xlarge'|'cc1.4xlarge'|'cc2.8xlarge'|'g2.2xlarge'|'g2.8xlarge'|'g3.4xlarge'|'g3.8xlarge'|'g3.16xlarge'|'g3s.xlarge'|'g4ad.4xlarge'|'g4ad.8xlarge'|'g4ad.16xlarge'|'g4dn.xlarge'|'g4dn.2xlarge'|'g4dn.4xlarge'|'g4dn.8xlarge'|'g4dn.12xlarge'|'g4dn.16xlarge'|'g4dn.metal'|'cg1.4xlarge'|'p2.xlarge'|'p2.8xlarge'|'p2.16xlarge'|'p3.2xlarge'|'p3.8xlarge'|'p3.16xlarge'|'p3dn.24xlarge'|'p4d.24xlarge'|'d2.xlarge'|'d2.2xlarge'|'d2.4xlarge'|'d2.8xlarge'|'d3.xlarge'|'d3.2xlarge'|'d3.4xlarge'|'d3.8xlarge'|'d3en.xlarge'|'d3en.2xlarge'|'d3en.4xlarge'|'d3en.6xlarge'|'d3en.8xlarge'|'d3en.12xlarge'|'f1.2xlarge'|'f1.4xlarge'|'f1.16xlarge'|'m5.large'|'m5.xlarge'|'m5.2xlarge'|'m5.4xlarge'|'m5.8xlarge'|'m5.12xlarge'|'m5.16xlarge'|'m5.24xlarge'|'m5.metal'|'m5a.large'|'m5a.xlarge'|'m5a.2xlarge'|'m5a.4xlarge'|'m5a.8xlarge'|'m5a.12xlarge'|'m5a.16xlarge'|'m5a.24xlarge'|'m5d.large'|'m5d.xlarge'|'m5d.2xlarge'|'m5d.4xlarge'|'m5d.8xlarge'|'m5d.12xlarge'|'m5d.16xlarge'|'m5d.24xlarge'|'m5d.metal'|'m5ad.large'|'m5ad.xlarge'|'m5ad.2xlarge'|'m5ad.4xlarge'|'m5ad.8xlarge'|'m5ad.12xlarge'|'m5ad.16xlarge'|'m5ad.24xlarge'|'m5zn.large'|'m5zn.xlarge'|'m5zn.2xlarge'|'m5zn.3xlarge'|'m5zn.6xlarge'|'m5zn.12xlarge'|'m5zn.metal'|'h1.2xlarge'|'h1.4xlarge'|'h1.8xlarge'|'h1.16xlarge'|'z1d.large'|'z1d.xlarge'|'z1d.2xlarge'|'z1d.3xlarge'|'z1d.6xlarge'|'z1d.12xlarge'|'z1d.metal'|'u-6tb1.metal'|'u-9tb1.metal'|'u-12tb1.metal'|'u-18tb1.metal'|'u-24tb1.metal'|'a1.medium'|'a1.large'|'a1.xlarge'|'a1.2xlarge'|'a1.4xlarge'|'a1.metal'|'m5dn.large'|'m5dn.xlarge'|'m5dn.2xlarge'|'m5dn.4xlarge'|'m5dn.8xlarge'|'m5dn.12xlarge'|'m5dn.16xlarge'|'m5dn.24xlarge'|'m5n.large'|'m5n.xlarge'|'m5n.2xlarge'|'m5n.4xlarge'|'m5n.8xlarge'|'m5n.12xlarge'|'m5n.16xlarge'|'m5n.24xlarge'|'r5dn.large'|'r5dn.xlarge'|'r5dn.2xlarge'|'r5dn.4xlarge'|'r5dn.8xlarge'|'r5dn.12xlarge'|'r5dn.16xlarge'|'r5dn.24xlarge'|'r5n.large'|'r5n.xlarge'|'r5n.2xlarge'|'r5n.4xlarge'|'r5n.8xlarge'|'r5n.12xlarge'|'r5n.16xlarge'|'r5n.24xlarge'|'inf1.xlarge'|'inf1.2xlarge'|'inf1.6xlarge'|'inf1.24xlarge'|'m6g.metal'|'m6g.medium'|'m6g.large'|'m6g.xlarge'|'m6g.2xlarge'|'m6g.4xlarge'|'m6g.8xlarge'|'m6g.12xlarge'|'m6g.16xlarge'|'m6gd.metal'|'m6gd.medium'|'m6gd.large'|'m6gd.xlarge'|'m6gd.2xlarge'|'m6gd.4xlarge'|'m6gd.8xlarge'|'m6gd.12xlarge'|'m6gd.16xlarge'|'mac1.metal',
    MaxDuration=123,
    MaxInstanceCount=123,
    MinDuration=123,
    OfferingClass='standard'|'convertible',
    ProductDescription='Linux/UNIX'|'Linux/UNIX (Amazon VPC)'|'Windows'|'Windows (Amazon VPC)',
    ReservedInstancesOfferingIds=[
        'string',
    ],
    DryRun=True|False,
    InstanceTenancy='default'|'dedicated'|'host',
    OfferingType='Heavy Utilization'|'Medium Utilization'|'Light Utilization'|'No Upfront'|'Partial Upfront'|'All Upfront',
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **AvailabilityZone** (_string_) -- The Availability Zone in which the Reserved Instance can be used.
- **Filters** (_list_) --

    One or more filters.

    - availability-zone - The Availability Zone where the Reserved Instance can be used.
    - duration - The duration of the Reserved Instance (for example, one year or three years), in seconds (31536000 | 94608000 ).
    - fixed-price - The purchase price of the Reserved Instance (for example, 9800.0).
    - instance-type - The instance type that is covered by the reservation.
    - marketplace - Set to true to show only Reserved Instance Marketplace offerings. When this filter is not used, which is the default behavior, all offerings from both AWS and the Reserved Instance Marketplace are listed.
    - product-description - The Reserved Instance product platform description. Instances that include (Amazon VPC) in the product platform description will only be displayed to EC2-Classic account holders and are for use with Amazon VPC. (Linux/UNIX | Linux/UNIX (Amazon VPC) | SUSE Linux | SUSE Linux (Amazon VPC) | Red Hat Enterprise Linux | Red Hat Enterprise Linux (Amazon VPC) | Windows | Windows (Amazon VPC) | Windows with SQL Server Standard | Windows with SQL Server Standard (Amazon VPC) | Windows with SQL Server Web | Windows with SQL Server Web (Amazon VPC) | Windows with SQL Server Enterprise | Windows with SQL Server Enterprise (Amazon VPC) )
    - reserved-instances-offering-id - The Reserved Instances offering ID.
    - scope - The scope of the Reserved Instance (Availability Zone or Region ).
    - usage-price - The usage price of the Reserved Instance, per hour (for example, 0.84).

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **IncludeMarketplace** (_boolean_) -- Include Reserved Instance Marketplace offerings in the response.
- **InstanceType** (_string_) -- The instance type that the reservation will cover (for example, m1.small ). For more information, see [Instance Types](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/instance-types.html) in the _Amazon Elastic Compute Cloud User Guide_ .
- **MaxDuration** (_integer_) --

    The maximum duration (in seconds) to filter when searching for offerings.

    Default: 94608000 (3 years)

- **MaxInstanceCount** (_integer_) --

    The maximum number of instances to filter when searching for offerings.

    Default: 20

- **MinDuration** (_integer_) --

    The minimum duration (in seconds) to filter when searching for offerings.

    Default: 2592000 (1 month)

- **OfferingClass** (_string_) -- The offering class of the Reserved Instance. Can be standard or convertible .
- **ProductDescription** (_string_) -- The Reserved Instance product platform description. Instances that include (Amazon VPC) in the description are for use with Amazon VPC.
- **ReservedInstancesOfferingIds** (_list_) --

    One or more Reserved Instances offering IDs.

    - _(string) --_
- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **InstanceTenancy** (_string_) --

    The tenancy of the instances covered by the reservation. A Reserved Instance with a tenancy of dedicated is applied to instances that run in a VPC on single-tenant hardware (i.e., Dedicated Instances).

    > **Important:** The host value cannot be used with this parameter. Use the default or dedicated values only.

    Default: default

- **OfferingType** (_string_) -- The Reserved Instance offering type. If you are using tools that predate the 2011-11-01 API version, you only have access to the Medium Utilization Reserved Instance offering type.
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'ReservedInstancesOfferings': [
        {
            'AvailabilityZone': 'string',
            'Duration': 123,
            'FixedPrice': ...,
            'InstanceType': 't1.micro'|'t2.nano'|'t2.micro'|'t2.small'|'t2.medium'|'t2.large'|'t2.xlarge'|'t2.2xlarge'|'t3.nano'|'t3.micro'|'t3.small'|'t3.medium'|'t3.large'|'t3.xlarge'|'t3.2xlarge'|'t3a.nano'|'t3a.micro'|'t3a.small'|'t3a.medium'|'t3a.large'|'t3a.xlarge'|'t3a.2xlarge'|'t4g.nano'|'t4g.micro'|'t4g.small'|'t4g.medium'|'t4g.large'|'t4g.xlarge'|'t4g.2xlarge'|'m1.small'|'m1.medium'|'m1.large'|'m1.xlarge'|'m3.medium'|'m3.large'|'m3.xlarge'|'m3.2xlarge'|'m4.large'|'m4.xlarge'|'m4.2xlarge'|'m4.4xlarge'|'m4.10xlarge'|'m4.16xlarge'|'m2.xlarge'|'m2.2xlarge'|'m2.4xlarge'|'cr1.8xlarge'|'r3.large'|'r3.xlarge'|'r3.2xlarge'|'r3.4xlarge'|'r3.8xlarge'|'r4.large'|'r4.xlarge'|'r4.2xlarge'|'r4.4xlarge'|'r4.8xlarge'|'r4.16xlarge'|'r5.large'|'r5.xlarge'|'r5.2xlarge'|'r5.4xlarge'|'r5.8xlarge'|'r5.12xlarge'|'r5.16xlarge'|'r5.24xlarge'|'r5.metal'|'r5a.large'|'r5a.xlarge'|'r5a.2xlarge'|'r5a.4xlarge'|'r5a.8xlarge'|'r5a.12xlarge'|'r5a.16xlarge'|'r5a.24xlarge'|'r5b.large'|'r5b.xlarge'|'r5b.2xlarge'|'r5b.4xlarge'|'r5b.8xlarge'|'r5b.12xlarge'|'r5b.16xlarge'|'r5b.24xlarge'|'r5b.metal'|'r5d.large'|'r5d.xlarge'|'r5d.2xlarge'|'r5d.4xlarge'|'r5d.8xlarge'|'r5d.12xlarge'|'r5d.16xlarge'|'r5d.24xlarge'|'r5d.metal'|'r5ad.large'|'r5ad.xlarge'|'r5ad.2xlarge'|'r5ad.4xlarge'|'r5ad.8xlarge'|'r5ad.12xlarge'|'r5ad.16xlarge'|'r5ad.24xlarge'|'r6g.metal'|'r6g.medium'|'r6g.large'|'r6g.xlarge'|'r6g.2xlarge'|'r6g.4xlarge'|'r6g.8xlarge'|'r6g.12xlarge'|'r6g.16xlarge'|'r6gd.metal'|'r6gd.medium'|'r6gd.large'|'r6gd.xlarge'|'r6gd.2xlarge'|'r6gd.4xlarge'|'r6gd.8xlarge'|'r6gd.12xlarge'|'r6gd.16xlarge'|'x1.16xlarge'|'x1.32xlarge'|'x1e.xlarge'|'x1e.2xlarge'|'x1e.4xlarge'|'x1e.8xlarge'|'x1e.16xlarge'|'x1e.32xlarge'|'i2.xlarge'|'i2.2xlarge'|'i2.4xlarge'|'i2.8xlarge'|'i3.large'|'i3.xlarge'|'i3.2xlarge'|'i3.4xlarge'|'i3.8xlarge'|'i3.16xlarge'|'i3.metal'|'i3en.large'|'i3en.xlarge'|'i3en.2xlarge'|'i3en.3xlarge'|'i3en.6xlarge'|'i3en.12xlarge'|'i3en.24xlarge'|'i3en.metal'|'hi1.4xlarge'|'hs1.8xlarge'|'c1.medium'|'c1.xlarge'|'c3.large'|'c3.xlarge'|'c3.2xlarge'|'c3.4xlarge'|'c3.8xlarge'|'c4.large'|'c4.xlarge'|'c4.2xlarge'|'c4.4xlarge'|'c4.8xlarge'|'c5.large'|'c5.xlarge'|'c5.2xlarge'|'c5.4xlarge'|'c5.9xlarge'|'c5.12xlarge'|'c5.18xlarge'|'c5.24xlarge'|'c5.metal'|'c5a.large'|'c5a.xlarge'|'c5a.2xlarge'|'c5a.4xlarge'|'c5a.8xlarge'|'c5a.12xlarge'|'c5a.16xlarge'|'c5a.24xlarge'|'c5ad.large'|'c5ad.xlarge'|'c5ad.2xlarge'|'c5ad.4xlarge'|'c5ad.8xlarge'|'c5ad.12xlarge'|'c5ad.16xlarge'|'c5ad.24xlarge'|'c5d.large'|'c5d.xlarge'|'c5d.2xlarge'|'c5d.4xlarge'|'c5d.9xlarge'|'c5d.12xlarge'|'c5d.18xlarge'|'c5d.24xlarge'|'c5d.metal'|'c5n.large'|'c5n.xlarge'|'c5n.2xlarge'|'c5n.4xlarge'|'c5n.9xlarge'|'c5n.18xlarge'|'c5n.metal'|'c6g.metal'|'c6g.medium'|'c6g.large'|'c6g.xlarge'|'c6g.2xlarge'|'c6g.4xlarge'|'c6g.8xlarge'|'c6g.12xlarge'|'c6g.16xlarge'|'c6gd.metal'|'c6gd.medium'|'c6gd.large'|'c6gd.xlarge'|'c6gd.2xlarge'|'c6gd.4xlarge'|'c6gd.8xlarge'|'c6gd.12xlarge'|'c6gd.16xlarge'|'c6gn.medium'|'c6gn.large'|'c6gn.xlarge'|'c6gn.2xlarge'|'c6gn.4xlarge'|'c6gn.8xlarge'|'c6gn.12xlarge'|'c6gn.16xlarge'|'cc1.4xlarge'|'cc2.8xlarge'|'g2.2xlarge'|'g2.8xlarge'|'g3.4xlarge'|'g3.8xlarge'|'g3.16xlarge'|'g3s.xlarge'|'g4ad.4xlarge'|'g4ad.8xlarge'|'g4ad.16xlarge'|'g4dn.xlarge'|'g4dn.2xlarge'|'g4dn.4xlarge'|'g4dn.8xlarge'|'g4dn.12xlarge'|'g4dn.16xlarge'|'g4dn.metal'|'cg1.4xlarge'|'p2.xlarge'|'p2.8xlarge'|'p2.16xlarge'|'p3.2xlarge'|'p3.8xlarge'|'p3.16xlarge'|'p3dn.24xlarge'|'p4d.24xlarge'|'d2.xlarge'|'d2.2xlarge'|'d2.4xlarge'|'d2.8xlarge'|'d3.xlarge'|'d3.2xlarge'|'d3.4xlarge'|'d3.8xlarge'|'d3en.xlarge'|'d3en.2xlarge'|'d3en.4xlarge'|'d3en.6xlarge'|'d3en.8xlarge'|'d3en.12xlarge'|'f1.2xlarge'|'f1.4xlarge'|'f1.16xlarge'|'m5.large'|'m5.xlarge'|'m5.2xlarge'|'m5.4xlarge'|'m5.8xlarge'|'m5.12xlarge'|'m5.16xlarge'|'m5.24xlarge'|'m5.metal'|'m5a.large'|'m5a.xlarge'|'m5a.2xlarge'|'m5a.4xlarge'|'m5a.8xlarge'|'m5a.12xlarge'|'m5a.16xlarge'|'m5a.24xlarge'|'m5d.large'|'m5d.xlarge'|'m5d.2xlarge'|'m5d.4xlarge'|'m5d.8xlarge'|'m5d.12xlarge'|'m5d.16xlarge'|'m5d.24xlarge'|'m5d.metal'|'m5ad.large'|'m5ad.xlarge'|'m5ad.2xlarge'|'m5ad.4xlarge'|'m5ad.8xlarge'|'m5ad.12xlarge'|'m5ad.16xlarge'|'m5ad.24xlarge'|'m5zn.large'|'m5zn.xlarge'|'m5zn.2xlarge'|'m5zn.3xlarge'|'m5zn.6xlarge'|'m5zn.12xlarge'|'m5zn.metal'|'h1.2xlarge'|'h1.4xlarge'|'h1.8xlarge'|'h1.16xlarge'|'z1d.large'|'z1d.xlarge'|'z1d.2xlarge'|'z1d.3xlarge'|'z1d.6xlarge'|'z1d.12xlarge'|'z1d.metal'|'u-6tb1.metal'|'u-9tb1.metal'|'u-12tb1.metal'|'u-18tb1.metal'|'u-24tb1.metal'|'a1.medium'|'a1.large'|'a1.xlarge'|'a1.2xlarge'|'a1.4xlarge'|'a1.metal'|'m5dn.large'|'m5dn.xlarge'|'m5dn.2xlarge'|'m5dn.4xlarge'|'m5dn.8xlarge'|'m5dn.12xlarge'|'m5dn.16xlarge'|'m5dn.24xlarge'|'m5n.large'|'m5n.xlarge'|'m5n.2xlarge'|'m5n.4xlarge'|'m5n.8xlarge'|'m5n.12xlarge'|'m5n.16xlarge'|'m5n.24xlarge'|'r5dn.large'|'r5dn.xlarge'|'r5dn.2xlarge'|'r5dn.4xlarge'|'r5dn.8xlarge'|'r5dn.12xlarge'|'r5dn.16xlarge'|'r5dn.24xlarge'|'r5n.large'|'r5n.xlarge'|'r5n.2xlarge'|'r5n.4xlarge'|'r5n.8xlarge'|'r5n.12xlarge'|'r5n.16xlarge'|'r5n.24xlarge'|'inf1.xlarge'|'inf1.2xlarge'|'inf1.6xlarge'|'inf1.24xlarge'|'m6g.metal'|'m6g.medium'|'m6g.large'|'m6g.xlarge'|'m6g.2xlarge'|'m6g.4xlarge'|'m6g.8xlarge'|'m6g.12xlarge'|'m6g.16xlarge'|'m6gd.metal'|'m6gd.medium'|'m6gd.large'|'m6gd.xlarge'|'m6gd.2xlarge'|'m6gd.4xlarge'|'m6gd.8xlarge'|'m6gd.12xlarge'|'m6gd.16xlarge'|'mac1.metal',
            'ProductDescription': 'Linux/UNIX'|'Linux/UNIX (Amazon VPC)'|'Windows'|'Windows (Amazon VPC)',
            'ReservedInstancesOfferingId': 'string',
            'UsagePrice': ...,
            'CurrencyCode': 'USD',
            'InstanceTenancy': 'default'|'dedicated'|'host',
            'Marketplace': True|False,
            'OfferingClass': 'standard'|'convertible',
            'OfferingType': 'Heavy Utilization'|'Medium Utilization'|'Light Utilization'|'No Upfront'|'Partial Upfront'|'All Upfront',
            'PricingDetails': [
                {
                    'Count': 123,
                    'Price': 123.0
                },
            ],
            'RecurringCharges': [
                {
                    'Amount': 123.0,
                    'Frequency': 'Hourly'
                },
            ],
            'Scope': 'Availability Zone'|'Region'
        },
    ],

}

**Response Structure**

- _(dict) --_

    Contains the output of DescribeReservedInstancesOfferings.

    - **ReservedInstancesOfferings** _(list) --_

        A list of Reserved Instances offerings.

        - _(dict) --_

            Describes a Reserved Instance offering.

            - **AvailabilityZone** _(string) --_

                The Availability Zone in which the Reserved Instance can be used.

            - **Duration** _(integer) --_

                The duration of the Reserved Instance, in seconds.

            - **FixedPrice** _(float) --_

                The purchase price of the Reserved Instance.

            - **InstanceType** _(string) --_

                The instance type on which the Reserved Instance can be used.

            - **ProductDescription** _(string) --_

                The Reserved Instance product platform description.

            - **ReservedInstancesOfferingId** _(string) --_

                The ID of the Reserved Instance offering. This is the offering ID used in GetReservedInstancesExchangeQuote to confirm that an exchange can be made.

            - **UsagePrice** _(float) --_

                The usage price of the Reserved Instance, per hour.

            - **CurrencyCode** _(string) --_

                The currency of the Reserved Instance offering you are purchasing. It's specified using ISO 4217 standard currency codes. At this time, the only supported currency is USD .

            - **InstanceTenancy** _(string) --_

                The tenancy of the instance.

            - **Marketplace** _(boolean) --_

                Indicates whether the offering is available through the Reserved Instance Marketplace (resale) or AWS. If it's a Reserved Instance Marketplace offering, this is true .

            - **OfferingClass** _(string) --_

                If convertible it can be exchanged for Reserved Instances of the same or higher monetary value, with different configurations. If standard , it is not possible to perform an exchange.

            - **OfferingType** _(string) --_

                The Reserved Instance offering type.

            - **PricingDetails** _(list) --_

                The pricing details of the Reserved Instance offering.

                - _(dict) --_

                    Describes a Reserved Instance offering.

                    - **Count** _(integer) --_

                        The number of reservations available for the price.

                    - **Price** _(float) --_

                        The price per instance.

            - **RecurringCharges** _(list) --_

                The recurring charge tag assigned to the resource.

                - _(dict) --_

                    Describes a recurring charge.

                    - **Amount** _(float) --_

                        The amount of the recurring charge.

                    - **Frequency** _(string) --_

                        The frequency of the recurring charge.

            - **Scope** _(string) --_

                Whether the Reserved Instance is applied to instances in a Region or an Availability Zone.


_class_ EC2.Paginator.DescribeRouteTables

paginator = client.get_paginator('describe_route_tables')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_route_tables()](#EC2.Client.describe_route_tables "EC2.Client.describe_route_tables").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeRouteTables)

**Request Syntax**

response_iterator = paginator.paginate(
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    DryRun=True|False,
    RouteTableIds=[
        'string',
    ],
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **Filters** (_list_) --

    One or more filters.

    - association.route-table-association-id - The ID of an association ID for the route table.
    - association.route-table-id - The ID of the route table involved in the association.
    - association.subnet-id - The ID of the subnet involved in the association.
    - association.main - Indicates whether the route table is the main route table for the VPC (true | false ). Route tables that do not have an association ID are not returned in the response.
    - owner-id - The ID of the AWS account that owns the route table.
    - route-table-id - The ID of the route table.
    - route.destination-cidr-block - The IPv4 CIDR range specified in a route in the table.
    - route.destination-ipv6-cidr-block - The IPv6 CIDR range specified in a route in the route table.
    - route.destination-prefix-list-id - The ID (prefix) of the AWS service specified in a route in the table.
    - route.egress-only-internet-gateway-id - The ID of an egress-only Internet gateway specified in a route in the route table.
    - route.gateway-id - The ID of a gateway specified in a route in the table.
    - route.instance-id - The ID of an instance specified in a route in the table.
    - route.nat-gateway-id - The ID of a NAT gateway.
    - route.transit-gateway-id - The ID of a transit gateway.
    - route.origin - Describes how the route was created. CreateRouteTable indicates that the route was automatically created when the route table was created; CreateRoute indicates that the route was manually added to the route table; EnableVgwRoutePropagation indicates that the route was propagated by route propagation.
    - route.state - The state of a route in the route table (active | blackhole ). The blackhole state indicates that the route's target isn't available (for example, the specified gateway isn't attached to the VPC, the specified NAT instance has been terminated, and so on).
    - route.vpc-peering-connection-id - The ID of a VPC peering connection specified in a route in the table.
    - tag :<key> - The key/value combination of a tag assigned to the resource. Use the tag key in the filter name and the tag value as the filter value. For example, to find all resources that have a tag with the key Owner and the value TeamA , specify tag:Owner for the filter name and TeamA for the filter value.
    - tag-key - The key of a tag assigned to the resource. Use this filter to find all resources assigned a tag with a specific key, regardless of the tag value.
    - vpc-id - The ID of the VPC for the route table.

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **RouteTableIds** (_list_) --

    One or more route table IDs.

    Default: Describes all your route tables.

    - _(string) --_
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'RouteTables': [
        {
            'Associations': [
                {
                    'Main': True|False,
                    'RouteTableAssociationId': 'string',
                    'RouteTableId': 'string',
                    'SubnetId': 'string',
                    'GatewayId': 'string',
                    'AssociationState': {
                        'State': 'associating'|'associated'|'disassociating'|'disassociated'|'failed',
                        'StatusMessage': 'string'
                    }
                },
            ],
            'PropagatingVgws': [
                {
                    'GatewayId': 'string'
                },
            ],
            'RouteTableId': 'string',
            'Routes': [
                {
                    'DestinationCidrBlock': 'string',
                    'DestinationIpv6CidrBlock': 'string',
                    'DestinationPrefixListId': 'string',
                    'EgressOnlyInternetGatewayId': 'string',
                    'GatewayId': 'string',
                    'InstanceId': 'string',
                    'InstanceOwnerId': 'string',
                    'NatGatewayId': 'string',
                    'TransitGatewayId': 'string',
                    'LocalGatewayId': 'string',
                    'CarrierGatewayId': 'string',
                    'NetworkInterfaceId': 'string',
                    'Origin': 'CreateRouteTable'|'CreateRoute'|'EnableVgwRoutePropagation',
                    'State': 'active'|'blackhole',
                    'VpcPeeringConnectionId': 'string'
                },
            ],
            'Tags': [
                {
                    'Key': 'string',
                    'Value': 'string'
                },
            ],
            'VpcId': 'string',
            'OwnerId': 'string'
        },
    ],

}

**Response Structure**

- _(dict) --_

    Contains the output of DescribeRouteTables.

    - **RouteTables** _(list) --_

        Information about one or more route tables.

        - _(dict) --_

            Describes a route table.

            - **Associations** _(list) --_

                The associations between the route table and one or more subnets or a gateway.

                - _(dict) --_

                    Describes an association between a route table and a subnet or gateway.

                    - **Main** _(boolean) --_

                        Indicates whether this is the main route table.

                    - **RouteTableAssociationId** _(string) --_

                        The ID of the association.

                    - **RouteTableId** _(string) --_

                        The ID of the route table.

                    - **SubnetId** _(string) --_

                        The ID of the subnet. A subnet ID is not returned for an implicit association.

                    - **GatewayId** _(string) --_

                        The ID of the internet gateway or virtual private gateway.

                    - **AssociationState** _(dict) --_

                        The state of the association.

                        - **State** _(string) --_

                            The state of the association.

                        - **StatusMessage** _(string) --_

                            The status message, if applicable.

            - **PropagatingVgws** _(list) --_

                Any virtual private gateway (VGW) propagating routes.

                - _(dict) --_

                    Describes a virtual private gateway propagating route.

                    - **GatewayId** _(string) --_

                        The ID of the virtual private gateway.

            - **RouteTableId** _(string) --_

                The ID of the route table.

            - **Routes** _(list) --_

                The routes in the route table.

                - _(dict) --_

                    Describes a route in a route table.

                    - **DestinationCidrBlock** _(string) --_

                        The IPv4 CIDR block used for the destination match.

                    - **DestinationIpv6CidrBlock** _(string) --_

                        The IPv6 CIDR block used for the destination match.

                    - **DestinationPrefixListId** _(string) --_

                        The prefix of the AWS service.

                    - **EgressOnlyInternetGatewayId** _(string) --_

                        The ID of the egress-only internet gateway.

                    - **GatewayId** _(string) --_

                        The ID of a gateway attached to your VPC.

                    - **InstanceId** _(string) --_

                        The ID of a NAT instance in your VPC.

                    - **InstanceOwnerId** _(string) --_

                        The AWS account ID of the owner of the instance.

                    - **NatGatewayId** _(string) --_

                        The ID of a NAT gateway.

                    - **TransitGatewayId** _(string) --_

                        The ID of a transit gateway.

                    - **LocalGatewayId** _(string) --_

                        The ID of the local gateway.

                    - **CarrierGatewayId** _(string) --_

                        The ID of the carrier gateway.

                    - **NetworkInterfaceId** _(string) --_

                        The ID of the network interface.

                    - **Origin** _(string) --_

                        Describes how the route was created.

                        - CreateRouteTable - The route was automatically created when the route table was created.
                        - CreateRoute - The route was manually added to the route table.
                        - EnableVgwRoutePropagation - The route was propagated by route propagation.
                    - **State** _(string) --_

                        The state of the route. The blackhole state indicates that the route's target isn't available (for example, the specified gateway isn't attached to the VPC, or the specified NAT instance has been terminated).

                    - **VpcPeeringConnectionId** _(string) --_

                        The ID of a VPC peering connection.

            - **Tags** _(list) --_

                Any tags assigned to the route table.

                - _(dict) --_

                    Describes a tag.

                    - **Key** _(string) --_

                        The key of the tag.

                        Constraints: Tag keys are case-sensitive and accept a maximum of 127 Unicode characters. May not begin with aws: .

                    - **Value** _(string) --_

                        The value of the tag.

                        Constraints: Tag values are case-sensitive and accept a maximum of 255 Unicode characters.

            - **VpcId** _(string) --_

                The ID of the VPC.

            - **OwnerId** _(string) --_

                The ID of the AWS account that owns the route table.


_class_ EC2.Paginator.DescribeScheduledInstanceAvailability

paginator = client.get_paginator('describe_scheduled_instance_availability')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_scheduled_instance_availability()](#EC2.Client.describe_scheduled_instance_availability "EC2.Client.describe_scheduled_instance_availability").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeScheduledInstanceAvailability)

**Request Syntax**

response_iterator = paginator.paginate(
    DryRun=True|False,
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    FirstSlotStartTimeRange={
        'EarliestTime': datetime(2015, 1, 1),
        'LatestTime': datetime(2015, 1, 1)
    },
    MaxSlotDurationInHours=123,
    MinSlotDurationInHours=123,
    Recurrence={
        'Frequency': 'string',
        'Interval': 123,
        'OccurrenceDays': [
            123,
        ],
        'OccurrenceRelativeToEnd': True|False,
        'OccurrenceUnit': 'string'
    },
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **Filters** (_list_) --

    The filters.

    - availability-zone - The Availability Zone (for example, us-west-2a ).
    - instance-type - The instance type (for example, c4.large ).
    - network-platform - The network platform (EC2-Classic or EC2-VPC ).
    - platform - The platform (Linux/UNIX or Windows ).

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **FirstSlotStartTimeRange** (_dict_) --

    **[REQUIRED]**

    The time period for the first schedule to start.

    - **EarliestTime** _(datetime) --_ **[REQUIRED]**

        The earliest date and time, in UTC, for the Scheduled Instance to start.

    - **LatestTime** _(datetime) --_ **[REQUIRED]**

        The latest date and time, in UTC, for the Scheduled Instance to start. This value must be later than or equal to the earliest date and at most three months in the future.

- **MaxSlotDurationInHours** (_integer_) -- The maximum available duration, in hours. This value must be greater than MinSlotDurationInHours and less than 1,720.
- **MinSlotDurationInHours** (_integer_) -- The minimum available duration, in hours. The minimum required duration is 1,200 hours per year. For example, the minimum daily schedule is 4 hours, the minimum weekly schedule is 24 hours, and the minimum monthly schedule is 100 hours.
- **Recurrence** (_dict_) --

    **[REQUIRED]**

    The schedule recurrence.

    - **Frequency** _(string) --_

        The frequency (Daily , Weekly , or Monthly ).

    - **Interval** _(integer) --_

        The interval quantity. The interval unit depends on the value of Frequency . For example, every 2 weeks or every 2 months.

    - **OccurrenceDays** _(list) --_

        The days. For a monthly schedule, this is one or more days of the month (1-31). For a weekly schedule, this is one or more days of the week (1-7, where 1 is Sunday). You can't specify this value with a daily schedule. If the occurrence is relative to the end of the month, you can specify only a single day.

        - _(integer) --_
    - **OccurrenceRelativeToEnd** _(boolean) --_

        Indicates whether the occurrence is relative to the end of the specified week or month. You can't specify this value with a daily schedule.

    - **OccurrenceUnit** _(string) --_

        The unit for OccurrenceDays (DayOfWeek or DayOfMonth ). This value is required for a monthly schedule. You can't specify DayOfWeek with a weekly schedule. You can't specify this value with a daily schedule.

- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'ScheduledInstanceAvailabilitySet': [
        {
            'AvailabilityZone': 'string',
            'AvailableInstanceCount': 123,
            'FirstSlotStartTime': datetime(2015, 1, 1),
            'HourlyPrice': 'string',
            'InstanceType': 'string',
            'MaxTermDurationInDays': 123,
            'MinTermDurationInDays': 123,
            'NetworkPlatform': 'string',
            'Platform': 'string',
            'PurchaseToken': 'string',
            'Recurrence': {
                'Frequency': 'string',
                'Interval': 123,
                'OccurrenceDaySet': [
                    123,
                ],
                'OccurrenceRelativeToEnd': True|False,
                'OccurrenceUnit': 'string'
            },
            'SlotDurationInHours': 123,
            'TotalScheduledInstanceHours': 123
        },
    ]
}

**Response Structure**

- _(dict) --_

    Contains the output of DescribeScheduledInstanceAvailability.

    - **ScheduledInstanceAvailabilitySet** _(list) --_

        Information about the available Scheduled Instances.

        - _(dict) --_

            Describes a schedule that is available for your Scheduled Instances.

            - **AvailabilityZone** _(string) --_

                The Availability Zone.

            - **AvailableInstanceCount** _(integer) --_

                The number of available instances.

            - **FirstSlotStartTime** _(datetime) --_

                The time period for the first schedule to start.

            - **HourlyPrice** _(string) --_

                The hourly price for a single instance.

            - **InstanceType** _(string) --_

                The instance type. You can specify one of the C3, C4, M4, or R3 instance types.

            - **MaxTermDurationInDays** _(integer) --_

                The maximum term. The only possible value is 365 days.

            - **MinTermDurationInDays** _(integer) --_

                The minimum term. The only possible value is 365 days.

            - **NetworkPlatform** _(string) --_

                The network platform (EC2-Classic or EC2-VPC ).

            - **Platform** _(string) --_

                The platform (Linux/UNIX or Windows ).

            - **PurchaseToken** _(string) --_

                The purchase token. This token expires in two hours.

            - **Recurrence** _(dict) --_

                The schedule recurrence.

                - **Frequency** _(string) --_

                    The frequency (Daily , Weekly , or Monthly ).

                - **Interval** _(integer) --_

                    The interval quantity. The interval unit depends on the value of frequency . For example, every 2 weeks or every 2 months.

                - **OccurrenceDaySet** _(list) --_

                    The days. For a monthly schedule, this is one or more days of the month (1-31). For a weekly schedule, this is one or more days of the week (1-7, where 1 is Sunday).

                    - _(integer) --_
                - **OccurrenceRelativeToEnd** _(boolean) --_

                    Indicates whether the occurrence is relative to the end of the specified week or month.

                - **OccurrenceUnit** _(string) --_

                    The unit for occurrenceDaySet (DayOfWeek or DayOfMonth ).

            - **SlotDurationInHours** _(integer) --_

                The number of hours in the schedule.

            - **TotalScheduledInstanceHours** _(integer) --_

                The total number of hours for a single instance for the entire term.


_class_ EC2.Paginator.DescribeScheduledInstances

paginator = client.get_paginator('describe_scheduled_instances')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_scheduled_instances()](#EC2.Client.describe_scheduled_instances "EC2.Client.describe_scheduled_instances").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeScheduledInstances)

**Request Syntax**

response_iterator = paginator.paginate(
    DryRun=True|False,
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    ScheduledInstanceIds=[
        'string',
    ],
    SlotStartTimeRange={
        'EarliestTime': datetime(2015, 1, 1),
        'LatestTime': datetime(2015, 1, 1)
    },
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **Filters** (_list_) --

    The filters.

    - availability-zone - The Availability Zone (for example, us-west-2a ).
    - instance-type - The instance type (for example, c4.large ).
    - network-platform - The network platform (EC2-Classic or EC2-VPC ).
    - platform - The platform (Linux/UNIX or Windows ).

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **ScheduledInstanceIds** (_list_) --

    The Scheduled Instance IDs.

    - _(string) --_
- **SlotStartTimeRange** (_dict_) --

    The time period for the first schedule to start.

    - **EarliestTime** _(datetime) --_

        The earliest date and time, in UTC, for the Scheduled Instance to start.

    - **LatestTime** _(datetime) --_

        The latest date and time, in UTC, for the Scheduled Instance to start.

- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'ScheduledInstanceSet': [
        {
            'AvailabilityZone': 'string',
            'CreateDate': datetime(2015, 1, 1),
            'HourlyPrice': 'string',
            'InstanceCount': 123,
            'InstanceType': 'string',
            'NetworkPlatform': 'string',
            'NextSlotStartTime': datetime(2015, 1, 1),
            'Platform': 'string',
            'PreviousSlotEndTime': datetime(2015, 1, 1),
            'Recurrence': {
                'Frequency': 'string',
                'Interval': 123,
                'OccurrenceDaySet': [
                    123,
                ],
                'OccurrenceRelativeToEnd': True|False,
                'OccurrenceUnit': 'string'
            },
            'ScheduledInstanceId': 'string',
            'SlotDurationInHours': 123,
            'TermEndDate': datetime(2015, 1, 1),
            'TermStartDate': datetime(2015, 1, 1),
            'TotalScheduledInstanceHours': 123
        },
    ]
}

**Response Structure**

- _(dict) --_

    Contains the output of DescribeScheduledInstances.

    - **ScheduledInstanceSet** _(list) --_

        Information about the Scheduled Instances.

        - _(dict) --_

            Describes a Scheduled Instance.

            - **AvailabilityZone** _(string) --_

                The Availability Zone.

            - **CreateDate** _(datetime) --_

                The date when the Scheduled Instance was purchased.

            - **HourlyPrice** _(string) --_

                The hourly price for a single instance.

            - **InstanceCount** _(integer) --_

                The number of instances.

            - **InstanceType** _(string) --_

                The instance type.

            - **NetworkPlatform** _(string) --_

                The network platform (EC2-Classic or EC2-VPC ).

            - **NextSlotStartTime** _(datetime) --_

                The time for the next schedule to start.

            - **Platform** _(string) --_

                The platform (Linux/UNIX or Windows ).

            - **PreviousSlotEndTime** _(datetime) --_

                The time that the previous schedule ended or will end.

            - **Recurrence** _(dict) --_

                The schedule recurrence.

                - **Frequency** _(string) --_

                    The frequency (Daily , Weekly , or Monthly ).

                - **Interval** _(integer) --_

                    The interval quantity. The interval unit depends on the value of frequency . For example, every 2 weeks or every 2 months.

                - **OccurrenceDaySet** _(list) --_

                    The days. For a monthly schedule, this is one or more days of the month (1-31). For a weekly schedule, this is one or more days of the week (1-7, where 1 is Sunday).

                    - _(integer) --_
                - **OccurrenceRelativeToEnd** _(boolean) --_

                    Indicates whether the occurrence is relative to the end of the specified week or month.

                - **OccurrenceUnit** _(string) --_

                    The unit for occurrenceDaySet (DayOfWeek or DayOfMonth ).

            - **ScheduledInstanceId** _(string) --_

                The Scheduled Instance ID.

            - **SlotDurationInHours** _(integer) --_

                The number of hours in the schedule.

            - **TermEndDate** _(datetime) --_

                The end date for the Scheduled Instance.

            - **TermStartDate** _(datetime) --_

                The start date for the Scheduled Instance.

            - **TotalScheduledInstanceHours** _(integer) --_

                The total number of hours for a single instance for the entire term.


_class_ EC2.Paginator.DescribeSecurityGroups

paginator = client.get_paginator('describe_security_groups')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_security_groups()](#EC2.Client.describe_security_groups "EC2.Client.describe_security_groups").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeSecurityGroups)

**Request Syntax**

response_iterator = paginator.paginate(
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    GroupIds=[
        'string',
    ],
    GroupNames=[
        'string',
    ],
    DryRun=True|False,
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **Filters** (_list_) --

    The filters. If using multiple filters for rules, the results include security groups for which any combination of rules - not necessarily a single rule - match all filters.

    - description - The description of the security group.
    - egress.ip-permission.cidr - An IPv4 CIDR block for an outbound security group rule.
    - egress.ip-permission.from-port - For an outbound rule, the start of port range for the TCP and UDP protocols, or an ICMP type number.
    - egress.ip-permission.group-id - The ID of a security group that has been referenced in an outbound security group rule.
    - egress.ip-permission.group-name - The name of a security group that has been referenced in an outbound security group rule.
    - egress.ip-permission.ipv6-cidr - An IPv6 CIDR block for an outbound security group rule.
    - egress.ip-permission.prefix-list-id - The ID of a prefix list to which a security group rule allows outbound access.
    - egress.ip-permission.protocol - The IP protocol for an outbound security group rule (tcp | udp | icmp or a protocol number).
    - egress.ip-permission.to-port - For an outbound rule, the end of port range for the TCP and UDP protocols, or an ICMP code.
    - egress.ip-permission.user-id - The ID of an AWS account that has been referenced in an outbound security group rule.
    - group-id - The ID of the security group.
    - group-name - The name of the security group.
    - ip-permission.cidr - An IPv4 CIDR block for an inbound security group rule.
    - ip-permission.from-port - For an inbound rule, the start of port range for the TCP and UDP protocols, or an ICMP type number.
    - ip-permission.group-id - The ID of a security group that has been referenced in an inbound security group rule.
    - ip-permission.group-name - The name of a security group that has been referenced in an inbound security group rule.
    - ip-permission.ipv6-cidr - An IPv6 CIDR block for an inbound security group rule.
    - ip-permission.prefix-list-id - The ID of a prefix list from which a security group rule allows inbound access.
    - ip-permission.protocol - The IP protocol for an inbound security group rule (tcp | udp | icmp or a protocol number).
    - ip-permission.to-port - For an inbound rule, the end of port range for the TCP and UDP protocols, or an ICMP code.
    - ip-permission.user-id - The ID of an AWS account that has been referenced in an inbound security group rule.
    - owner-id - The AWS account ID of the owner of the security group.
    - tag :<key> - The key/value combination of a tag assigned to the resource. Use the tag key in the filter name and the tag value as the filter value. For example, to find all resources that have a tag with the key Owner and the value TeamA , specify tag:Owner for the filter name and TeamA for the filter value.
    - tag-key - The key of a tag assigned to the resource. Use this filter to find all resources assigned a tag with a specific key, regardless of the tag value.
    - vpc-id - The ID of the VPC specified when the security group was created.

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **GroupIds** (_list_) --

    The IDs of the security groups. Required for security groups in a nondefault VPC.

    Default: Describes all your security groups.

    - _(string) --_
- **GroupNames** (_list_) --

    [EC2-Classic and default VPC only] The names of the security groups. You can specify either the security group name or the security group ID. For security groups in a nondefault VPC, use the group-name filter to describe security groups by name.

    Default: Describes all your security groups.

    - _(string) --_
- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'SecurityGroups': [
        {
            'Description': 'string',
            'GroupName': 'string',
            'IpPermissions': [
                {
                    'FromPort': 123,
                    'IpProtocol': 'string',
                    'IpRanges': [
                        {
                            'CidrIp': 'string',
                            'Description': 'string'
                        },
                    ],
                    'Ipv6Ranges': [
                        {
                            'CidrIpv6': 'string',
                            'Description': 'string'
                        },
                    ],
                    'PrefixListIds': [
                        {
                            'Description': 'string',
                            'PrefixListId': 'string'
                        },
                    ],
                    'ToPort': 123,
                    'UserIdGroupPairs': [
                        {
                            'Description': 'string',
                            'GroupId': 'string',
                            'GroupName': 'string',
                            'PeeringStatus': 'string',
                            'UserId': 'string',
                            'VpcId': 'string',
                            'VpcPeeringConnectionId': 'string'
                        },
                    ]
                },
            ],
            'OwnerId': 'string',
            'GroupId': 'string',
            'IpPermissionsEgress': [
                {
                    'FromPort': 123,
                    'IpProtocol': 'string',
                    'IpRanges': [
                        {
                            'CidrIp': 'string',
                            'Description': 'string'
                        },
                    ],
                    'Ipv6Ranges': [
                        {
                            'CidrIpv6': 'string',
                            'Description': 'string'
                        },
                    ],
                    'PrefixListIds': [
                        {
                            'Description': 'string',
                            'PrefixListId': 'string'
                        },
                    ],
                    'ToPort': 123,
                    'UserIdGroupPairs': [
                        {
                            'Description': 'string',
                            'GroupId': 'string',
                            'GroupName': 'string',
                            'PeeringStatus': 'string',
                            'UserId': 'string',
                            'VpcId': 'string',
                            'VpcPeeringConnectionId': 'string'
                        },
                    ]
                },
            ],
            'Tags': [
                {
                    'Key': 'string',
                    'Value': 'string'
                },
            ],
            'VpcId': 'string'
        },
    ],

}

**Response Structure**

- _(dict) --_

    - **SecurityGroups** _(list) --_

        Information about the security groups.

        - _(dict) --_

            Describes a security group

            - **Description** _(string) --_

                A description of the security group.

            - **GroupName** _(string) --_

                The name of the security group.

            - **IpPermissions** _(list) --_

                The inbound rules associated with the security group.

                - _(dict) --_

                    Describes a set of permissions for a security group rule.

                    - **FromPort** _(integer) --_

                        The start of port range for the TCP and UDP protocols, or an ICMP/ICMPv6 type number. A value of -1 indicates all ICMP/ICMPv6 types. If you specify all ICMP/ICMPv6 types, you must specify all codes.

                    - **IpProtocol** _(string) --_

                        The IP protocol name (tcp , udp , icmp , icmpv6 ) or number (see [Protocol Numbers](https://www.iana.org/assignments/protocol-numbers/protocol-numbers.xhtml) ).

                        [VPC only] Use -1 to specify all protocols. When authorizing security group rules, specifying -1 or a protocol number other than tcp , udp , icmp , or icmpv6 allows traffic on all ports, regardless of any port range you specify. For tcp , udp , and icmp , you must specify a port range. For icmpv6 , the port range is optional; if you omit the port range, traffic for all types and codes is allowed.

                    - **IpRanges** _(list) --_

                        The IPv4 ranges.

                        - _(dict) --_

                            Describes an IPv4 range.

                            - **CidrIp** _(string) --_

                                The IPv4 CIDR range. You can either specify a CIDR range or a source security group, not both. To specify a single IPv4 address, use the /32 prefix length.

                            - **Description** _(string) --_

                                A description for the security group rule that references this IPv4 address range.

                                Constraints: Up to 255 characters in length. Allowed characters are a-z, A-Z, 0-9, spaces, and ._-:/()#,@[]+=&;{}!$*

                    - **Ipv6Ranges** _(list) --_

                        [VPC only] The IPv6 ranges.

                        - _(dict) --_

                            [EC2-VPC only] Describes an IPv6 range.

                            - **CidrIpv6** _(string) --_

                                The IPv6 CIDR range. You can either specify a CIDR range or a source security group, not both. To specify a single IPv6 address, use the /128 prefix length.

                            - **Description** _(string) --_

                                A description for the security group rule that references this IPv6 address range.

                                Constraints: Up to 255 characters in length. Allowed characters are a-z, A-Z, 0-9, spaces, and ._-:/()#,@[]+=&;{}!$*

                    - **PrefixListIds** _(list) --_

                        [VPC only] The prefix list IDs.

                        - _(dict) --_

                            Describes a prefix list ID.

                            - **Description** _(string) --_

                                A description for the security group rule that references this prefix list ID.

                                Constraints: Up to 255 characters in length. Allowed characters are a-z, A-Z, 0-9, spaces, and ._-:/()#,@[]+=;{}!$*

                            - **PrefixListId** _(string) --_

                                The ID of the prefix.

                    - **ToPort** _(integer) --_

                        The end of port range for the TCP and UDP protocols, or an ICMP/ICMPv6 code. A value of -1 indicates all ICMP/ICMPv6 codes. If you specify all ICMP/ICMPv6 types, you must specify all codes.

                    - **UserIdGroupPairs** _(list) --_

                        The security group and AWS account ID pairs.

                        - _(dict) --_

                            Describes a security group and AWS account ID pair.

                            - **Description** _(string) --_

                                A description for the security group rule that references this user ID group pair.

                                Constraints: Up to 255 characters in length. Allowed characters are a-z, A-Z, 0-9, spaces, and ._-:/()#,@[]+=;{}!$*

                            - **GroupId** _(string) --_

                                The ID of the security group.

                            - **GroupName** _(string) --_

                                The name of the security group. In a request, use this parameter for a security group in EC2-Classic or a default VPC only. For a security group in a nondefault VPC, use the security group ID.

                                For a referenced security group in another VPC, this value is not returned if the referenced security group is deleted.

                            - **PeeringStatus** _(string) --_

                                The status of a VPC peering connection, if applicable.

                            - **UserId** _(string) --_

                                The ID of an AWS account.

                                For a referenced security group in another VPC, the account ID of the referenced security group is returned in the response. If the referenced security group is deleted, this value is not returned.

                                [EC2-Classic] Required when adding or removing rules that reference a security group in another AWS account.

                            - **VpcId** _(string) --_

                                The ID of the VPC for the referenced security group, if applicable.

                            - **VpcPeeringConnectionId** _(string) --_

                                The ID of the VPC peering connection, if applicable.

            - **OwnerId** _(string) --_

                The AWS account ID of the owner of the security group.

            - **GroupId** _(string) --_

                The ID of the security group.

            - **IpPermissionsEgress** _(list) --_

                [VPC only] The outbound rules associated with the security group.

                - _(dict) --_

                    Describes a set of permissions for a security group rule.

                    - **FromPort** _(integer) --_

                        The start of port range for the TCP and UDP protocols, or an ICMP/ICMPv6 type number. A value of -1 indicates all ICMP/ICMPv6 types. If you specify all ICMP/ICMPv6 types, you must specify all codes.

                    - **IpProtocol** _(string) --_

                        The IP protocol name (tcp , udp , icmp , icmpv6 ) or number (see [Protocol Numbers](https://www.iana.org/assignments/protocol-numbers/protocol-numbers.xhtml) ).

                        [VPC only] Use -1 to specify all protocols. When authorizing security group rules, specifying -1 or a protocol number other than tcp , udp , icmp , or icmpv6 allows traffic on all ports, regardless of any port range you specify. For tcp , udp , and icmp , you must specify a port range. For icmpv6 , the port range is optional; if you omit the port range, traffic for all types and codes is allowed.

                    - **IpRanges** _(list) --_

                        The IPv4 ranges.

                        - _(dict) --_

                            Describes an IPv4 range.

                            - **CidrIp** _(string) --_

                                The IPv4 CIDR range. You can either specify a CIDR range or a source security group, not both. To specify a single IPv4 address, use the /32 prefix length.

                            - **Description** _(string) --_

                                A description for the security group rule that references this IPv4 address range.

                                Constraints: Up to 255 characters in length. Allowed characters are a-z, A-Z, 0-9, spaces, and ._-:/()#,@[]+=&;{}!$*

                    - **Ipv6Ranges** _(list) --_

                        [VPC only] The IPv6 ranges.

                        - _(dict) --_

                            [EC2-VPC only] Describes an IPv6 range.

                            - **CidrIpv6** _(string) --_

                                The IPv6 CIDR range. You can either specify a CIDR range or a source security group, not both. To specify a single IPv6 address, use the /128 prefix length.

                            - **Description** _(string) --_

                                A description for the security group rule that references this IPv6 address range.

                                Constraints: Up to 255 characters in length. Allowed characters are a-z, A-Z, 0-9, spaces, and ._-:/()#,@[]+=&;{}!$*

                    - **PrefixListIds** _(list) --_

                        [VPC only] The prefix list IDs.

                        - _(dict) --_

                            Describes a prefix list ID.

                            - **Description** _(string) --_

                                A description for the security group rule that references this prefix list ID.

                                Constraints: Up to 255 characters in length. Allowed characters are a-z, A-Z, 0-9, spaces, and ._-:/()#,@[]+=;{}!$*

                            - **PrefixListId** _(string) --_

                                The ID of the prefix.

                    - **ToPort** _(integer) --_

                        The end of port range for the TCP and UDP protocols, or an ICMP/ICMPv6 code. A value of -1 indicates all ICMP/ICMPv6 codes. If you specify all ICMP/ICMPv6 types, you must specify all codes.

                    - **UserIdGroupPairs** _(list) --_

                        The security group and AWS account ID pairs.

                        - _(dict) --_

                            Describes a security group and AWS account ID pair.

                            - **Description** _(string) --_

                                A description for the security group rule that references this user ID group pair.

                                Constraints: Up to 255 characters in length. Allowed characters are a-z, A-Z, 0-9, spaces, and ._-:/()#,@[]+=;{}!$*

                            - **GroupId** _(string) --_

                                The ID of the security group.

                            - **GroupName** _(string) --_

                                The name of the security group. In a request, use this parameter for a security group in EC2-Classic or a default VPC only. For a security group in a nondefault VPC, use the security group ID.

                                For a referenced security group in another VPC, this value is not returned if the referenced security group is deleted.

                            - **PeeringStatus** _(string) --_

                                The status of a VPC peering connection, if applicable.

                            - **UserId** _(string) --_

                                The ID of an AWS account.

                                For a referenced security group in another VPC, the account ID of the referenced security group is returned in the response. If the referenced security group is deleted, this value is not returned.

                                [EC2-Classic] Required when adding or removing rules that reference a security group in another AWS account.

                            - **VpcId** _(string) --_

                                The ID of the VPC for the referenced security group, if applicable.

                            - **VpcPeeringConnectionId** _(string) --_

                                The ID of the VPC peering connection, if applicable.

            - **Tags** _(list) --_

                Any tags assigned to the security group.

                - _(dict) --_

                    Describes a tag.

                    - **Key** _(string) --_

                        The key of the tag.

                        Constraints: Tag keys are case-sensitive and accept a maximum of 127 Unicode characters. May not begin with aws: .

                    - **Value** _(string) --_

                        The value of the tag.

                        Constraints: Tag values are case-sensitive and accept a maximum of 255 Unicode characters.

            - **VpcId** _(string) --_

                [VPC only] The ID of the VPC for the security group.


_class_ EC2.Paginator.DescribeSnapshots

paginator = client.get_paginator('describe_snapshots')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_snapshots()](#EC2.Client.describe_snapshots "EC2.Client.describe_snapshots").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeSnapshots)

**Request Syntax**

response_iterator = paginator.paginate(
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    OwnerIds=[
        'string',
    ],
    RestorableByUserIds=[
        'string',
    ],
    SnapshotIds=[
        'string',
    ],
    DryRun=True|False,
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **Filters** (_list_) --

    The filters.

    - description - A description of the snapshot.
    - encrypted - Indicates whether the snapshot is encrypted (true | false )
    - owner-alias - The owner alias, from an Amazon-maintained list (amazon ). This is not the user-configured AWS account alias set using the IAM console. We recommend that you use the related parameter instead of this filter.
    - owner-id - The AWS account ID of the owner. We recommend that you use the related parameter instead of this filter.
    - progress - The progress of the snapshot, as a percentage (for example, 80%).
    - snapshot-id - The snapshot ID.
    - start-time - The time stamp when the snapshot was initiated.
    - status - The status of the snapshot (pending | completed | error ).
    - tag :<key> - The key/value combination of a tag assigned to the resource. Use the tag key in the filter name and the tag value as the filter value. For example, to find all resources that have a tag with the key Owner and the value TeamA , specify tag:Owner for the filter name and TeamA for the filter value.
    - tag-key - The key of a tag assigned to the resource. Use this filter to find all resources assigned a tag with a specific key, regardless of the tag value.
    - volume-id - The ID of the volume the snapshot is for.
    - volume-size - The size of the volume, in GiB.

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **OwnerIds** (_list_) --

    Scopes the results to snapshots with the specified owners. You can specify a combination of AWS account IDs, self , and amazon .

    - _(string) --_
- **RestorableByUserIds** (_list_) --

    The IDs of the AWS accounts that can create volumes from the snapshot.

    - _(string) --_
- **SnapshotIds** (_list_) --

    The snapshot IDs.

    Default: Describes the snapshots for which you have create volume permissions.

    - _(string) --_
- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'Snapshots': [
        {
            'DataEncryptionKeyId': 'string',
            'Description': 'string',
            'Encrypted': True|False,
            'KmsKeyId': 'string',
            'OwnerId': 'string',
            'Progress': 'string',
            'SnapshotId': 'string',
            'StartTime': datetime(2015, 1, 1),
            'State': 'pending'|'completed'|'error',
            'StateMessage': 'string',
            'VolumeId': 'string',
            'VolumeSize': 123,
            'OwnerAlias': 'string',
            'Tags': [
                {
                    'Key': 'string',
                    'Value': 'string'
                },
            ]
        },
    ],

}

**Response Structure**

- _(dict) --_

    - **Snapshots** _(list) --_

        Information about the snapshots.

        - _(dict) --_

            Describes a snapshot.

            - **DataEncryptionKeyId** _(string) --_

                The data encryption key identifier for the snapshot. This value is a unique identifier that corresponds to the data encryption key that was used to encrypt the original volume or snapshot copy. Because data encryption keys are inherited by volumes created from snapshots, and vice versa, if snapshots share the same data encryption key identifier, then they belong to the same volume/snapshot lineage. This parameter is only returned by DescribeSnapshots .

            - **Description** _(string) --_

                The description for the snapshot.

            - **Encrypted** _(boolean) --_

                Indicates whether the snapshot is encrypted.

            - **KmsKeyId** _(string) --_

                The Amazon Resource Name (ARN) of the AWS Key Management Service (AWS KMS) customer master key (CMK) that was used to protect the volume encryption key for the parent volume.

            - **OwnerId** _(string) --_

                The AWS account ID of the EBS snapshot owner.

            - **Progress** _(string) --_

                The progress of the snapshot, as a percentage.

            - **SnapshotId** _(string) --_

                The ID of the snapshot. Each snapshot receives a unique identifier when it is created.

            - **StartTime** _(datetime) --_

                The time stamp when the snapshot was initiated.

            - **State** _(string) --_

                The snapshot state.

            - **StateMessage** _(string) --_

                Encrypted Amazon EBS snapshots are copied asynchronously. If a snapshot copy operation fails (for example, if the proper AWS Key Management Service (AWS KMS) permissions are not obtained) this field displays error state details to help you diagnose why the error occurred. This parameter is only returned by DescribeSnapshots .

            - **VolumeId** _(string) --_

                The ID of the volume that was used to create the snapshot. Snapshots created by the CopySnapshot action have an arbitrary volume ID that should not be used for any purpose.

            - **VolumeSize** _(integer) --_

                The size of the volume, in GiB.

            - **OwnerAlias** _(string) --_

                The AWS owner alias, from an Amazon-maintained list (amazon ). This is not the user-configured AWS account alias set using the IAM console.

            - **Tags** _(list) --_

                Any tags assigned to the snapshot.

                - _(dict) --_

                    Describes a tag.

                    - **Key** _(string) --_

                        The key of the tag.

                        Constraints: Tag keys are case-sensitive and accept a maximum of 127 Unicode characters. May not begin with aws: .

                    - **Value** _(string) --_

                        The value of the tag.

                        Constraints: Tag values are case-sensitive and accept a maximum of 255 Unicode characters.


_class_ EC2.Paginator.DescribeSpotFleetInstances

paginator = client.get_paginator('describe_spot_fleet_instances')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_spot_fleet_instances()](#EC2.Client.describe_spot_fleet_instances "EC2.Client.describe_spot_fleet_instances").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeSpotFleetInstances)

**Request Syntax**

response_iterator = paginator.paginate(
    DryRun=True|False,
    SpotFleetRequestId='string',
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **SpotFleetRequestId** (_string_) --

    **[REQUIRED]**

    The ID of the Spot Fleet request.

- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'ActiveInstances': [
        {
            'InstanceId': 'string',
            'InstanceType': 'string',
            'SpotInstanceRequestId': 'string',
            'InstanceHealth': 'healthy'|'unhealthy'
        },
    ],
    'SpotFleetRequestId': 'string'
}

**Response Structure**

- _(dict) --_

    Contains the output of DescribeSpotFleetInstances.

    - **ActiveInstances** _(list) --_

        The running instances. This list is refreshed periodically and might be out of date.

        - _(dict) --_

            Describes a running instance in a Spot Fleet.

            - **InstanceId** _(string) --_

                The ID of the instance.

            - **InstanceType** _(string) --_

                The instance type.

            - **SpotInstanceRequestId** _(string) --_

                The ID of the Spot Instance request.

            - **InstanceHealth** _(string) --_

                The health status of the instance. If the status of either the instance status check or the system status check is impaired , the health status of the instance is unhealthy . Otherwise, the health status is healthy .

    - **SpotFleetRequestId** _(string) --_

        The ID of the Spot Fleet request.


_class_ EC2.Paginator.DescribeSpotFleetRequests

paginator = client.get_paginator('describe_spot_fleet_requests')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_spot_fleet_requests()](#EC2.Client.describe_spot_fleet_requests "EC2.Client.describe_spot_fleet_requests").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeSpotFleetRequests)

**Request Syntax**

response_iterator = paginator.paginate(
    DryRun=True|False,
    SpotFleetRequestIds=[
        'string',
    ],
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **SpotFleetRequestIds** (_list_) --

    The IDs of the Spot Fleet requests.

    - _(string) --_
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'SpotFleetRequestConfigs': [
        {
            'ActivityStatus': 'error'|'pending_fulfillment'|'pending_termination'|'fulfilled',
            'CreateTime': datetime(2015, 1, 1),
            'SpotFleetRequestConfig': {
                'AllocationStrategy': 'lowestPrice'|'diversified'|'capacityOptimized',
                'OnDemandAllocationStrategy': 'lowestPrice'|'prioritized',
                'SpotMaintenanceStrategies': {
                    'CapacityRebalance': {
                        'ReplacementStrategy': 'launch'
                    }
                },
                'ClientToken': 'string',
                'ExcessCapacityTerminationPolicy': 'noTermination'|'default',
                'FulfilledCapacity': 123.0,
                'OnDemandFulfilledCapacity': 123.0,
                'IamFleetRole': 'string',
                'LaunchSpecifications': [
                    {
                        'SecurityGroups': [
                            {
                                'GroupName': 'string',
                                'GroupId': 'string'
                            },
                        ],
                        'AddressingType': 'string',
                        'BlockDeviceMappings': [
                            {
                                'DeviceName': 'string',
                                'VirtualName': 'string',
                                'Ebs': {
                                    'DeleteOnTermination': True|False,
                                    'Iops': 123,
                                    'SnapshotId': 'string',
                                    'VolumeSize': 123,
                                    'VolumeType': 'standard'|'io1'|'io2'|'gp2'|'sc1'|'st1'|'gp3',
                                    'KmsKeyId': 'string',
                                    'Throughput': 123,
                                    'Encrypted': True|False
                                },
                                'NoDevice': 'string'
                            },
                        ],
                        'EbsOptimized': True|False,
                        'IamInstanceProfile': {
                            'Arn': 'string',
                            'Name': 'string'
                        },
                        'ImageId': 'string',
                        'InstanceType': 't1.micro'|'t2.nano'|'t2.micro'|'t2.small'|'t2.medium'|'t2.large'|'t2.xlarge'|'t2.2xlarge'|'t3.nano'|'t3.micro'|'t3.small'|'t3.medium'|'t3.large'|'t3.xlarge'|'t3.2xlarge'|'t3a.nano'|'t3a.micro'|'t3a.small'|'t3a.medium'|'t3a.large'|'t3a.xlarge'|'t3a.2xlarge'|'t4g.nano'|'t4g.micro'|'t4g.small'|'t4g.medium'|'t4g.large'|'t4g.xlarge'|'t4g.2xlarge'|'m1.small'|'m1.medium'|'m1.large'|'m1.xlarge'|'m3.medium'|'m3.large'|'m3.xlarge'|'m3.2xlarge'|'m4.large'|'m4.xlarge'|'m4.2xlarge'|'m4.4xlarge'|'m4.10xlarge'|'m4.16xlarge'|'m2.xlarge'|'m2.2xlarge'|'m2.4xlarge'|'cr1.8xlarge'|'r3.large'|'r3.xlarge'|'r3.2xlarge'|'r3.4xlarge'|'r3.8xlarge'|'r4.large'|'r4.xlarge'|'r4.2xlarge'|'r4.4xlarge'|'r4.8xlarge'|'r4.16xlarge'|'r5.large'|'r5.xlarge'|'r5.2xlarge'|'r5.4xlarge'|'r5.8xlarge'|'r5.12xlarge'|'r5.16xlarge'|'r5.24xlarge'|'r5.metal'|'r5a.large'|'r5a.xlarge'|'r5a.2xlarge'|'r5a.4xlarge'|'r5a.8xlarge'|'r5a.12xlarge'|'r5a.16xlarge'|'r5a.24xlarge'|'r5b.large'|'r5b.xlarge'|'r5b.2xlarge'|'r5b.4xlarge'|'r5b.8xlarge'|'r5b.12xlarge'|'r5b.16xlarge'|'r5b.24xlarge'|'r5b.metal'|'r5d.large'|'r5d.xlarge'|'r5d.2xlarge'|'r5d.4xlarge'|'r5d.8xlarge'|'r5d.12xlarge'|'r5d.16xlarge'|'r5d.24xlarge'|'r5d.metal'|'r5ad.large'|'r5ad.xlarge'|'r5ad.2xlarge'|'r5ad.4xlarge'|'r5ad.8xlarge'|'r5ad.12xlarge'|'r5ad.16xlarge'|'r5ad.24xlarge'|'r6g.metal'|'r6g.medium'|'r6g.large'|'r6g.xlarge'|'r6g.2xlarge'|'r6g.4xlarge'|'r6g.8xlarge'|'r6g.12xlarge'|'r6g.16xlarge'|'r6gd.metal'|'r6gd.medium'|'r6gd.large'|'r6gd.xlarge'|'r6gd.2xlarge'|'r6gd.4xlarge'|'r6gd.8xlarge'|'r6gd.12xlarge'|'r6gd.16xlarge'|'x1.16xlarge'|'x1.32xlarge'|'x1e.xlarge'|'x1e.2xlarge'|'x1e.4xlarge'|'x1e.8xlarge'|'x1e.16xlarge'|'x1e.32xlarge'|'i2.xlarge'|'i2.2xlarge'|'i2.4xlarge'|'i2.8xlarge'|'i3.large'|'i3.xlarge'|'i3.2xlarge'|'i3.4xlarge'|'i3.8xlarge'|'i3.16xlarge'|'i3.metal'|'i3en.large'|'i3en.xlarge'|'i3en.2xlarge'|'i3en.3xlarge'|'i3en.6xlarge'|'i3en.12xlarge'|'i3en.24xlarge'|'i3en.metal'|'hi1.4xlarge'|'hs1.8xlarge'|'c1.medium'|'c1.xlarge'|'c3.large'|'c3.xlarge'|'c3.2xlarge'|'c3.4xlarge'|'c3.8xlarge'|'c4.large'|'c4.xlarge'|'c4.2xlarge'|'c4.4xlarge'|'c4.8xlarge'|'c5.large'|'c5.xlarge'|'c5.2xlarge'|'c5.4xlarge'|'c5.9xlarge'|'c5.12xlarge'|'c5.18xlarge'|'c5.24xlarge'|'c5.metal'|'c5a.large'|'c5a.xlarge'|'c5a.2xlarge'|'c5a.4xlarge'|'c5a.8xlarge'|'c5a.12xlarge'|'c5a.16xlarge'|'c5a.24xlarge'|'c5ad.large'|'c5ad.xlarge'|'c5ad.2xlarge'|'c5ad.4xlarge'|'c5ad.8xlarge'|'c5ad.12xlarge'|'c5ad.16xlarge'|'c5ad.24xlarge'|'c5d.large'|'c5d.xlarge'|'c5d.2xlarge'|'c5d.4xlarge'|'c5d.9xlarge'|'c5d.12xlarge'|'c5d.18xlarge'|'c5d.24xlarge'|'c5d.metal'|'c5n.large'|'c5n.xlarge'|'c5n.2xlarge'|'c5n.4xlarge'|'c5n.9xlarge'|'c5n.18xlarge'|'c5n.metal'|'c6g.metal'|'c6g.medium'|'c6g.large'|'c6g.xlarge'|'c6g.2xlarge'|'c6g.4xlarge'|'c6g.8xlarge'|'c6g.12xlarge'|'c6g.16xlarge'|'c6gd.metal'|'c6gd.medium'|'c6gd.large'|'c6gd.xlarge'|'c6gd.2xlarge'|'c6gd.4xlarge'|'c6gd.8xlarge'|'c6gd.12xlarge'|'c6gd.16xlarge'|'c6gn.medium'|'c6gn.large'|'c6gn.xlarge'|'c6gn.2xlarge'|'c6gn.4xlarge'|'c6gn.8xlarge'|'c6gn.12xlarge'|'c6gn.16xlarge'|'cc1.4xlarge'|'cc2.8xlarge'|'g2.2xlarge'|'g2.8xlarge'|'g3.4xlarge'|'g3.8xlarge'|'g3.16xlarge'|'g3s.xlarge'|'g4ad.4xlarge'|'g4ad.8xlarge'|'g4ad.16xlarge'|'g4dn.xlarge'|'g4dn.2xlarge'|'g4dn.4xlarge'|'g4dn.8xlarge'|'g4dn.12xlarge'|'g4dn.16xlarge'|'g4dn.metal'|'cg1.4xlarge'|'p2.xlarge'|'p2.8xlarge'|'p2.16xlarge'|'p3.2xlarge'|'p3.8xlarge'|'p3.16xlarge'|'p3dn.24xlarge'|'p4d.24xlarge'|'d2.xlarge'|'d2.2xlarge'|'d2.4xlarge'|'d2.8xlarge'|'d3.xlarge'|'d3.2xlarge'|'d3.4xlarge'|'d3.8xlarge'|'d3en.xlarge'|'d3en.2xlarge'|'d3en.4xlarge'|'d3en.6xlarge'|'d3en.8xlarge'|'d3en.12xlarge'|'f1.2xlarge'|'f1.4xlarge'|'f1.16xlarge'|'m5.large'|'m5.xlarge'|'m5.2xlarge'|'m5.4xlarge'|'m5.8xlarge'|'m5.12xlarge'|'m5.16xlarge'|'m5.24xlarge'|'m5.metal'|'m5a.large'|'m5a.xlarge'|'m5a.2xlarge'|'m5a.4xlarge'|'m5a.8xlarge'|'m5a.12xlarge'|'m5a.16xlarge'|'m5a.24xlarge'|'m5d.large'|'m5d.xlarge'|'m5d.2xlarge'|'m5d.4xlarge'|'m5d.8xlarge'|'m5d.12xlarge'|'m5d.16xlarge'|'m5d.24xlarge'|'m5d.metal'|'m5ad.large'|'m5ad.xlarge'|'m5ad.2xlarge'|'m5ad.4xlarge'|'m5ad.8xlarge'|'m5ad.12xlarge'|'m5ad.16xlarge'|'m5ad.24xlarge'|'m5zn.large'|'m5zn.xlarge'|'m5zn.2xlarge'|'m5zn.3xlarge'|'m5zn.6xlarge'|'m5zn.12xlarge'|'m5zn.metal'|'h1.2xlarge'|'h1.4xlarge'|'h1.8xlarge'|'h1.16xlarge'|'z1d.large'|'z1d.xlarge'|'z1d.2xlarge'|'z1d.3xlarge'|'z1d.6xlarge'|'z1d.12xlarge'|'z1d.metal'|'u-6tb1.metal'|'u-9tb1.metal'|'u-12tb1.metal'|'u-18tb1.metal'|'u-24tb1.metal'|'a1.medium'|'a1.large'|'a1.xlarge'|'a1.2xlarge'|'a1.4xlarge'|'a1.metal'|'m5dn.large'|'m5dn.xlarge'|'m5dn.2xlarge'|'m5dn.4xlarge'|'m5dn.8xlarge'|'m5dn.12xlarge'|'m5dn.16xlarge'|'m5dn.24xlarge'|'m5n.large'|'m5n.xlarge'|'m5n.2xlarge'|'m5n.4xlarge'|'m5n.8xlarge'|'m5n.12xlarge'|'m5n.16xlarge'|'m5n.24xlarge'|'r5dn.large'|'r5dn.xlarge'|'r5dn.2xlarge'|'r5dn.4xlarge'|'r5dn.8xlarge'|'r5dn.12xlarge'|'r5dn.16xlarge'|'r5dn.24xlarge'|'r5n.large'|'r5n.xlarge'|'r5n.2xlarge'|'r5n.4xlarge'|'r5n.8xlarge'|'r5n.12xlarge'|'r5n.16xlarge'|'r5n.24xlarge'|'inf1.xlarge'|'inf1.2xlarge'|'inf1.6xlarge'|'inf1.24xlarge'|'m6g.metal'|'m6g.medium'|'m6g.large'|'m6g.xlarge'|'m6g.2xlarge'|'m6g.4xlarge'|'m6g.8xlarge'|'m6g.12xlarge'|'m6g.16xlarge'|'m6gd.metal'|'m6gd.medium'|'m6gd.large'|'m6gd.xlarge'|'m6gd.2xlarge'|'m6gd.4xlarge'|'m6gd.8xlarge'|'m6gd.12xlarge'|'m6gd.16xlarge'|'mac1.metal',
                        'KernelId': 'string',
                        'KeyName': 'string',
                        'Monitoring': {
                            'Enabled': True|False
                        },
                        'NetworkInterfaces': [
                            {
                                'AssociatePublicIpAddress': True|False,
                                'DeleteOnTermination': True|False,
                                'Description': 'string',
                                'DeviceIndex': 123,
                                'Groups': [
                                    'string',
                                ],
                                'Ipv6AddressCount': 123,
                                'Ipv6Addresses': [
                                    {
                                        'Ipv6Address': 'string'
                                    },
                                ],
                                'NetworkInterfaceId': 'string',
                                'PrivateIpAddress': 'string',
                                'PrivateIpAddresses': [
                                    {
                                        'Primary': True|False,
                                        'PrivateIpAddress': 'string'
                                    },
                                ],
                                'SecondaryPrivateIpAddressCount': 123,
                                'SubnetId': 'string',
                                'AssociateCarrierIpAddress': True|False,
                                'InterfaceType': 'string',
                                'NetworkCardIndex': 123
                            },
                        ],
                        'Placement': {
                            'AvailabilityZone': 'string',
                            'GroupName': 'string',
                            'Tenancy': 'default'|'dedicated'|'host'
                        },
                        'RamdiskId': 'string',
                        'SpotPrice': 'string',
                        'SubnetId': 'string',
                        'UserData': 'string',
                        'WeightedCapacity': 123.0,
                        'TagSpecifications': [
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
                    },
                ],
                'LaunchTemplateConfigs': [
                    {
                        'LaunchTemplateSpecification': {
                            'LaunchTemplateId': 'string',
                            'LaunchTemplateName': 'string',
                            'Version': 'string'
                        },
                        'Overrides': [
                            {
                                'InstanceType': 't1.micro'|'t2.nano'|'t2.micro'|'t2.small'|'t2.medium'|'t2.large'|'t2.xlarge'|'t2.2xlarge'|'t3.nano'|'t3.micro'|'t3.small'|'t3.medium'|'t3.large'|'t3.xlarge'|'t3.2xlarge'|'t3a.nano'|'t3a.micro'|'t3a.small'|'t3a.medium'|'t3a.large'|'t3a.xlarge'|'t3a.2xlarge'|'t4g.nano'|'t4g.micro'|'t4g.small'|'t4g.medium'|'t4g.large'|'t4g.xlarge'|'t4g.2xlarge'|'m1.small'|'m1.medium'|'m1.large'|'m1.xlarge'|'m3.medium'|'m3.large'|'m3.xlarge'|'m3.2xlarge'|'m4.large'|'m4.xlarge'|'m4.2xlarge'|'m4.4xlarge'|'m4.10xlarge'|'m4.16xlarge'|'m2.xlarge'|'m2.2xlarge'|'m2.4xlarge'|'cr1.8xlarge'|'r3.large'|'r3.xlarge'|'r3.2xlarge'|'r3.4xlarge'|'r3.8xlarge'|'r4.large'|'r4.xlarge'|'r4.2xlarge'|'r4.4xlarge'|'r4.8xlarge'|'r4.16xlarge'|'r5.large'|'r5.xlarge'|'r5.2xlarge'|'r5.4xlarge'|'r5.8xlarge'|'r5.12xlarge'|'r5.16xlarge'|'r5.24xlarge'|'r5.metal'|'r5a.large'|'r5a.xlarge'|'r5a.2xlarge'|'r5a.4xlarge'|'r5a.8xlarge'|'r5a.12xlarge'|'r5a.16xlarge'|'r5a.24xlarge'|'r5b.large'|'r5b.xlarge'|'r5b.2xlarge'|'r5b.4xlarge'|'r5b.8xlarge'|'r5b.12xlarge'|'r5b.16xlarge'|'r5b.24xlarge'|'r5b.metal'|'r5d.large'|'r5d.xlarge'|'r5d.2xlarge'|'r5d.4xlarge'|'r5d.8xlarge'|'r5d.12xlarge'|'r5d.16xlarge'|'r5d.24xlarge'|'r5d.metal'|'r5ad.large'|'r5ad.xlarge'|'r5ad.2xlarge'|'r5ad.4xlarge'|'r5ad.8xlarge'|'r5ad.12xlarge'|'r5ad.16xlarge'|'r5ad.24xlarge'|'r6g.metal'|'r6g.medium'|'r6g.large'|'r6g.xlarge'|'r6g.2xlarge'|'r6g.4xlarge'|'r6g.8xlarge'|'r6g.12xlarge'|'r6g.16xlarge'|'r6gd.metal'|'r6gd.medium'|'r6gd.large'|'r6gd.xlarge'|'r6gd.2xlarge'|'r6gd.4xlarge'|'r6gd.8xlarge'|'r6gd.12xlarge'|'r6gd.16xlarge'|'x1.16xlarge'|'x1.32xlarge'|'x1e.xlarge'|'x1e.2xlarge'|'x1e.4xlarge'|'x1e.8xlarge'|'x1e.16xlarge'|'x1e.32xlarge'|'i2.xlarge'|'i2.2xlarge'|'i2.4xlarge'|'i2.8xlarge'|'i3.large'|'i3.xlarge'|'i3.2xlarge'|'i3.4xlarge'|'i3.8xlarge'|'i3.16xlarge'|'i3.metal'|'i3en.large'|'i3en.xlarge'|'i3en.2xlarge'|'i3en.3xlarge'|'i3en.6xlarge'|'i3en.12xlarge'|'i3en.24xlarge'|'i3en.metal'|'hi1.4xlarge'|'hs1.8xlarge'|'c1.medium'|'c1.xlarge'|'c3.large'|'c3.xlarge'|'c3.2xlarge'|'c3.4xlarge'|'c3.8xlarge'|'c4.large'|'c4.xlarge'|'c4.2xlarge'|'c4.4xlarge'|'c4.8xlarge'|'c5.large'|'c5.xlarge'|'c5.2xlarge'|'c5.4xlarge'|'c5.9xlarge'|'c5.12xlarge'|'c5.18xlarge'|'c5.24xlarge'|'c5.metal'|'c5a.large'|'c5a.xlarge'|'c5a.2xlarge'|'c5a.4xlarge'|'c5a.8xlarge'|'c5a.12xlarge'|'c5a.16xlarge'|'c5a.24xlarge'|'c5ad.large'|'c5ad.xlarge'|'c5ad.2xlarge'|'c5ad.4xlarge'|'c5ad.8xlarge'|'c5ad.12xlarge'|'c5ad.16xlarge'|'c5ad.24xlarge'|'c5d.large'|'c5d.xlarge'|'c5d.2xlarge'|'c5d.4xlarge'|'c5d.9xlarge'|'c5d.12xlarge'|'c5d.18xlarge'|'c5d.24xlarge'|'c5d.metal'|'c5n.large'|'c5n.xlarge'|'c5n.2xlarge'|'c5n.4xlarge'|'c5n.9xlarge'|'c5n.18xlarge'|'c5n.metal'|'c6g.metal'|'c6g.medium'|'c6g.large'|'c6g.xlarge'|'c6g.2xlarge'|'c6g.4xlarge'|'c6g.8xlarge'|'c6g.12xlarge'|'c6g.16xlarge'|'c6gd.metal'|'c6gd.medium'|'c6gd.large'|'c6gd.xlarge'|'c6gd.2xlarge'|'c6gd.4xlarge'|'c6gd.8xlarge'|'c6gd.12xlarge'|'c6gd.16xlarge'|'c6gn.medium'|'c6gn.large'|'c6gn.xlarge'|'c6gn.2xlarge'|'c6gn.4xlarge'|'c6gn.8xlarge'|'c6gn.12xlarge'|'c6gn.16xlarge'|'cc1.4xlarge'|'cc2.8xlarge'|'g2.2xlarge'|'g2.8xlarge'|'g3.4xlarge'|'g3.8xlarge'|'g3.16xlarge'|'g3s.xlarge'|'g4ad.4xlarge'|'g4ad.8xlarge'|'g4ad.16xlarge'|'g4dn.xlarge'|'g4dn.2xlarge'|'g4dn.4xlarge'|'g4dn.8xlarge'|'g4dn.12xlarge'|'g4dn.16xlarge'|'g4dn.metal'|'cg1.4xlarge'|'p2.xlarge'|'p2.8xlarge'|'p2.16xlarge'|'p3.2xlarge'|'p3.8xlarge'|'p3.16xlarge'|'p3dn.24xlarge'|'p4d.24xlarge'|'d2.xlarge'|'d2.2xlarge'|'d2.4xlarge'|'d2.8xlarge'|'d3.xlarge'|'d3.2xlarge'|'d3.4xlarge'|'d3.8xlarge'|'d3en.xlarge'|'d3en.2xlarge'|'d3en.4xlarge'|'d3en.6xlarge'|'d3en.8xlarge'|'d3en.12xlarge'|'f1.2xlarge'|'f1.4xlarge'|'f1.16xlarge'|'m5.large'|'m5.xlarge'|'m5.2xlarge'|'m5.4xlarge'|'m5.8xlarge'|'m5.12xlarge'|'m5.16xlarge'|'m5.24xlarge'|'m5.metal'|'m5a.large'|'m5a.xlarge'|'m5a.2xlarge'|'m5a.4xlarge'|'m5a.8xlarge'|'m5a.12xlarge'|'m5a.16xlarge'|'m5a.24xlarge'|'m5d.large'|'m5d.xlarge'|'m5d.2xlarge'|'m5d.4xlarge'|'m5d.8xlarge'|'m5d.12xlarge'|'m5d.16xlarge'|'m5d.24xlarge'|'m5d.metal'|'m5ad.large'|'m5ad.xlarge'|'m5ad.2xlarge'|'m5ad.4xlarge'|'m5ad.8xlarge'|'m5ad.12xlarge'|'m5ad.16xlarge'|'m5ad.24xlarge'|'m5zn.large'|'m5zn.xlarge'|'m5zn.2xlarge'|'m5zn.3xlarge'|'m5zn.6xlarge'|'m5zn.12xlarge'|'m5zn.metal'|'h1.2xlarge'|'h1.4xlarge'|'h1.8xlarge'|'h1.16xlarge'|'z1d.large'|'z1d.xlarge'|'z1d.2xlarge'|'z1d.3xlarge'|'z1d.6xlarge'|'z1d.12xlarge'|'z1d.metal'|'u-6tb1.metal'|'u-9tb1.metal'|'u-12tb1.metal'|'u-18tb1.metal'|'u-24tb1.metal'|'a1.medium'|'a1.large'|'a1.xlarge'|'a1.2xlarge'|'a1.4xlarge'|'a1.metal'|'m5dn.large'|'m5dn.xlarge'|'m5dn.2xlarge'|'m5dn.4xlarge'|'m5dn.8xlarge'|'m5dn.12xlarge'|'m5dn.16xlarge'|'m5dn.24xlarge'|'m5n.large'|'m5n.xlarge'|'m5n.2xlarge'|'m5n.4xlarge'|'m5n.8xlarge'|'m5n.12xlarge'|'m5n.16xlarge'|'m5n.24xlarge'|'r5dn.large'|'r5dn.xlarge'|'r5dn.2xlarge'|'r5dn.4xlarge'|'r5dn.8xlarge'|'r5dn.12xlarge'|'r5dn.16xlarge'|'r5dn.24xlarge'|'r5n.large'|'r5n.xlarge'|'r5n.2xlarge'|'r5n.4xlarge'|'r5n.8xlarge'|'r5n.12xlarge'|'r5n.16xlarge'|'r5n.24xlarge'|'inf1.xlarge'|'inf1.2xlarge'|'inf1.6xlarge'|'inf1.24xlarge'|'m6g.metal'|'m6g.medium'|'m6g.large'|'m6g.xlarge'|'m6g.2xlarge'|'m6g.4xlarge'|'m6g.8xlarge'|'m6g.12xlarge'|'m6g.16xlarge'|'m6gd.metal'|'m6gd.medium'|'m6gd.large'|'m6gd.xlarge'|'m6gd.2xlarge'|'m6gd.4xlarge'|'m6gd.8xlarge'|'m6gd.12xlarge'|'m6gd.16xlarge'|'mac1.metal',
                                'SpotPrice': 'string',
                                'SubnetId': 'string',
                                'AvailabilityZone': 'string',
                                'WeightedCapacity': 123.0,
                                'Priority': 123.0
                            },
                        ]
                    },
                ],
                'SpotPrice': 'string',
                'TargetCapacity': 123,
                'OnDemandTargetCapacity': 123,
                'OnDemandMaxTotalPrice': 'string',
                'SpotMaxTotalPrice': 'string',
                'TerminateInstancesWithExpiration': True|False,
                'Type': 'request'|'maintain'|'instant',
                'ValidFrom': datetime(2015, 1, 1),
                'ValidUntil': datetime(2015, 1, 1),
                'ReplaceUnhealthyInstances': True|False,
                'InstanceInterruptionBehavior': 'hibernate'|'stop'|'terminate',
                'LoadBalancersConfig': {
                    'ClassicLoadBalancersConfig': {
                        'ClassicLoadBalancers': [
                            {
                                'Name': 'string'
                            },
                        ]
                    },
                    'TargetGroupsConfig': {
                        'TargetGroups': [
                            {
                                'Arn': 'string'
                            },
                        ]
                    }
                },
                'InstancePoolsToUseCount': 123,
                'TagSpecifications': [
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
            },
            'SpotFleetRequestId': 'string',
            'SpotFleetRequestState': 'submitted'|'active'|'cancelled'|'failed'|'cancelled_running'|'cancelled_terminating'|'modifying',
            'Tags': [
                {
                    'Key': 'string',
                    'Value': 'string'
                },
            ]
        },
    ]
}

**Response Structure**

- _(dict) --_

    Contains the output of DescribeSpotFleetRequests.

    - **SpotFleetRequestConfigs** _(list) --_

        Information about the configuration of your Spot Fleet.

        - _(dict) --_

            Describes a Spot Fleet request.

            - **ActivityStatus** _(string) --_

                The progress of the Spot Fleet request. If there is an error, the status is error . After all requests are placed, the status is pending_fulfillment . If the size of the fleet is equal to or greater than its target capacity, the status is fulfilled . If the size of the fleet is decreased, the status is pending_termination while Spot Instances are terminating.

            - **CreateTime** _(datetime) --_

                The creation date and time of the request.

            - **SpotFleetRequestConfig** _(dict) --_

                The configuration of the Spot Fleet request.

                - **AllocationStrategy** _(string) --_

                    Indicates how to allocate the target Spot Instance capacity across the Spot Instance pools specified by the Spot Fleet request.

                    If the allocation strategy is lowestPrice , Spot Fleet launches instances from the Spot Instance pools with the lowest price. This is the default allocation strategy.

                    If the allocation strategy is diversified , Spot Fleet launches instances from all the Spot Instance pools that you specify.

                    If the allocation strategy is capacityOptimized , Spot Fleet launches instances from Spot Instance pools with optimal capacity for the number of instances that are launching.

                - **OnDemandAllocationStrategy** _(string) --_

                    The order of the launch template overrides to use in fulfilling On-Demand capacity. If you specify lowestPrice , Spot Fleet uses price to determine the order, launching the lowest price first. If you specify prioritized , Spot Fleet uses the priority that you assign to each Spot Fleet launch template override, launching the highest priority first. If you do not specify a value, Spot Fleet defaults to lowestPrice .

                - **SpotMaintenanceStrategies** _(dict) --_

                    The strategies for managing your Spot Instances that are at an elevated risk of being interrupted.

                    - **CapacityRebalance** _(dict) --_

                        The strategy to use when Amazon EC2 emits a signal that your Spot Instance is at an elevated risk of being interrupted.

                        - **ReplacementStrategy** _(string) --_

                            The replacement strategy to use. Only available for fleets of type maintain . You must specify a value, otherwise you get an error.

                            To allow Spot Fleet to launch a replacement Spot Instance when an instance rebalance notification is emitted for a Spot Instance in the fleet, specify launch .

                            Note

                            When a replacement instance is launched, the instance marked for rebalance is not automatically terminated. You can terminate it, or you can leave it running. You are charged for all instances while they are running.

                - **ClientToken** _(string) --_

                    A unique, case-sensitive identifier that you provide to ensure the idempotency of your listings. This helps to avoid duplicate listings. For more information, see [Ensuring Idempotency](https://docs.aws.amazon.com/AWSEC2/latest/APIReference/Run_Instance_Idempotency.html) .

                - **ExcessCapacityTerminationPolicy** _(string) --_

                    Indicates whether running Spot Instances should be terminated if you decrease the target capacity of the Spot Fleet request below the current size of the Spot Fleet.

                - **FulfilledCapacity** _(float) --_

                    The number of units fulfilled by this request compared to the set target capacity. You cannot set this value.

                - **OnDemandFulfilledCapacity** _(float) --_

                    The number of On-Demand units fulfilled by this request compared to the set target On-Demand capacity.

                - **IamFleetRole** _(string) --_

                    The Amazon Resource Name (ARN) of an AWS Identity and Access Management (IAM) role that grants the Spot Fleet the permission to request, launch, terminate, and tag instances on your behalf. For more information, see [Spot Fleet prerequisites](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/spot-fleet-requests.html#spot-fleet-prerequisites) in the _Amazon EC2 User Guide for Linux Instances_ . Spot Fleet can terminate Spot Instances on your behalf when you cancel its Spot Fleet request using [CancelSpotFleetRequests](https://docs.aws.amazon.com/AWSEC2/latest/APIReference/API_CancelSpotFleetRequests) or when the Spot Fleet request expires, if you set TerminateInstancesWithExpiration .

                - **LaunchSpecifications** _(list) --_

                    The launch specifications for the Spot Fleet request. If you specify LaunchSpecifications , you can't specify LaunchTemplateConfigs . If you include On-Demand capacity in your request, you must use LaunchTemplateConfigs .

                    - _(dict) --_

                        Describes the launch specification for one or more Spot Instances. If you include On-Demand capacity in your fleet request or want to specify an EFA network device, you can't use SpotFleetLaunchSpecification ; you must use [LaunchTemplateConfig](https://docs.aws.amazon.com/AWSEC2/latest/APIReference/API_LaunchTemplateConfig.html) .

                        - **SecurityGroups** _(list) --_

                            One or more security groups. When requesting instances in a VPC, you must specify the IDs of the security groups. When requesting instances in EC2-Classic, you can specify the names or the IDs of the security groups.

                            - _(dict) --_

                                Describes a security group.

                                - **GroupName** _(string) --_

                                    The name of the security group.

                                - **GroupId** _(string) --_

                                    The ID of the security group.

                        - **AddressingType** _(string) --_

                            Deprecated.

                        - **BlockDeviceMappings** _(list) --_

                            One or more block devices that are mapped to the Spot Instances. You can't specify both a snapshot ID and an encryption value. This is because only blank volumes can be encrypted on creation. If a snapshot is the basis for a volume, it is not blank and its encryption status is used for the volume encryption status.

                            - _(dict) --_

                                Describes a block device mapping.

                                - **DeviceName** _(string) --_

                                    The device name (for example, /dev/sdh or xvdh ).

                                - **VirtualName** _(string) --_

                                    The virtual device name (ephemeral N). Instance store volumes are numbered starting from 0. An instance type with 2 available instance store volumes can specify mappings for ephemeral0 and ephemeral1 . The number of available instance store volumes depends on the instance type. After you connect to the instance, you must mount the volume.

                                    NVMe instance store volumes are automatically enumerated and assigned a device name. Including them in your block device mapping has no effect.

                                    Constraints: For M3 instances, you must specify instance store volumes in the block device mapping for the instance. When you launch an M3 instance, we ignore any instance store volumes specified in the block device mapping for the AMI.

                                - **Ebs** _(dict) --_

                                    Parameters used to automatically set up EBS volumes when the instance is launched.

                                    - **DeleteOnTermination** _(boolean) --_

                                        Indicates whether the EBS volume is deleted on instance termination. For more information, see [Preserving Amazon EBS volumes on instance termination](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/terminating-instances.html#preserving-volumes-on-termination) in the Amazon Elastic Compute Cloud User Guide.

                                    - **Iops** _(integer) --_

                                        The number of I/O operations per second (IOPS). For gp3 , io1 , and io2 volumes, this represents the number of IOPS that are provisioned for the volume. For gp2 volumes, this represents the baseline performance of the volume and the rate at which the volume accumulates I/O credits for bursting.

                                        The following are the supported values for each volume type:

                                        - gp3 : 3,000-16,000 IOPS
                                        - io1 : 100-64,000 IOPS
                                        - io2 : 100-64,000 IOPS

                                        For io1 and io2 volumes, we guarantee 64,000 IOPS only for [Instances built on the Nitro System](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/instance-types.html#ec2-nitro-instances) . Other instance families guarantee performance up to 32,000 IOPS.

                                        This parameter is required for io1 and io2 volumes. The default for gp3 volumes is 3,000 IOPS. This parameter is not supported for gp2 , st1 , sc1 , or standard volumes.

                                    - **SnapshotId** _(string) --_

                                        The ID of the snapshot.

                                    - **VolumeSize** _(integer) --_

                                        The size of the volume, in GiBs. You must specify either a snapshot ID or a volume size. If you specify a snapshot, the default is the snapshot size. You can specify a volume size that is equal to or larger than the snapshot size.

                                        The following are the supported volumes sizes for each volume type:

                                        - gp2 and gp3 :1-16,384
                                        - io1 and io2 : 4-16,384
                                        - st1 : 500-16,384
                                        - sc1 : 500-16,384
                                        - standard : 1-1,024
                                    - **VolumeType** _(string) --_

                                        The volume type. For more information, see [Amazon EBS volume types](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EBSVolumeTypes.html) in the _Amazon Elastic Compute Cloud User Guide_ . If the volume type is io1 or io2 , you must specify the IOPS that the volume supports.

                                    - **KmsKeyId** _(string) --_

                                        Identifier (key ID, key alias, ID ARN, or alias ARN) for a customer managed CMK under which the EBS volume is encrypted.

                                        This parameter is only supported on BlockDeviceMapping objects called by [RunInstances](https://docs.aws.amazon.com/AWSEC2/latest/APIReference/API_RunInstances.html) , [RequestSpotFleet](https://docs.aws.amazon.com/AWSEC2/latest/APIReference/API_RequestSpotFleet.html) , and [RequestSpotInstances](https://docs.aws.amazon.com/AWSEC2/latest/APIReference/API_RequestSpotInstances.html) .

                                    - **Throughput** _(integer) --_

                                        The throughput that the volume supports, in MiB/s.

                                        This parameter is valid only for gp3 volumes.

                                        Valid Range: Minimum value of 125. Maximum value of 1000.

                                    - **Encrypted** _(boolean) --_

                                        Indicates whether the encryption state of an EBS volume is changed while being restored from a backing snapshot. The effect of setting the encryption state to true depends on the volume origin (new or from a snapshot), starting encryption state, ownership, and whether encryption by default is enabled. For more information, see [Amazon EBS Encryption](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EBSEncryption.html#encryption-parameters) in the _Amazon Elastic Compute Cloud User Guide_ .

                                        In no case can you remove encryption from an encrypted volume.

                                        Encrypted volumes can only be attached to instances that support Amazon EBS encryption. For more information, see [Supported instance types](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EBSEncryption.html#EBSEncryption_supported_instances) .

                                        This parameter is not returned by .

                                - **NoDevice** _(string) --_

                                    Suppresses the specified device included in the block device mapping of the AMI.

                        - **EbsOptimized** _(boolean) --_

                            Indicates whether the instances are optimized for EBS I/O. This optimization provides dedicated throughput to Amazon EBS and an optimized configuration stack to provide optimal EBS I/O performance. This optimization isn't available with all instance types. Additional usage charges apply when using an EBS Optimized instance.

                            Default: false

                        - **IamInstanceProfile** _(dict) --_

                            The IAM instance profile.

                            - **Arn** _(string) --_

                                The Amazon Resource Name (ARN) of the instance profile.

                            - **Name** _(string) --_

                                The name of the instance profile.

                        - **ImageId** _(string) --_

                            The ID of the AMI.

                        - **InstanceType** _(string) --_

                            The instance type.

                        - **KernelId** _(string) --_

                            The ID of the kernel.

                        - **KeyName** _(string) --_

                            The name of the key pair.

                        - **Monitoring** _(dict) --_

                            Enable or disable monitoring for the instances.

                            - **Enabled** _(boolean) --_

                                Enables monitoring for the instance.

                                Default: false

                        - **NetworkInterfaces** _(list) --_

                            One or more network interfaces. If you specify a network interface, you must specify subnet IDs and security group IDs using the network interface.

                            Note

                            SpotFleetLaunchSpecification currently does not support Elastic Fabric Adapter (EFA). To specify an EFA, you must use [LaunchTemplateConfig](https://docs.aws.amazon.com/AWSEC2/latest/APIReference/API_LaunchTemplateConfig.html) .

                            - _(dict) --_

                                Describes a network interface.

                                - **AssociatePublicIpAddress** _(boolean) --_

                                    Indicates whether to assign a public IPv4 address to an instance you launch in a VPC. The public IP address can only be assigned to a network interface for eth0, and can only be assigned to a new network interface, not an existing one. You cannot specify more than one network interface in the request. If launching into a default subnet, the default value is true .

                                - **DeleteOnTermination** _(boolean) --_

                                    If set to true , the interface is deleted when the instance is terminated. You can specify true only if creating a new network interface when launching an instance.

                                - **Description** _(string) --_

                                    The description of the network interface. Applies only if creating a network interface when launching an instance.

                                - **DeviceIndex** _(integer) --_

                                    The position of the network interface in the attachment order. A primary network interface has a device index of 0.

                                    If you specify a network interface when launching an instance, you must specify the device index.

                                - **Groups** _(list) --_

                                    The IDs of the security groups for the network interface. Applies only if creating a network interface when launching an instance.

                                    - _(string) --_
                                - **Ipv6AddressCount** _(integer) --_

                                    A number of IPv6 addresses to assign to the network interface. Amazon EC2 chooses the IPv6 addresses from the range of the subnet. You cannot specify this option and the option to assign specific IPv6 addresses in the same request. You can specify this option if you've specified a minimum number of instances to launch.

                                - **Ipv6Addresses** _(list) --_

                                    One or more IPv6 addresses to assign to the network interface. You cannot specify this option and the option to assign a number of IPv6 addresses in the same request. You cannot specify this option if you've specified a minimum number of instances to launch.

                                    - _(dict) --_

                                        Describes an IPv6 address.

                                        - **Ipv6Address** _(string) --_

                                            The IPv6 address.

                                - **NetworkInterfaceId** _(string) --_

                                    The ID of the network interface.

                                    If you are creating a Spot Fleet, omit this parameter because you can’t specify a network interface ID in a launch specification.

                                - **PrivateIpAddress** _(string) --_

                                    The private IPv4 address of the network interface. Applies only if creating a network interface when launching an instance. You cannot specify this option if you're launching more than one instance in a [RunInstances](https://docs.aws.amazon.com/AWSEC2/latest/APIReference/API_RunInstances.html) request.

                                - **PrivateIpAddresses** _(list) --_

                                    One or more private IPv4 addresses to assign to the network interface. Only one private IPv4 address can be designated as primary. You cannot specify this option if you're launching more than one instance in a [RunInstances](https://docs.aws.amazon.com/AWSEC2/latest/APIReference/API_RunInstances.html) request.

                                    - _(dict) --_

                                        Describes a secondary private IPv4 address for a network interface.

                                        - **Primary** _(boolean) --_

                                            Indicates whether the private IPv4 address is the primary private IPv4 address. Only one IPv4 address can be designated as primary.

                                        - **PrivateIpAddress** _(string) --_

                                            The private IPv4 addresses.

                                - **SecondaryPrivateIpAddressCount** _(integer) --_

                                    The number of secondary private IPv4 addresses. You can't specify this option and specify more than one private IP address using the private IP addresses option. You cannot specify this option if you're launching more than one instance in a [RunInstances](https://docs.aws.amazon.com/AWSEC2/latest/APIReference/API_RunInstances.html) request.

                                - **SubnetId** _(string) --_

                                    The ID of the subnet associated with the network interface. Applies only if creating a network interface when launching an instance.

                                - **AssociateCarrierIpAddress** _(boolean) --_

                                    Indicates whether to assign a carrier IP address to the network interface.

                                    You can only assign a carrier IP address to a network interface that is in a subnet in a Wavelength Zone. For more information about carrier IP addresses, see Carrier IP addresses in the AWS Wavelength Developer Guide.

                                - **InterfaceType** _(string) --_

                                    The type of network interface.

                                    To create an Elastic Fabric Adapter (EFA), specify efa . For more information, see [Elastic Fabric Adapter](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html) in the _Amazon Elastic Compute Cloud User Guide_ .

                                    If you are not creating an EFA, specify interface or omit this parameter.

                                    Valid values: interface | efa

                                - **NetworkCardIndex** _(integer) --_

                                    The index of the network card. Some instance types support multiple network cards. The primary network interface must be assigned to network card index 0. The default is network card index 0.

                        - **Placement** _(dict) --_

                            The placement information.

                            - **AvailabilityZone** _(string) --_

                                The Availability Zone.

                                [Spot Fleet only] To specify multiple Availability Zones, separate them using commas; for example, "us-west-2a, us-west-2b".

                            - **GroupName** _(string) --_

                                The name of the placement group.

                            - **Tenancy** _(string) --_

                                The tenancy of the instance (if the instance is running in a VPC). An instance with a tenancy of dedicated runs on single-tenant hardware. The host tenancy is not supported for Spot Instances.

                        - **RamdiskId** _(string) --_

                            The ID of the RAM disk. Some kernels require additional drivers at launch. Check the kernel requirements for information about whether you need to specify a RAM disk. To find kernel requirements, refer to the AWS Resource Center and search for the kernel ID.

                        - **SpotPrice** _(string) --_

                            The maximum price per unit hour that you are willing to pay for a Spot Instance. If this value is not specified, the default is the Spot price specified for the fleet. To determine the Spot price per unit hour, divide the Spot price by the value of WeightedCapacity .

                        - **SubnetId** _(string) --_

                            The IDs of the subnets in which to launch the instances. To specify multiple subnets, separate them using commas; for example, "subnet-1234abcdeexample1, subnet-0987cdef6example2".

                        - **UserData** _(string) --_

                            The Base64-encoded user data that instances use when starting up.

                        - **WeightedCapacity** _(float) --_

                            The number of units provided by the specified instance type. These are the same units that you chose to set the target capacity in terms of instances, or a performance characteristic such as vCPUs, memory, or I/O.

                            If the target capacity divided by this value is not a whole number, Amazon EC2 rounds the number of instances to the next whole number. If this value is not specified, the default is 1.

                        - **TagSpecifications** _(list) --_

                            The tags to apply during creation.

                            - _(dict) --_

                                The tags for a Spot Fleet resource.

                                - **ResourceType** _(string) --_

                                    The type of resource. Currently, the only resource type that is supported is instance . To tag the Spot Fleet request on creation, use the TagSpecifications parameter in ` SpotFleetRequestConfigData [https://docs.aws.amazon.com/AWSEC2/latest/APIReference/API_SpotFleetRequestConfigData](https://docs.aws.amazon.com/AWSEC2/latest/APIReference/API_SpotFleetRequestConfigData).html`__ .

                                - **Tags** _(list) --_

                                    The tags.

                                    - _(dict) --_

                                        Describes a tag.

                                        - **Key** _(string) --_

                                            The key of the tag.

                                            Constraints: Tag keys are case-sensitive and accept a maximum of 127 Unicode characters. May not begin with aws: .

                                        - **Value** _(string) --_

                                            The value of the tag.

                                            Constraints: Tag values are case-sensitive and accept a maximum of 255 Unicode characters.

                - **LaunchTemplateConfigs** _(list) --_

                    The launch template and overrides. If you specify LaunchTemplateConfigs , you can't specify LaunchSpecifications . If you include On-Demand capacity in your request, you must use LaunchTemplateConfigs .

                    - _(dict) --_

                        Describes a launch template and overrides.

                        - **LaunchTemplateSpecification** _(dict) --_

                            The launch template.

                            - **LaunchTemplateId** _(string) --_

                                The ID of the launch template. If you specify the template ID, you can't specify the template name.

                            - **LaunchTemplateName** _(string) --_

                                The name of the launch template. If you specify the template name, you can't specify the template ID.

                            - **Version** _(string) --_

                                The launch template version number, $Latest , or $Default . You must specify a value, otherwise the request fails.

                                If the value is $Latest , Amazon EC2 uses the latest version of the launch template.

                                If the value is $Default , Amazon EC2 uses the default version of the launch template.

                        - **Overrides** _(list) --_

                            Any parameters that you specify override the same parameters in the launch template.

                            - _(dict) --_

                                Describes overrides for a launch template.

                                - **InstanceType** _(string) --_

                                    The instance type.

                                - **SpotPrice** _(string) --_

                                    The maximum price per unit hour that you are willing to pay for a Spot Instance.

                                - **SubnetId** _(string) --_

                                    The ID of the subnet in which to launch the instances.

                                - **AvailabilityZone** _(string) --_

                                    The Availability Zone in which to launch the instances.

                                - **WeightedCapacity** _(float) --_

                                    The number of units provided by the specified instance type.

                                - **Priority** _(float) --_

                                    The priority for the launch template override. If **OnDemandAllocationStrategy** is set to prioritized , Spot Fleet uses priority to determine which launch template override to use first in fulfilling On-Demand capacity. The highest priority is launched first. Valid values are whole numbers starting at 0 . The lower the number, the higher the priority. If no number is set, the launch template override has the lowest priority.

                - **SpotPrice** _(string) --_

                    The maximum price per unit hour that you are willing to pay for a Spot Instance. The default is the On-Demand price.

                - **TargetCapacity** _(integer) --_

                    The number of units to request for the Spot Fleet. You can choose to set the target capacity in terms of instances or a performance characteristic that is important to your application workload, such as vCPUs, memory, or I/O. If the request type is maintain , you can specify a target capacity of 0 and add capacity later.

                - **OnDemandTargetCapacity** _(integer) --_

                    The number of On-Demand units to request. You can choose to set the target capacity in terms of instances or a performance characteristic that is important to your application workload, such as vCPUs, memory, or I/O. If the request type is maintain , you can specify a target capacity of 0 and add capacity later.

                - **OnDemandMaxTotalPrice** _(string) --_

                    The maximum amount per hour for On-Demand Instances that you're willing to pay. You can use the onDemandMaxTotalPrice parameter, the spotMaxTotalPrice parameter, or both parameters to ensure that your fleet cost does not exceed your budget. If you set a maximum price per hour for the On-Demand Instances and Spot Instances in your request, Spot Fleet will launch instances until it reaches the maximum amount you're willing to pay. When the maximum amount you're willing to pay is reached, the fleet stops launching instances even if it hasn’t met the target capacity.

                - **SpotMaxTotalPrice** _(string) --_

                    The maximum amount per hour for Spot Instances that you're willing to pay. You can use the spotdMaxTotalPrice parameter, the onDemandMaxTotalPrice parameter, or both parameters to ensure that your fleet cost does not exceed your budget. If you set a maximum price per hour for the On-Demand Instances and Spot Instances in your request, Spot Fleet will launch instances until it reaches the maximum amount you're willing to pay. When the maximum amount you're willing to pay is reached, the fleet stops launching instances even if it hasn’t met the target capacity.

                - **TerminateInstancesWithExpiration** _(boolean) --_

                    Indicates whether running Spot Instances are terminated when the Spot Fleet request expires.

                - **Type** _(string) --_

                    The type of request. Indicates whether the Spot Fleet only requests the target capacity or also attempts to maintain it. When this value is request , the Spot Fleet only places the required requests. It does not attempt to replenish Spot Instances if capacity is diminished, nor does it submit requests in alternative Spot pools if capacity is not available. When this value is maintain , the Spot Fleet maintains the target capacity. The Spot Fleet places the required requests to meet capacity and automatically replenishes any interrupted instances. Default: maintain . instant is listed but is not used by Spot Fleet.

                - **ValidFrom** _(datetime) --_

                    The start date and time of the request, in UTC format (_YYYY_ -_MM_ -_DD_ T*HH* :_MM_ :_SS_ Z). By default, Amazon EC2 starts fulfilling the request immediately.

                - **ValidUntil** _(datetime) --_

                    The end date and time of the request, in UTC format (_YYYY_ -_MM_ -_DD_ T*HH* :_MM_ :_SS_ Z). After the end date and time, no new Spot Instance requests are placed or able to fulfill the request. If no value is specified, the Spot Fleet request remains until you cancel it.

                - **ReplaceUnhealthyInstances** _(boolean) --_

                    Indicates whether Spot Fleet should replace unhealthy instances.

                - **InstanceInterruptionBehavior** _(string) --_

                    The behavior when a Spot Instance is interrupted. The default is terminate .

                - **LoadBalancersConfig** _(dict) --_

                    One or more Classic Load Balancers and target groups to attach to the Spot Fleet request. Spot Fleet registers the running Spot Instances with the specified Classic Load Balancers and target groups.

                    With Network Load Balancers, Spot Fleet cannot register instances that have the following instance types: C1, CC1, CC2, CG1, CG2, CR1, CS1, G1, G2, HI1, HS1, M1, M2, M3, and T1.

                    - **ClassicLoadBalancersConfig** _(dict) --_

                        The Classic Load Balancers.

                        - **ClassicLoadBalancers** _(list) --_

                            One or more Classic Load Balancers.

                            - _(dict) --_

                                Describes a Classic Load Balancer.

                                - **Name** _(string) --_

                                    The name of the load balancer.

                    - **TargetGroupsConfig** _(dict) --_

                        The target groups.

                        - **TargetGroups** _(list) --_

                            One or more target groups.

                            - _(dict) --_

                                Describes a load balancer target group.

                                - **Arn** _(string) --_

                                    The Amazon Resource Name (ARN) of the target group.

                - **InstancePoolsToUseCount** _(integer) --_

                    The number of Spot pools across which to allocate your target Spot capacity. Valid only when Spot **AllocationStrategy** is set to lowest-price . Spot Fleet selects the cheapest Spot pools and evenly allocates your target Spot capacity across the number of Spot pools that you specify.

                - **TagSpecifications** _(list) --_

                    The key-value pair for tagging the Spot Fleet request on creation. The value for ResourceType must be spot-fleet-request , otherwise the Spot Fleet request fails. To tag instances at launch, specify the tags in the [launch template](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-launch-templates.html#create-launch-template) (valid only if you use LaunchTemplateConfigs ) or in the ` SpotFleetTagSpecification [https://docs.aws.amazon.com/AWSEC2/latest/APIReference/API_SpotFleetTagSpecification](https://docs.aws.amazon.com/AWSEC2/latest/APIReference/API_SpotFleetTagSpecification).html`__ (valid only if you use LaunchSpecifications ). For information about tagging after launch, see [Tagging Your Resources](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/Using_Tags.html#tag-resources) .

                    - _(dict) --_

                        The tags to apply to a resource when the resource is being created.

                        - **ResourceType** _(string) --_

                            The type of resource to tag. Currently, the resource types that support tagging on creation are: capacity-reservation | carrier-gateway | client-vpn-endpoint | customer-gateway | dedicated-host | dhcp-options | egress-only-internet-gateway | elastic-ip | elastic-gpu | export-image-task | export-instance-task | fleet | fpga-image | host-reservation | image | import-image-task | import-snapshot-task | instance | internet-gateway | ipv4pool-ec2 | ipv6pool-ec2 | key-pair | launch-template | local-gateway-route-table-vpc-association | placement-group | prefix-list | natgateway | network-acl | network-interface | reserved-instances [|](#id725)route-table | security-group | snapshot | spot-fleet-request | spot-instances-request | snapshot | subnet | traffic-mirror-filter | traffic-mirror-session | traffic-mirror-target | transit-gateway | transit-gateway-attachment | transit-gateway-multicast-domain | transit-gateway-route-table | volume [|](#id727)vpc | vpc-peering-connection | vpc-endpoint (for interface and gateway endpoints) | vpc-endpoint-service (for AWS PrivateLink) | vpc-flow-log | vpn-connection | vpn-gateway .

                            To tag a resource after it has been created, see [CreateTags](https://docs.aws.amazon.com/AWSEC2/latest/APIReference/API_CreateTags.html) .

                        - **Tags** _(list) --_

                            The tags to apply to the resource.

                            - _(dict) --_

                                Describes a tag.

                                - **Key** _(string) --_

                                    The key of the tag.

                                    Constraints: Tag keys are case-sensitive and accept a maximum of 127 Unicode characters. May not begin with aws: .

                                - **Value** _(string) --_

                                    The value of the tag.

                                    Constraints: Tag values are case-sensitive and accept a maximum of 255 Unicode characters.

            - **SpotFleetRequestId** _(string) --_

                The ID of the Spot Fleet request.

            - **SpotFleetRequestState** _(string) --_

                The state of the Spot Fleet request.

            - **Tags** _(list) --_

                The tags for a Spot Fleet resource.

                - _(dict) --_

                    Describes a tag.

                    - **Key** _(string) --_

                        The key of the tag.

                        Constraints: Tag keys are case-sensitive and accept a maximum of 127 Unicode characters. May not begin with aws: .

                    - **Value** _(string) --_

                        The value of the tag.

                        Constraints: Tag values are case-sensitive and accept a maximum of 255 Unicode characters.


_class_ EC2.Paginator.DescribeSpotInstanceRequests

paginator = client.get_paginator('describe_spot_instance_requests')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_spot_instance_requests()](#EC2.Client.describe_spot_instance_requests "EC2.Client.describe_spot_instance_requests").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeSpotInstanceRequests)

**Request Syntax**

response_iterator = paginator.paginate(
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    DryRun=True|False,
    SpotInstanceRequestIds=[
        'string',
    ],
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **Filters** (_list_) --

    One or more filters.

    - availability-zone-group - The Availability Zone group.
    - create-time - The time stamp when the Spot Instance request was created.
    - fault-code - The fault code related to the request.
    - fault-message - The fault message related to the request.
    - instance-id - The ID of the instance that fulfilled the request.
    - launch-group - The Spot Instance launch group.
    - launch.block-device-mapping.delete-on-termination - Indicates whether the EBS volume is deleted on instance termination.
    - launch.block-device-mapping.device-name - The device name for the volume in the block device mapping (for example, /dev/sdh or xvdh ).
    - launch.block-device-mapping.snapshot-id - The ID of the snapshot for the EBS volume.
    - launch.block-device-mapping.volume-size - The size of the EBS volume, in GiB.
    - launch.block-device-mapping.volume-type - The type of EBS volume: gp2 for General Purpose SSD, io1 or io2 for Provisioned IOPS SSD, st1 for Throughput Optimized HDD, sc1 for Cold HDD, or standard for Magnetic.
    - launch.group-id - The ID of the security group for the instance.
    - launch.group-name - The name of the security group for the instance.
    - launch.image-id - The ID of the AMI.
    - launch.instance-type - The type of instance (for example, m3.medium ).
    - launch.kernel-id - The kernel ID.
    - launch.key-name - The name of the key pair the instance launched with.
    - launch.monitoring-enabled - Whether detailed monitoring is enabled for the Spot Instance.
    - launch.ramdisk-id - The RAM disk ID.
    - launched-availability-zone - The Availability Zone in which the request is launched.
    - network-interface.addresses.primary - Indicates whether the IP address is the primary private IP address.
    - network-interface.delete-on-termination - Indicates whether the network interface is deleted when the instance is terminated.
    - network-interface.description - A description of the network interface.
    - network-interface.device-index - The index of the device for the network interface attachment on the instance.
    - network-interface.group-id - The ID of the security group associated with the network interface.
    - network-interface.network-interface-id - The ID of the network interface.
    - network-interface.private-ip-address - The primary private IP address of the network interface.
    - network-interface.subnet-id - The ID of the subnet for the instance.
    - product-description - The product description associated with the instance (Linux/UNIX | Windows ).
    - spot-instance-request-id - The Spot Instance request ID.
    - spot-price - The maximum hourly price for any Spot Instance launched to fulfill the request.
    - state - The state of the Spot Instance request (open | active | closed | cancelled | failed ). Spot request status information can help you track your Amazon EC2 Spot Instance requests. For more information, see [Spot request status](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/spot-bid-status.html) in the _Amazon EC2 User Guide for Linux Instances_ .
    - status-code - The short code describing the most recent evaluation of your Spot Instance request.
    - status-message - The message explaining the status of the Spot Instance request.
    - tag :<key> - The key/value combination of a tag assigned to the resource. Use the tag key in the filter name and the tag value as the filter value. For example, to find all resources that have a tag with the key Owner and the value TeamA , specify tag:Owner for the filter name and TeamA for the filter value.
    - tag-key - The key of a tag assigned to the resource. Use this filter to find all resources assigned a tag with a specific key, regardless of the tag value.
    - type - The type of Spot Instance request (one-time | persistent ).
    - valid-from - The start date of the request.
    - valid-until - The end date of the request.

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **SpotInstanceRequestIds** (_list_) --

    One or more Spot Instance request IDs.

    - _(string) --_
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'SpotInstanceRequests': [
        {
            'ActualBlockHourlyPrice': 'string',
            'AvailabilityZoneGroup': 'string',
            'BlockDurationMinutes': 123,
            'CreateTime': datetime(2015, 1, 1),
            'Fault': {
                'Code': 'string',
                'Message': 'string'
            },
            'InstanceId': 'string',
            'LaunchGroup': 'string',
            'LaunchSpecification': {
                'UserData': 'string',
                'SecurityGroups': [
                    {
                        'GroupName': 'string',
                        'GroupId': 'string'
                    },
                ],
                'AddressingType': 'string',
                'BlockDeviceMappings': [
                    {
                        'DeviceName': 'string',
                        'VirtualName': 'string',
                        'Ebs': {
                            'DeleteOnTermination': True|False,
                            'Iops': 123,
                            'SnapshotId': 'string',
                            'VolumeSize': 123,
                            'VolumeType': 'standard'|'io1'|'io2'|'gp2'|'sc1'|'st1'|'gp3',
                            'KmsKeyId': 'string',
                            'Throughput': 123,
                            'Encrypted': True|False
                        },
                        'NoDevice': 'string'
                    },
                ],
                'EbsOptimized': True|False,
                'IamInstanceProfile': {
                    'Arn': 'string',
                    'Name': 'string'
                },
                'ImageId': 'string',
                'InstanceType': 't1.micro'|'t2.nano'|'t2.micro'|'t2.small'|'t2.medium'|'t2.large'|'t2.xlarge'|'t2.2xlarge'|'t3.nano'|'t3.micro'|'t3.small'|'t3.medium'|'t3.large'|'t3.xlarge'|'t3.2xlarge'|'t3a.nano'|'t3a.micro'|'t3a.small'|'t3a.medium'|'t3a.large'|'t3a.xlarge'|'t3a.2xlarge'|'t4g.nano'|'t4g.micro'|'t4g.small'|'t4g.medium'|'t4g.large'|'t4g.xlarge'|'t4g.2xlarge'|'m1.small'|'m1.medium'|'m1.large'|'m1.xlarge'|'m3.medium'|'m3.large'|'m3.xlarge'|'m3.2xlarge'|'m4.large'|'m4.xlarge'|'m4.2xlarge'|'m4.4xlarge'|'m4.10xlarge'|'m4.16xlarge'|'m2.xlarge'|'m2.2xlarge'|'m2.4xlarge'|'cr1.8xlarge'|'r3.large'|'r3.xlarge'|'r3.2xlarge'|'r3.4xlarge'|'r3.8xlarge'|'r4.large'|'r4.xlarge'|'r4.2xlarge'|'r4.4xlarge'|'r4.8xlarge'|'r4.16xlarge'|'r5.large'|'r5.xlarge'|'r5.2xlarge'|'r5.4xlarge'|'r5.8xlarge'|'r5.12xlarge'|'r5.16xlarge'|'r5.24xlarge'|'r5.metal'|'r5a.large'|'r5a.xlarge'|'r5a.2xlarge'|'r5a.4xlarge'|'r5a.8xlarge'|'r5a.12xlarge'|'r5a.16xlarge'|'r5a.24xlarge'|'r5b.large'|'r5b.xlarge'|'r5b.2xlarge'|'r5b.4xlarge'|'r5b.8xlarge'|'r5b.12xlarge'|'r5b.16xlarge'|'r5b.24xlarge'|'r5b.metal'|'r5d.large'|'r5d.xlarge'|'r5d.2xlarge'|'r5d.4xlarge'|'r5d.8xlarge'|'r5d.12xlarge'|'r5d.16xlarge'|'r5d.24xlarge'|'r5d.metal'|'r5ad.large'|'r5ad.xlarge'|'r5ad.2xlarge'|'r5ad.4xlarge'|'r5ad.8xlarge'|'r5ad.12xlarge'|'r5ad.16xlarge'|'r5ad.24xlarge'|'r6g.metal'|'r6g.medium'|'r6g.large'|'r6g.xlarge'|'r6g.2xlarge'|'r6g.4xlarge'|'r6g.8xlarge'|'r6g.12xlarge'|'r6g.16xlarge'|'r6gd.metal'|'r6gd.medium'|'r6gd.large'|'r6gd.xlarge'|'r6gd.2xlarge'|'r6gd.4xlarge'|'r6gd.8xlarge'|'r6gd.12xlarge'|'r6gd.16xlarge'|'x1.16xlarge'|'x1.32xlarge'|'x1e.xlarge'|'x1e.2xlarge'|'x1e.4xlarge'|'x1e.8xlarge'|'x1e.16xlarge'|'x1e.32xlarge'|'i2.xlarge'|'i2.2xlarge'|'i2.4xlarge'|'i2.8xlarge'|'i3.large'|'i3.xlarge'|'i3.2xlarge'|'i3.4xlarge'|'i3.8xlarge'|'i3.16xlarge'|'i3.metal'|'i3en.large'|'i3en.xlarge'|'i3en.2xlarge'|'i3en.3xlarge'|'i3en.6xlarge'|'i3en.12xlarge'|'i3en.24xlarge'|'i3en.metal'|'hi1.4xlarge'|'hs1.8xlarge'|'c1.medium'|'c1.xlarge'|'c3.large'|'c3.xlarge'|'c3.2xlarge'|'c3.4xlarge'|'c3.8xlarge'|'c4.large'|'c4.xlarge'|'c4.2xlarge'|'c4.4xlarge'|'c4.8xlarge'|'c5.large'|'c5.xlarge'|'c5.2xlarge'|'c5.4xlarge'|'c5.9xlarge'|'c5.12xlarge'|'c5.18xlarge'|'c5.24xlarge'|'c5.metal'|'c5a.large'|'c5a.xlarge'|'c5a.2xlarge'|'c5a.4xlarge'|'c5a.8xlarge'|'c5a.12xlarge'|'c5a.16xlarge'|'c5a.24xlarge'|'c5ad.large'|'c5ad.xlarge'|'c5ad.2xlarge'|'c5ad.4xlarge'|'c5ad.8xlarge'|'c5ad.12xlarge'|'c5ad.16xlarge'|'c5ad.24xlarge'|'c5d.large'|'c5d.xlarge'|'c5d.2xlarge'|'c5d.4xlarge'|'c5d.9xlarge'|'c5d.12xlarge'|'c5d.18xlarge'|'c5d.24xlarge'|'c5d.metal'|'c5n.large'|'c5n.xlarge'|'c5n.2xlarge'|'c5n.4xlarge'|'c5n.9xlarge'|'c5n.18xlarge'|'c5n.metal'|'c6g.metal'|'c6g.medium'|'c6g.large'|'c6g.xlarge'|'c6g.2xlarge'|'c6g.4xlarge'|'c6g.8xlarge'|'c6g.12xlarge'|'c6g.16xlarge'|'c6gd.metal'|'c6gd.medium'|'c6gd.large'|'c6gd.xlarge'|'c6gd.2xlarge'|'c6gd.4xlarge'|'c6gd.8xlarge'|'c6gd.12xlarge'|'c6gd.16xlarge'|'c6gn.medium'|'c6gn.large'|'c6gn.xlarge'|'c6gn.2xlarge'|'c6gn.4xlarge'|'c6gn.8xlarge'|'c6gn.12xlarge'|'c6gn.16xlarge'|'cc1.4xlarge'|'cc2.8xlarge'|'g2.2xlarge'|'g2.8xlarge'|'g3.4xlarge'|'g3.8xlarge'|'g3.16xlarge'|'g3s.xlarge'|'g4ad.4xlarge'|'g4ad.8xlarge'|'g4ad.16xlarge'|'g4dn.xlarge'|'g4dn.2xlarge'|'g4dn.4xlarge'|'g4dn.8xlarge'|'g4dn.12xlarge'|'g4dn.16xlarge'|'g4dn.metal'|'cg1.4xlarge'|'p2.xlarge'|'p2.8xlarge'|'p2.16xlarge'|'p3.2xlarge'|'p3.8xlarge'|'p3.16xlarge'|'p3dn.24xlarge'|'p4d.24xlarge'|'d2.xlarge'|'d2.2xlarge'|'d2.4xlarge'|'d2.8xlarge'|'d3.xlarge'|'d3.2xlarge'|'d3.4xlarge'|'d3.8xlarge'|'d3en.xlarge'|'d3en.2xlarge'|'d3en.4xlarge'|'d3en.6xlarge'|'d3en.8xlarge'|'d3en.12xlarge'|'f1.2xlarge'|'f1.4xlarge'|'f1.16xlarge'|'m5.large'|'m5.xlarge'|'m5.2xlarge'|'m5.4xlarge'|'m5.8xlarge'|'m5.12xlarge'|'m5.16xlarge'|'m5.24xlarge'|'m5.metal'|'m5a.large'|'m5a.xlarge'|'m5a.2xlarge'|'m5a.4xlarge'|'m5a.8xlarge'|'m5a.12xlarge'|'m5a.16xlarge'|'m5a.24xlarge'|'m5d.large'|'m5d.xlarge'|'m5d.2xlarge'|'m5d.4xlarge'|'m5d.8xlarge'|'m5d.12xlarge'|'m5d.16xlarge'|'m5d.24xlarge'|'m5d.metal'|'m5ad.large'|'m5ad.xlarge'|'m5ad.2xlarge'|'m5ad.4xlarge'|'m5ad.8xlarge'|'m5ad.12xlarge'|'m5ad.16xlarge'|'m5ad.24xlarge'|'m5zn.large'|'m5zn.xlarge'|'m5zn.2xlarge'|'m5zn.3xlarge'|'m5zn.6xlarge'|'m5zn.12xlarge'|'m5zn.metal'|'h1.2xlarge'|'h1.4xlarge'|'h1.8xlarge'|'h1.16xlarge'|'z1d.large'|'z1d.xlarge'|'z1d.2xlarge'|'z1d.3xlarge'|'z1d.6xlarge'|'z1d.12xlarge'|'z1d.metal'|'u-6tb1.metal'|'u-9tb1.metal'|'u-12tb1.metal'|'u-18tb1.metal'|'u-24tb1.metal'|'a1.medium'|'a1.large'|'a1.xlarge'|'a1.2xlarge'|'a1.4xlarge'|'a1.metal'|'m5dn.large'|'m5dn.xlarge'|'m5dn.2xlarge'|'m5dn.4xlarge'|'m5dn.8xlarge'|'m5dn.12xlarge'|'m5dn.16xlarge'|'m5dn.24xlarge'|'m5n.large'|'m5n.xlarge'|'m5n.2xlarge'|'m5n.4xlarge'|'m5n.8xlarge'|'m5n.12xlarge'|'m5n.16xlarge'|'m5n.24xlarge'|'r5dn.large'|'r5dn.xlarge'|'r5dn.2xlarge'|'r5dn.4xlarge'|'r5dn.8xlarge'|'r5dn.12xlarge'|'r5dn.16xlarge'|'r5dn.24xlarge'|'r5n.large'|'r5n.xlarge'|'r5n.2xlarge'|'r5n.4xlarge'|'r5n.8xlarge'|'r5n.12xlarge'|'r5n.16xlarge'|'r5n.24xlarge'|'inf1.xlarge'|'inf1.2xlarge'|'inf1.6xlarge'|'inf1.24xlarge'|'m6g.metal'|'m6g.medium'|'m6g.large'|'m6g.xlarge'|'m6g.2xlarge'|'m6g.4xlarge'|'m6g.8xlarge'|'m6g.12xlarge'|'m6g.16xlarge'|'m6gd.metal'|'m6gd.medium'|'m6gd.large'|'m6gd.xlarge'|'m6gd.2xlarge'|'m6gd.4xlarge'|'m6gd.8xlarge'|'m6gd.12xlarge'|'m6gd.16xlarge'|'mac1.metal',
                'KernelId': 'string',
                'KeyName': 'string',
                'NetworkInterfaces': [
                    {
                        'AssociatePublicIpAddress': True|False,
                        'DeleteOnTermination': True|False,
                        'Description': 'string',
                        'DeviceIndex': 123,
                        'Groups': [
                            'string',
                        ],
                        'Ipv6AddressCount': 123,
                        'Ipv6Addresses': [
                            {
                                'Ipv6Address': 'string'
                            },
                        ],
                        'NetworkInterfaceId': 'string',
                        'PrivateIpAddress': 'string',
                        'PrivateIpAddresses': [
                            {
                                'Primary': True|False,
                                'PrivateIpAddress': 'string'
                            },
                        ],
                        'SecondaryPrivateIpAddressCount': 123,
                        'SubnetId': 'string',
                        'AssociateCarrierIpAddress': True|False,
                        'InterfaceType': 'string',
                        'NetworkCardIndex': 123
                    },
                ],
                'Placement': {
                    'AvailabilityZone': 'string',
                    'GroupName': 'string',
                    'Tenancy': 'default'|'dedicated'|'host'
                },
                'RamdiskId': 'string',
                'SubnetId': 'string',
                'Monitoring': {
                    'Enabled': True|False
                }
            },
            'LaunchedAvailabilityZone': 'string',
            'ProductDescription': 'Linux/UNIX'|'Linux/UNIX (Amazon VPC)'|'Windows'|'Windows (Amazon VPC)',
            'SpotInstanceRequestId': 'string',
            'SpotPrice': 'string',
            'State': 'open'|'active'|'closed'|'cancelled'|'failed',
            'Status': {
                'Code': 'string',
                'Message': 'string',
                'UpdateTime': datetime(2015, 1, 1)
            },
            'Tags': [
                {
                    'Key': 'string',
                    'Value': 'string'
                },
            ],
            'Type': 'one-time'|'persistent',
            'ValidFrom': datetime(2015, 1, 1),
            'ValidUntil': datetime(2015, 1, 1),
            'InstanceInterruptionBehavior': 'hibernate'|'stop'|'terminate'
        },
    ],

}

**Response Structure**

- _(dict) --_

    Contains the output of DescribeSpotInstanceRequests.

    - **SpotInstanceRequests** _(list) --_

        One or more Spot Instance requests.

        - _(dict) --_

            Describes a Spot Instance request.

            - **ActualBlockHourlyPrice** _(string) --_

                If you specified a duration and your Spot Instance request was fulfilled, this is the fixed hourly price in effect for the Spot Instance while it runs.

            - **AvailabilityZoneGroup** _(string) --_

                The Availability Zone group. If you specify the same Availability Zone group for all Spot Instance requests, all Spot Instances are launched in the same Availability Zone.

            - **BlockDurationMinutes** _(integer) --_

                The duration for the Spot Instance, in minutes.

            - **CreateTime** _(datetime) --_

                The date and time when the Spot Instance request was created, in UTC format (for example, _YYYY_ -_MM_ -_DD_ T*HH* :_MM_ :_SS_ Z).

            - **Fault** _(dict) --_

                The fault codes for the Spot Instance request, if any.

                - **Code** _(string) --_

                    The reason code for the Spot Instance state change.

                - **Message** _(string) --_

                    The message for the Spot Instance state change.

            - **InstanceId** _(string) --_

                The instance ID, if an instance has been launched to fulfill the Spot Instance request.

            - **LaunchGroup** _(string) --_

                The instance launch group. Launch groups are Spot Instances that launch together and terminate together.

            - **LaunchSpecification** _(dict) --_

                Additional information for launching instances.

                - **UserData** _(string) --_

                    The Base64-encoded user data for the instance.

                - **SecurityGroups** _(list) --_

                    One or more security groups. When requesting instances in a VPC, you must specify the IDs of the security groups. When requesting instances in EC2-Classic, you can specify the names or the IDs of the security groups.

                    - _(dict) --_

                        Describes a security group.

                        - **GroupName** _(string) --_

                            The name of the security group.

                        - **GroupId** _(string) --_

                            The ID of the security group.

                - **AddressingType** _(string) --_

                    Deprecated.

                - **BlockDeviceMappings** _(list) --_

                    One or more block device mapping entries.

                    - _(dict) --_

                        Describes a block device mapping.

                        - **DeviceName** _(string) --_

                            The device name (for example, /dev/sdh or xvdh ).

                        - **VirtualName** _(string) --_

                            The virtual device name (ephemeral N). Instance store volumes are numbered starting from 0. An instance type with 2 available instance store volumes can specify mappings for ephemeral0 and ephemeral1 . The number of available instance store volumes depends on the instance type. After you connect to the instance, you must mount the volume.

                            NVMe instance store volumes are automatically enumerated and assigned a device name. Including them in your block device mapping has no effect.

                            Constraints: For M3 instances, you must specify instance store volumes in the block device mapping for the instance. When you launch an M3 instance, we ignore any instance store volumes specified in the block device mapping for the AMI.

                        - **Ebs** _(dict) --_

                            Parameters used to automatically set up EBS volumes when the instance is launched.

                            - **DeleteOnTermination** _(boolean) --_

                                Indicates whether the EBS volume is deleted on instance termination. For more information, see [Preserving Amazon EBS volumes on instance termination](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/terminating-instances.html#preserving-volumes-on-termination) in the Amazon Elastic Compute Cloud User Guide.

                            - **Iops** _(integer) --_

                                The number of I/O operations per second (IOPS). For gp3 , io1 , and io2 volumes, this represents the number of IOPS that are provisioned for the volume. For gp2 volumes, this represents the baseline performance of the volume and the rate at which the volume accumulates I/O credits for bursting.

                                The following are the supported values for each volume type:

                                - gp3 : 3,000-16,000 IOPS
                                - io1 : 100-64,000 IOPS
                                - io2 : 100-64,000 IOPS

                                For io1 and io2 volumes, we guarantee 64,000 IOPS only for [Instances built on the Nitro System](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/instance-types.html#ec2-nitro-instances) . Other instance families guarantee performance up to 32,000 IOPS.

                                This parameter is required for io1 and io2 volumes. The default for gp3 volumes is 3,000 IOPS. This parameter is not supported for gp2 , st1 , sc1 , or standard volumes.

                            - **SnapshotId** _(string) --_

                                The ID of the snapshot.

                            - **VolumeSize** _(integer) --_

                                The size of the volume, in GiBs. You must specify either a snapshot ID or a volume size. If you specify a snapshot, the default is the snapshot size. You can specify a volume size that is equal to or larger than the snapshot size.

                                The following are the supported volumes sizes for each volume type:

                                - gp2 and gp3 :1-16,384
                                - io1 and io2 : 4-16,384
                                - st1 : 500-16,384
                                - sc1 : 500-16,384
                                - standard : 1-1,024
                            - **VolumeType** _(string) --_

                                The volume type. For more information, see [Amazon EBS volume types](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EBSVolumeTypes.html) in the _Amazon Elastic Compute Cloud User Guide_ . If the volume type is io1 or io2 , you must specify the IOPS that the volume supports.

                            - **KmsKeyId** _(string) --_

                                Identifier (key ID, key alias, ID ARN, or alias ARN) for a customer managed CMK under which the EBS volume is encrypted.

                                This parameter is only supported on BlockDeviceMapping objects called by [RunInstances](https://docs.aws.amazon.com/AWSEC2/latest/APIReference/API_RunInstances.html) , [RequestSpotFleet](https://docs.aws.amazon.com/AWSEC2/latest/APIReference/API_RequestSpotFleet.html) , and [RequestSpotInstances](https://docs.aws.amazon.com/AWSEC2/latest/APIReference/API_RequestSpotInstances.html) .

                            - **Throughput** _(integer) --_

                                The throughput that the volume supports, in MiB/s.

                                This parameter is valid only for gp3 volumes.

                                Valid Range: Minimum value of 125. Maximum value of 1000.

                            - **Encrypted** _(boolean) --_

                                Indicates whether the encryption state of an EBS volume is changed while being restored from a backing snapshot. The effect of setting the encryption state to true depends on the volume origin (new or from a snapshot), starting encryption state, ownership, and whether encryption by default is enabled. For more information, see [Amazon EBS Encryption](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EBSEncryption.html#encryption-parameters) in the _Amazon Elastic Compute Cloud User Guide_ .

                                In no case can you remove encryption from an encrypted volume.

                                Encrypted volumes can only be attached to instances that support Amazon EBS encryption. For more information, see [Supported instance types](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EBSEncryption.html#EBSEncryption_supported_instances) .

                                This parameter is not returned by .

                        - **NoDevice** _(string) --_

                            Suppresses the specified device included in the block device mapping of the AMI.

                - **EbsOptimized** _(boolean) --_

                    Indicates whether the instance is optimized for EBS I/O. This optimization provides dedicated throughput to Amazon EBS and an optimized configuration stack to provide optimal EBS I/O performance. This optimization isn't available with all instance types. Additional usage charges apply when using an EBS Optimized instance.

                    Default: false

                - **IamInstanceProfile** _(dict) --_

                    The IAM instance profile.

                    - **Arn** _(string) --_

                        The Amazon Resource Name (ARN) of the instance profile.

                    - **Name** _(string) --_

                        The name of the instance profile.

                - **ImageId** _(string) --_

                    The ID of the AMI.

                - **InstanceType** _(string) --_

                    The instance type.

                - **KernelId** _(string) --_

                    The ID of the kernel.

                - **KeyName** _(string) --_

                    The name of the key pair.

                - **NetworkInterfaces** _(list) --_

                    One or more network interfaces. If you specify a network interface, you must specify subnet IDs and security group IDs using the network interface.

                    - _(dict) --_

                        Describes a network interface.

                        - **AssociatePublicIpAddress** _(boolean) --_

                            Indicates whether to assign a public IPv4 address to an instance you launch in a VPC. The public IP address can only be assigned to a network interface for eth0, and can only be assigned to a new network interface, not an existing one. You cannot specify more than one network interface in the request. If launching into a default subnet, the default value is true .

                        - **DeleteOnTermination** _(boolean) --_

                            If set to true , the interface is deleted when the instance is terminated. You can specify true only if creating a new network interface when launching an instance.

                        - **Description** _(string) --_

                            The description of the network interface. Applies only if creating a network interface when launching an instance.

                        - **DeviceIndex** _(integer) --_

                            The position of the network interface in the attachment order. A primary network interface has a device index of 0.

                            If you specify a network interface when launching an instance, you must specify the device index.

                        - **Groups** _(list) --_

                            The IDs of the security groups for the network interface. Applies only if creating a network interface when launching an instance.

                            - _(string) --_
                        - **Ipv6AddressCount** _(integer) --_

                            A number of IPv6 addresses to assign to the network interface. Amazon EC2 chooses the IPv6 addresses from the range of the subnet. You cannot specify this option and the option to assign specific IPv6 addresses in the same request. You can specify this option if you've specified a minimum number of instances to launch.

                        - **Ipv6Addresses** _(list) --_

                            One or more IPv6 addresses to assign to the network interface. You cannot specify this option and the option to assign a number of IPv6 addresses in the same request. You cannot specify this option if you've specified a minimum number of instances to launch.

                            - _(dict) --_

                                Describes an IPv6 address.

                                - **Ipv6Address** _(string) --_

                                    The IPv6 address.

                        - **NetworkInterfaceId** _(string) --_

                            The ID of the network interface.

                            If you are creating a Spot Fleet, omit this parameter because you can’t specify a network interface ID in a launch specification.

                        - **PrivateIpAddress** _(string) --_

                            The private IPv4 address of the network interface. Applies only if creating a network interface when launching an instance. You cannot specify this option if you're launching more than one instance in a [RunInstances](https://docs.aws.amazon.com/AWSEC2/latest/APIReference/API_RunInstances.html) request.

                        - **PrivateIpAddresses** _(list) --_

                            One or more private IPv4 addresses to assign to the network interface. Only one private IPv4 address can be designated as primary. You cannot specify this option if you're launching more than one instance in a [RunInstances](https://docs.aws.amazon.com/AWSEC2/latest/APIReference/API_RunInstances.html) request.

                            - _(dict) --_

                                Describes a secondary private IPv4 address for a network interface.

                                - **Primary** _(boolean) --_

                                    Indicates whether the private IPv4 address is the primary private IPv4 address. Only one IPv4 address can be designated as primary.

                                - **PrivateIpAddress** _(string) --_

                                    The private IPv4 addresses.

                        - **SecondaryPrivateIpAddressCount** _(integer) --_

                            The number of secondary private IPv4 addresses. You can't specify this option and specify more than one private IP address using the private IP addresses option. You cannot specify this option if you're launching more than one instance in a [RunInstances](https://docs.aws.amazon.com/AWSEC2/latest/APIReference/API_RunInstances.html) request.

                        - **SubnetId** _(string) --_

                            The ID of the subnet associated with the network interface. Applies only if creating a network interface when launching an instance.

                        - **AssociateCarrierIpAddress** _(boolean) --_

                            Indicates whether to assign a carrier IP address to the network interface.

                            You can only assign a carrier IP address to a network interface that is in a subnet in a Wavelength Zone. For more information about carrier IP addresses, see Carrier IP addresses in the AWS Wavelength Developer Guide.

                        - **InterfaceType** _(string) --_

                            The type of network interface.

                            To create an Elastic Fabric Adapter (EFA), specify efa . For more information, see [Elastic Fabric Adapter](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html) in the _Amazon Elastic Compute Cloud User Guide_ .

                            If you are not creating an EFA, specify interface or omit this parameter.

                            Valid values: interface | efa

                        - **NetworkCardIndex** _(integer) --_

                            The index of the network card. Some instance types support multiple network cards. The primary network interface must be assigned to network card index 0. The default is network card index 0.

                - **Placement** _(dict) --_

                    The placement information for the instance.

                    - **AvailabilityZone** _(string) --_

                        The Availability Zone.

                        [Spot Fleet only] To specify multiple Availability Zones, separate them using commas; for example, "us-west-2a, us-west-2b".

                    - **GroupName** _(string) --_

                        The name of the placement group.

                    - **Tenancy** _(string) --_

                        The tenancy of the instance (if the instance is running in a VPC). An instance with a tenancy of dedicated runs on single-tenant hardware. The host tenancy is not supported for Spot Instances.

                - **RamdiskId** _(string) --_

                    The ID of the RAM disk.

                - **SubnetId** _(string) --_

                    The ID of the subnet in which to launch the instance.

                - **Monitoring** _(dict) --_

                    Describes the monitoring of an instance.

                    - **Enabled** _(boolean) --_

                        Indicates whether detailed monitoring is enabled. Otherwise, basic monitoring is enabled.

            - **LaunchedAvailabilityZone** _(string) --_

                The Availability Zone in which the request is launched.

            - **ProductDescription** _(string) --_

                The product description associated with the Spot Instance.

            - **SpotInstanceRequestId** _(string) --_

                The ID of the Spot Instance request.

            - **SpotPrice** _(string) --_

                The maximum price per hour that you are willing to pay for a Spot Instance.

            - **State** _(string) --_

                The state of the Spot Instance request. Spot status information helps track your Spot Instance requests. For more information, see [Spot status](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/spot-bid-status.html) in the _Amazon EC2 User Guide for Linux Instances_ .

            - **Status** _(dict) --_

                The status code and status message describing the Spot Instance request.

                - **Code** _(string) --_

                    The status code. For a list of status codes, see [Spot status codes](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/spot-bid-status.html#spot-instance-bid-status-understand) in the _Amazon EC2 User Guide for Linux Instances_ .

                - **Message** _(string) --_

                    The description for the status code.

                - **UpdateTime** _(datetime) --_

                    The date and time of the most recent status update, in UTC format (for example, _YYYY_ -_MM_ -_DD_ T*HH* :_MM_ :_SS_ Z).

            - **Tags** _(list) --_

                Any tags assigned to the resource.

                - _(dict) --_

                    Describes a tag.

                    - **Key** _(string) --_

                        The key of the tag.

                        Constraints: Tag keys are case-sensitive and accept a maximum of 127 Unicode characters. May not begin with aws: .

                    - **Value** _(string) --_

                        The value of the tag.

                        Constraints: Tag values are case-sensitive and accept a maximum of 255 Unicode characters.

            - **Type** _(string) --_

                The Spot Instance request type.

            - **ValidFrom** _(datetime) --_

                The start date of the request, in UTC format (for example, _YYYY_ -_MM_ -_DD_ T*HH* :_MM_ :_SS_ Z). The request becomes active at this date and time.

            - **ValidUntil** _(datetime) --_

                The end date of the request, in UTC format (_YYYY_ -_MM_ -_DD_ T*HH* :_MM_ :_SS_ Z).

                - For a persistent request, the request remains active until the validUntil date and time is reached. Otherwise, the request remains active until you cancel it.
                - For a one-time request, the request remains active until all instances launch, the request is canceled, or the validUntil date and time is reached. By default, the request is valid for 7 days from the date the request was created.
            - **InstanceInterruptionBehavior** _(string) --_

                The behavior when a Spot Instance is interrupted.


_class_ EC2.Paginator.DescribeSpotPriceHistory

paginator = client.get_paginator('describe_spot_price_history')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_spot_price_history()](#EC2.Client.describe_spot_price_history "EC2.Client.describe_spot_price_history").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeSpotPriceHistory)

**Request Syntax**

response_iterator = paginator.paginate(
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    AvailabilityZone='string',
    DryRun=True|False,
    EndTime=datetime(2015, 1, 1),
    InstanceTypes=[
        't1.micro'|'t2.nano'|'t2.micro'|'t2.small'|'t2.medium'|'t2.large'|'t2.xlarge'|'t2.2xlarge'|'t3.nano'|'t3.micro'|'t3.small'|'t3.medium'|'t3.large'|'t3.xlarge'|'t3.2xlarge'|'t3a.nano'|'t3a.micro'|'t3a.small'|'t3a.medium'|'t3a.large'|'t3a.xlarge'|'t3a.2xlarge'|'t4g.nano'|'t4g.micro'|'t4g.small'|'t4g.medium'|'t4g.large'|'t4g.xlarge'|'t4g.2xlarge'|'m1.small'|'m1.medium'|'m1.large'|'m1.xlarge'|'m3.medium'|'m3.large'|'m3.xlarge'|'m3.2xlarge'|'m4.large'|'m4.xlarge'|'m4.2xlarge'|'m4.4xlarge'|'m4.10xlarge'|'m4.16xlarge'|'m2.xlarge'|'m2.2xlarge'|'m2.4xlarge'|'cr1.8xlarge'|'r3.large'|'r3.xlarge'|'r3.2xlarge'|'r3.4xlarge'|'r3.8xlarge'|'r4.large'|'r4.xlarge'|'r4.2xlarge'|'r4.4xlarge'|'r4.8xlarge'|'r4.16xlarge'|'r5.large'|'r5.xlarge'|'r5.2xlarge'|'r5.4xlarge'|'r5.8xlarge'|'r5.12xlarge'|'r5.16xlarge'|'r5.24xlarge'|'r5.metal'|'r5a.large'|'r5a.xlarge'|'r5a.2xlarge'|'r5a.4xlarge'|'r5a.8xlarge'|'r5a.12xlarge'|'r5a.16xlarge'|'r5a.24xlarge'|'r5b.large'|'r5b.xlarge'|'r5b.2xlarge'|'r5b.4xlarge'|'r5b.8xlarge'|'r5b.12xlarge'|'r5b.16xlarge'|'r5b.24xlarge'|'r5b.metal'|'r5d.large'|'r5d.xlarge'|'r5d.2xlarge'|'r5d.4xlarge'|'r5d.8xlarge'|'r5d.12xlarge'|'r5d.16xlarge'|'r5d.24xlarge'|'r5d.metal'|'r5ad.large'|'r5ad.xlarge'|'r5ad.2xlarge'|'r5ad.4xlarge'|'r5ad.8xlarge'|'r5ad.12xlarge'|'r5ad.16xlarge'|'r5ad.24xlarge'|'r6g.metal'|'r6g.medium'|'r6g.large'|'r6g.xlarge'|'r6g.2xlarge'|'r6g.4xlarge'|'r6g.8xlarge'|'r6g.12xlarge'|'r6g.16xlarge'|'r6gd.metal'|'r6gd.medium'|'r6gd.large'|'r6gd.xlarge'|'r6gd.2xlarge'|'r6gd.4xlarge'|'r6gd.8xlarge'|'r6gd.12xlarge'|'r6gd.16xlarge'|'x1.16xlarge'|'x1.32xlarge'|'x1e.xlarge'|'x1e.2xlarge'|'x1e.4xlarge'|'x1e.8xlarge'|'x1e.16xlarge'|'x1e.32xlarge'|'i2.xlarge'|'i2.2xlarge'|'i2.4xlarge'|'i2.8xlarge'|'i3.large'|'i3.xlarge'|'i3.2xlarge'|'i3.4xlarge'|'i3.8xlarge'|'i3.16xlarge'|'i3.metal'|'i3en.large'|'i3en.xlarge'|'i3en.2xlarge'|'i3en.3xlarge'|'i3en.6xlarge'|'i3en.12xlarge'|'i3en.24xlarge'|'i3en.metal'|'hi1.4xlarge'|'hs1.8xlarge'|'c1.medium'|'c1.xlarge'|'c3.large'|'c3.xlarge'|'c3.2xlarge'|'c3.4xlarge'|'c3.8xlarge'|'c4.large'|'c4.xlarge'|'c4.2xlarge'|'c4.4xlarge'|'c4.8xlarge'|'c5.large'|'c5.xlarge'|'c5.2xlarge'|'c5.4xlarge'|'c5.9xlarge'|'c5.12xlarge'|'c5.18xlarge'|'c5.24xlarge'|'c5.metal'|'c5a.large'|'c5a.xlarge'|'c5a.2xlarge'|'c5a.4xlarge'|'c5a.8xlarge'|'c5a.12xlarge'|'c5a.16xlarge'|'c5a.24xlarge'|'c5ad.large'|'c5ad.xlarge'|'c5ad.2xlarge'|'c5ad.4xlarge'|'c5ad.8xlarge'|'c5ad.12xlarge'|'c5ad.16xlarge'|'c5ad.24xlarge'|'c5d.large'|'c5d.xlarge'|'c5d.2xlarge'|'c5d.4xlarge'|'c5d.9xlarge'|'c5d.12xlarge'|'c5d.18xlarge'|'c5d.24xlarge'|'c5d.metal'|'c5n.large'|'c5n.xlarge'|'c5n.2xlarge'|'c5n.4xlarge'|'c5n.9xlarge'|'c5n.18xlarge'|'c5n.metal'|'c6g.metal'|'c6g.medium'|'c6g.large'|'c6g.xlarge'|'c6g.2xlarge'|'c6g.4xlarge'|'c6g.8xlarge'|'c6g.12xlarge'|'c6g.16xlarge'|'c6gd.metal'|'c6gd.medium'|'c6gd.large'|'c6gd.xlarge'|'c6gd.2xlarge'|'c6gd.4xlarge'|'c6gd.8xlarge'|'c6gd.12xlarge'|'c6gd.16xlarge'|'c6gn.medium'|'c6gn.large'|'c6gn.xlarge'|'c6gn.2xlarge'|'c6gn.4xlarge'|'c6gn.8xlarge'|'c6gn.12xlarge'|'c6gn.16xlarge'|'cc1.4xlarge'|'cc2.8xlarge'|'g2.2xlarge'|'g2.8xlarge'|'g3.4xlarge'|'g3.8xlarge'|'g3.16xlarge'|'g3s.xlarge'|'g4ad.4xlarge'|'g4ad.8xlarge'|'g4ad.16xlarge'|'g4dn.xlarge'|'g4dn.2xlarge'|'g4dn.4xlarge'|'g4dn.8xlarge'|'g4dn.12xlarge'|'g4dn.16xlarge'|'g4dn.metal'|'cg1.4xlarge'|'p2.xlarge'|'p2.8xlarge'|'p2.16xlarge'|'p3.2xlarge'|'p3.8xlarge'|'p3.16xlarge'|'p3dn.24xlarge'|'p4d.24xlarge'|'d2.xlarge'|'d2.2xlarge'|'d2.4xlarge'|'d2.8xlarge'|'d3.xlarge'|'d3.2xlarge'|'d3.4xlarge'|'d3.8xlarge'|'d3en.xlarge'|'d3en.2xlarge'|'d3en.4xlarge'|'d3en.6xlarge'|'d3en.8xlarge'|'d3en.12xlarge'|'f1.2xlarge'|'f1.4xlarge'|'f1.16xlarge'|'m5.large'|'m5.xlarge'|'m5.2xlarge'|'m5.4xlarge'|'m5.8xlarge'|'m5.12xlarge'|'m5.16xlarge'|'m5.24xlarge'|'m5.metal'|'m5a.large'|'m5a.xlarge'|'m5a.2xlarge'|'m5a.4xlarge'|'m5a.8xlarge'|'m5a.12xlarge'|'m5a.16xlarge'|'m5a.24xlarge'|'m5d.large'|'m5d.xlarge'|'m5d.2xlarge'|'m5d.4xlarge'|'m5d.8xlarge'|'m5d.12xlarge'|'m5d.16xlarge'|'m5d.24xlarge'|'m5d.metal'|'m5ad.large'|'m5ad.xlarge'|'m5ad.2xlarge'|'m5ad.4xlarge'|'m5ad.8xlarge'|'m5ad.12xlarge'|'m5ad.16xlarge'|'m5ad.24xlarge'|'m5zn.large'|'m5zn.xlarge'|'m5zn.2xlarge'|'m5zn.3xlarge'|'m5zn.6xlarge'|'m5zn.12xlarge'|'m5zn.metal'|'h1.2xlarge'|'h1.4xlarge'|'h1.8xlarge'|'h1.16xlarge'|'z1d.large'|'z1d.xlarge'|'z1d.2xlarge'|'z1d.3xlarge'|'z1d.6xlarge'|'z1d.12xlarge'|'z1d.metal'|'u-6tb1.metal'|'u-9tb1.metal'|'u-12tb1.metal'|'u-18tb1.metal'|'u-24tb1.metal'|'a1.medium'|'a1.large'|'a1.xlarge'|'a1.2xlarge'|'a1.4xlarge'|'a1.metal'|'m5dn.large'|'m5dn.xlarge'|'m5dn.2xlarge'|'m5dn.4xlarge'|'m5dn.8xlarge'|'m5dn.12xlarge'|'m5dn.16xlarge'|'m5dn.24xlarge'|'m5n.large'|'m5n.xlarge'|'m5n.2xlarge'|'m5n.4xlarge'|'m5n.8xlarge'|'m5n.12xlarge'|'m5n.16xlarge'|'m5n.24xlarge'|'r5dn.large'|'r5dn.xlarge'|'r5dn.2xlarge'|'r5dn.4xlarge'|'r5dn.8xlarge'|'r5dn.12xlarge'|'r5dn.16xlarge'|'r5dn.24xlarge'|'r5n.large'|'r5n.xlarge'|'r5n.2xlarge'|'r5n.4xlarge'|'r5n.8xlarge'|'r5n.12xlarge'|'r5n.16xlarge'|'r5n.24xlarge'|'inf1.xlarge'|'inf1.2xlarge'|'inf1.6xlarge'|'inf1.24xlarge'|'m6g.metal'|'m6g.medium'|'m6g.large'|'m6g.xlarge'|'m6g.2xlarge'|'m6g.4xlarge'|'m6g.8xlarge'|'m6g.12xlarge'|'m6g.16xlarge'|'m6gd.metal'|'m6gd.medium'|'m6gd.large'|'m6gd.xlarge'|'m6gd.2xlarge'|'m6gd.4xlarge'|'m6gd.8xlarge'|'m6gd.12xlarge'|'m6gd.16xlarge'|'mac1.metal',
    ],
    ProductDescriptions=[
        'string',
    ],
    StartTime=datetime(2015, 1, 1),
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **Filters** (_list_) --

    One or more filters.

    - availability-zone - The Availability Zone for which prices should be returned.
    - instance-type - The type of instance (for example, m3.medium ).
    - product-description - The product description for the Spot price (Linux/UNIX | Red Hat Enterprise Linux | SUSE Linux | Windows | Linux/UNIX (Amazon VPC) | Red Hat Enterprise Linux (Amazon VPC) | SUSE Linux (Amazon VPC) | Windows (Amazon VPC) ).
    - spot-price - The Spot price. The value must match exactly (or use wildcards; greater than or less than comparison is not supported).
    - timestamp - The time stamp of the Spot price history, in UTC format (for example, _YYYY_ -_MM_ -_DD_ T*HH* :_MM_ :_SS_ Z). You can use wildcards (* and ?). Greater than or less than comparison is not supported.

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **AvailabilityZone** (_string_) -- Filters the results by the specified Availability Zone.
- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **EndTime** (_datetime_) -- The date and time, up to the current date, from which to stop retrieving the price history data, in UTC format (for example, _YYYY_ -_MM_ -_DD_ T*HH* :_MM_ :_SS_ Z).
- **InstanceTypes** (_list_) --

    Filters the results by the specified instance types.

    - _(string) --_
- **ProductDescriptions** (_list_) --

    Filters the results by the specified basic product descriptions.

    - _(string) --_
- **StartTime** (_datetime_) -- The date and time, up to the past 90 days, from which to start retrieving the price history data, in UTC format (for example, _YYYY_ -_MM_ -_DD_ T*HH* :_MM_ :_SS_ Z).
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'SpotPriceHistory': [
        {
            'AvailabilityZone': 'string',
            'InstanceType': 't1.micro'|'t2.nano'|'t2.micro'|'t2.small'|'t2.medium'|'t2.large'|'t2.xlarge'|'t2.2xlarge'|'t3.nano'|'t3.micro'|'t3.small'|'t3.medium'|'t3.large'|'t3.xlarge'|'t3.2xlarge'|'t3a.nano'|'t3a.micro'|'t3a.small'|'t3a.medium'|'t3a.large'|'t3a.xlarge'|'t3a.2xlarge'|'t4g.nano'|'t4g.micro'|'t4g.small'|'t4g.medium'|'t4g.large'|'t4g.xlarge'|'t4g.2xlarge'|'m1.small'|'m1.medium'|'m1.large'|'m1.xlarge'|'m3.medium'|'m3.large'|'m3.xlarge'|'m3.2xlarge'|'m4.large'|'m4.xlarge'|'m4.2xlarge'|'m4.4xlarge'|'m4.10xlarge'|'m4.16xlarge'|'m2.xlarge'|'m2.2xlarge'|'m2.4xlarge'|'cr1.8xlarge'|'r3.large'|'r3.xlarge'|'r3.2xlarge'|'r3.4xlarge'|'r3.8xlarge'|'r4.large'|'r4.xlarge'|'r4.2xlarge'|'r4.4xlarge'|'r4.8xlarge'|'r4.16xlarge'|'r5.large'|'r5.xlarge'|'r5.2xlarge'|'r5.4xlarge'|'r5.8xlarge'|'r5.12xlarge'|'r5.16xlarge'|'r5.24xlarge'|'r5.metal'|'r5a.large'|'r5a.xlarge'|'r5a.2xlarge'|'r5a.4xlarge'|'r5a.8xlarge'|'r5a.12xlarge'|'r5a.16xlarge'|'r5a.24xlarge'|'r5b.large'|'r5b.xlarge'|'r5b.2xlarge'|'r5b.4xlarge'|'r5b.8xlarge'|'r5b.12xlarge'|'r5b.16xlarge'|'r5b.24xlarge'|'r5b.metal'|'r5d.large'|'r5d.xlarge'|'r5d.2xlarge'|'r5d.4xlarge'|'r5d.8xlarge'|'r5d.12xlarge'|'r5d.16xlarge'|'r5d.24xlarge'|'r5d.metal'|'r5ad.large'|'r5ad.xlarge'|'r5ad.2xlarge'|'r5ad.4xlarge'|'r5ad.8xlarge'|'r5ad.12xlarge'|'r5ad.16xlarge'|'r5ad.24xlarge'|'r6g.metal'|'r6g.medium'|'r6g.large'|'r6g.xlarge'|'r6g.2xlarge'|'r6g.4xlarge'|'r6g.8xlarge'|'r6g.12xlarge'|'r6g.16xlarge'|'r6gd.metal'|'r6gd.medium'|'r6gd.large'|'r6gd.xlarge'|'r6gd.2xlarge'|'r6gd.4xlarge'|'r6gd.8xlarge'|'r6gd.12xlarge'|'r6gd.16xlarge'|'x1.16xlarge'|'x1.32xlarge'|'x1e.xlarge'|'x1e.2xlarge'|'x1e.4xlarge'|'x1e.8xlarge'|'x1e.16xlarge'|'x1e.32xlarge'|'i2.xlarge'|'i2.2xlarge'|'i2.4xlarge'|'i2.8xlarge'|'i3.large'|'i3.xlarge'|'i3.2xlarge'|'i3.4xlarge'|'i3.8xlarge'|'i3.16xlarge'|'i3.metal'|'i3en.large'|'i3en.xlarge'|'i3en.2xlarge'|'i3en.3xlarge'|'i3en.6xlarge'|'i3en.12xlarge'|'i3en.24xlarge'|'i3en.metal'|'hi1.4xlarge'|'hs1.8xlarge'|'c1.medium'|'c1.xlarge'|'c3.large'|'c3.xlarge'|'c3.2xlarge'|'c3.4xlarge'|'c3.8xlarge'|'c4.large'|'c4.xlarge'|'c4.2xlarge'|'c4.4xlarge'|'c4.8xlarge'|'c5.large'|'c5.xlarge'|'c5.2xlarge'|'c5.4xlarge'|'c5.9xlarge'|'c5.12xlarge'|'c5.18xlarge'|'c5.24xlarge'|'c5.metal'|'c5a.large'|'c5a.xlarge'|'c5a.2xlarge'|'c5a.4xlarge'|'c5a.8xlarge'|'c5a.12xlarge'|'c5a.16xlarge'|'c5a.24xlarge'|'c5ad.large'|'c5ad.xlarge'|'c5ad.2xlarge'|'c5ad.4xlarge'|'c5ad.8xlarge'|'c5ad.12xlarge'|'c5ad.16xlarge'|'c5ad.24xlarge'|'c5d.large'|'c5d.xlarge'|'c5d.2xlarge'|'c5d.4xlarge'|'c5d.9xlarge'|'c5d.12xlarge'|'c5d.18xlarge'|'c5d.24xlarge'|'c5d.metal'|'c5n.large'|'c5n.xlarge'|'c5n.2xlarge'|'c5n.4xlarge'|'c5n.9xlarge'|'c5n.18xlarge'|'c5n.metal'|'c6g.metal'|'c6g.medium'|'c6g.large'|'c6g.xlarge'|'c6g.2xlarge'|'c6g.4xlarge'|'c6g.8xlarge'|'c6g.12xlarge'|'c6g.16xlarge'|'c6gd.metal'|'c6gd.medium'|'c6gd.large'|'c6gd.xlarge'|'c6gd.2xlarge'|'c6gd.4xlarge'|'c6gd.8xlarge'|'c6gd.12xlarge'|'c6gd.16xlarge'|'c6gn.medium'|'c6gn.large'|'c6gn.xlarge'|'c6gn.2xlarge'|'c6gn.4xlarge'|'c6gn.8xlarge'|'c6gn.12xlarge'|'c6gn.16xlarge'|'cc1.4xlarge'|'cc2.8xlarge'|'g2.2xlarge'|'g2.8xlarge'|'g3.4xlarge'|'g3.8xlarge'|'g3.16xlarge'|'g3s.xlarge'|'g4ad.4xlarge'|'g4ad.8xlarge'|'g4ad.16xlarge'|'g4dn.xlarge'|'g4dn.2xlarge'|'g4dn.4xlarge'|'g4dn.8xlarge'|'g4dn.12xlarge'|'g4dn.16xlarge'|'g4dn.metal'|'cg1.4xlarge'|'p2.xlarge'|'p2.8xlarge'|'p2.16xlarge'|'p3.2xlarge'|'p3.8xlarge'|'p3.16xlarge'|'p3dn.24xlarge'|'p4d.24xlarge'|'d2.xlarge'|'d2.2xlarge'|'d2.4xlarge'|'d2.8xlarge'|'d3.xlarge'|'d3.2xlarge'|'d3.4xlarge'|'d3.8xlarge'|'d3en.xlarge'|'d3en.2xlarge'|'d3en.4xlarge'|'d3en.6xlarge'|'d3en.8xlarge'|'d3en.12xlarge'|'f1.2xlarge'|'f1.4xlarge'|'f1.16xlarge'|'m5.large'|'m5.xlarge'|'m5.2xlarge'|'m5.4xlarge'|'m5.8xlarge'|'m5.12xlarge'|'m5.16xlarge'|'m5.24xlarge'|'m5.metal'|'m5a.large'|'m5a.xlarge'|'m5a.2xlarge'|'m5a.4xlarge'|'m5a.8xlarge'|'m5a.12xlarge'|'m5a.16xlarge'|'m5a.24xlarge'|'m5d.large'|'m5d.xlarge'|'m5d.2xlarge'|'m5d.4xlarge'|'m5d.8xlarge'|'m5d.12xlarge'|'m5d.16xlarge'|'m5d.24xlarge'|'m5d.metal'|'m5ad.large'|'m5ad.xlarge'|'m5ad.2xlarge'|'m5ad.4xlarge'|'m5ad.8xlarge'|'m5ad.12xlarge'|'m5ad.16xlarge'|'m5ad.24xlarge'|'m5zn.large'|'m5zn.xlarge'|'m5zn.2xlarge'|'m5zn.3xlarge'|'m5zn.6xlarge'|'m5zn.12xlarge'|'m5zn.metal'|'h1.2xlarge'|'h1.4xlarge'|'h1.8xlarge'|'h1.16xlarge'|'z1d.large'|'z1d.xlarge'|'z1d.2xlarge'|'z1d.3xlarge'|'z1d.6xlarge'|'z1d.12xlarge'|'z1d.metal'|'u-6tb1.metal'|'u-9tb1.metal'|'u-12tb1.metal'|'u-18tb1.metal'|'u-24tb1.metal'|'a1.medium'|'a1.large'|'a1.xlarge'|'a1.2xlarge'|'a1.4xlarge'|'a1.metal'|'m5dn.large'|'m5dn.xlarge'|'m5dn.2xlarge'|'m5dn.4xlarge'|'m5dn.8xlarge'|'m5dn.12xlarge'|'m5dn.16xlarge'|'m5dn.24xlarge'|'m5n.large'|'m5n.xlarge'|'m5n.2xlarge'|'m5n.4xlarge'|'m5n.8xlarge'|'m5n.12xlarge'|'m5n.16xlarge'|'m5n.24xlarge'|'r5dn.large'|'r5dn.xlarge'|'r5dn.2xlarge'|'r5dn.4xlarge'|'r5dn.8xlarge'|'r5dn.12xlarge'|'r5dn.16xlarge'|'r5dn.24xlarge'|'r5n.large'|'r5n.xlarge'|'r5n.2xlarge'|'r5n.4xlarge'|'r5n.8xlarge'|'r5n.12xlarge'|'r5n.16xlarge'|'r5n.24xlarge'|'inf1.xlarge'|'inf1.2xlarge'|'inf1.6xlarge'|'inf1.24xlarge'|'m6g.metal'|'m6g.medium'|'m6g.large'|'m6g.xlarge'|'m6g.2xlarge'|'m6g.4xlarge'|'m6g.8xlarge'|'m6g.12xlarge'|'m6g.16xlarge'|'m6gd.metal'|'m6gd.medium'|'m6gd.large'|'m6gd.xlarge'|'m6gd.2xlarge'|'m6gd.4xlarge'|'m6gd.8xlarge'|'m6gd.12xlarge'|'m6gd.16xlarge'|'mac1.metal',
            'ProductDescription': 'Linux/UNIX'|'Linux/UNIX (Amazon VPC)'|'Windows'|'Windows (Amazon VPC)',
            'SpotPrice': 'string',
            'Timestamp': datetime(2015, 1, 1)
        },
    ]
}

**Response Structure**

- _(dict) --_

    Contains the output of DescribeSpotPriceHistory.

    - **SpotPriceHistory** _(list) --_

        The historical Spot prices.

        - _(dict) --_

            Describes the maximum price per hour that you are willing to pay for a Spot Instance.

            - **AvailabilityZone** _(string) --_

                The Availability Zone.

            - **InstanceType** _(string) --_

                The instance type.

            - **ProductDescription** _(string) --_

                A general description of the AMI.

            - **SpotPrice** _(string) --_

                The maximum price per hour that you are willing to pay for a Spot Instance.

            - **Timestamp** _(datetime) --_

                The date and time the request was created, in UTC format (for example, _YYYY_ -_MM_ -_DD_ T*HH* :_MM_ :_SS_ Z).


_class_ EC2.Paginator.DescribeStaleSecurityGroups

paginator = client.get_paginator('describe_stale_security_groups')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_stale_security_groups()](#EC2.Client.describe_stale_security_groups "EC2.Client.describe_stale_security_groups").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeStaleSecurityGroups)

**Request Syntax**

response_iterator = paginator.paginate(
    DryRun=True|False,
    VpcId='string',
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **VpcId** (_string_) --

    **[REQUIRED]**

    The ID of the VPC.

- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'StaleSecurityGroupSet': [
        {
            'Description': 'string',
            'GroupId': 'string',
            'GroupName': 'string',
            'StaleIpPermissions': [
                {
                    'FromPort': 123,
                    'IpProtocol': 'string',
                    'IpRanges': [
                        'string',
                    ],
                    'PrefixListIds': [
                        'string',
                    ],
                    'ToPort': 123,
                    'UserIdGroupPairs': [
                        {
                            'Description': 'string',
                            'GroupId': 'string',
                            'GroupName': 'string',
                            'PeeringStatus': 'string',
                            'UserId': 'string',
                            'VpcId': 'string',
                            'VpcPeeringConnectionId': 'string'
                        },
                    ]
                },
            ],
            'StaleIpPermissionsEgress': [
                {
                    'FromPort': 123,
                    'IpProtocol': 'string',
                    'IpRanges': [
                        'string',
                    ],
                    'PrefixListIds': [
                        'string',
                    ],
                    'ToPort': 123,
                    'UserIdGroupPairs': [
                        {
                            'Description': 'string',
                            'GroupId': 'string',
                            'GroupName': 'string',
                            'PeeringStatus': 'string',
                            'UserId': 'string',
                            'VpcId': 'string',
                            'VpcPeeringConnectionId': 'string'
                        },
                    ]
                },
            ],
            'VpcId': 'string'
        },
    ]
}

**Response Structure**

- _(dict) --_

    - **StaleSecurityGroupSet** _(list) --_

        Information about the stale security groups.

        - _(dict) --_

            Describes a stale security group (a security group that contains stale rules).

            - **Description** _(string) --_

                The description of the security group.

            - **GroupId** _(string) --_

                The ID of the security group.

            - **GroupName** _(string) --_

                The name of the security group.

            - **StaleIpPermissions** _(list) --_

                Information about the stale inbound rules in the security group.

                - _(dict) --_

                    Describes a stale rule in a security group.

                    - **FromPort** _(integer) --_

                        The start of the port range for the TCP and UDP protocols, or an ICMP type number. A value of -1 indicates all ICMP types.

                    - **IpProtocol** _(string) --_

                        The IP protocol name (for tcp , udp , and icmp ) or number (see [Protocol Numbers)](https://www.iana.org/assignments/protocol-numbers/protocol-numbers.xhtml) .

                    - **IpRanges** _(list) --_

                        The IP ranges. Not applicable for stale security group rules.

                        - _(string) --_
                    - **PrefixListIds** _(list) --_

                        The prefix list IDs. Not applicable for stale security group rules.

                        - _(string) --_
                    - **ToPort** _(integer) --_

                        The end of the port range for the TCP and UDP protocols, or an ICMP type number. A value of -1 indicates all ICMP types.

                    - **UserIdGroupPairs** _(list) --_

                        The security group pairs. Returns the ID of the referenced security group and VPC, and the ID and status of the VPC peering connection.

                        - _(dict) --_

                            Describes a security group and AWS account ID pair.

                            - **Description** _(string) --_

                                A description for the security group rule that references this user ID group pair.

                                Constraints: Up to 255 characters in length. Allowed characters are a-z, A-Z, 0-9, spaces, and ._-:/()#,@[]+=;{}!$*

                            - **GroupId** _(string) --_

                                The ID of the security group.

                            - **GroupName** _(string) --_

                                The name of the security group. In a request, use this parameter for a security group in EC2-Classic or a default VPC only. For a security group in a nondefault VPC, use the security group ID.

                                For a referenced security group in another VPC, this value is not returned if the referenced security group is deleted.

                            - **PeeringStatus** _(string) --_

                                The status of a VPC peering connection, if applicable.

                            - **UserId** _(string) --_

                                The ID of an AWS account.

                                For a referenced security group in another VPC, the account ID of the referenced security group is returned in the response. If the referenced security group is deleted, this value is not returned.

                                [EC2-Classic] Required when adding or removing rules that reference a security group in another AWS account.

                            - **VpcId** _(string) --_

                                The ID of the VPC for the referenced security group, if applicable.

                            - **VpcPeeringConnectionId** _(string) --_

                                The ID of the VPC peering connection, if applicable.

            - **StaleIpPermissionsEgress** _(list) --_

                Information about the stale outbound rules in the security group.

                - _(dict) --_

                    Describes a stale rule in a security group.

                    - **FromPort** _(integer) --_

                        The start of the port range for the TCP and UDP protocols, or an ICMP type number. A value of -1 indicates all ICMP types.

                    - **IpProtocol** _(string) --_

                        The IP protocol name (for tcp , udp , and icmp ) or number (see [Protocol Numbers)](https://www.iana.org/assignments/protocol-numbers/protocol-numbers.xhtml) .

                    - **IpRanges** _(list) --_

                        The IP ranges. Not applicable for stale security group rules.

                        - _(string) --_
                    - **PrefixListIds** _(list) --_

                        The prefix list IDs. Not applicable for stale security group rules.

                        - _(string) --_
                    - **ToPort** _(integer) --_

                        The end of the port range for the TCP and UDP protocols, or an ICMP type number. A value of -1 indicates all ICMP types.

                    - **UserIdGroupPairs** _(list) --_

                        The security group pairs. Returns the ID of the referenced security group and VPC, and the ID and status of the VPC peering connection.

                        - _(dict) --_

                            Describes a security group and AWS account ID pair.

                            - **Description** _(string) --_

                                A description for the security group rule that references this user ID group pair.

                                Constraints: Up to 255 characters in length. Allowed characters are a-z, A-Z, 0-9, spaces, and ._-:/()#,@[]+=;{}!$*

                            - **GroupId** _(string) --_

                                The ID of the security group.

                            - **GroupName** _(string) --_

                                The name of the security group. In a request, use this parameter for a security group in EC2-Classic or a default VPC only. For a security group in a nondefault VPC, use the security group ID.

                                For a referenced security group in another VPC, this value is not returned if the referenced security group is deleted.

                            - **PeeringStatus** _(string) --_

                                The status of a VPC peering connection, if applicable.

                            - **UserId** _(string) --_

                                The ID of an AWS account.

                                For a referenced security group in another VPC, the account ID of the referenced security group is returned in the response. If the referenced security group is deleted, this value is not returned.

                                [EC2-Classic] Required when adding or removing rules that reference a security group in another AWS account.

                            - **VpcId** _(string) --_

                                The ID of the VPC for the referenced security group, if applicable.

                            - **VpcPeeringConnectionId** _(string) --_

                                The ID of the VPC peering connection, if applicable.

            - **VpcId** _(string) --_

                The ID of the VPC for the security group.


_class_ EC2.Paginator.DescribeSubnets

paginator = client.get_paginator('describe_subnets')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_subnets()](#EC2.Client.describe_subnets "EC2.Client.describe_subnets").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeSubnets)

**Request Syntax**

response_iterator = paginator.paginate(
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    SubnetIds=[
        'string',
    ],
    DryRun=True|False,
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **Filters** (_list_) --

    One or more filters.

    - availability-zone - The Availability Zone for the subnet. You can also use availabilityZone as the filter name.
    - availability-zone-id - The ID of the Availability Zone for the subnet. You can also use availabilityZoneId as the filter name.
    - available-ip-address-count - The number of IPv4 addresses in the subnet that are available.
    - cidr-block - The IPv4 CIDR block of the subnet. The CIDR block you specify must exactly match the subnet's CIDR block for information to be returned for the subnet. You can also use cidr or cidrBlock as the filter names.
    - default-for-az - Indicates whether this is the default subnet for the Availability Zone. You can also use defaultForAz as the filter name.
    - ipv6-cidr-block-association.ipv6-cidr-block - An IPv6 CIDR block associated with the subnet.
    - ipv6-cidr-block-association.association-id - An association ID for an IPv6 CIDR block associated with the subnet.
    - ipv6-cidr-block-association.state - The state of an IPv6 CIDR block associated with the subnet.
    - owner-id - The ID of the AWS account that owns the subnet.
    - state - The state of the subnet (pending | available ).
    - subnet-arn - The Amazon Resource Name (ARN) of the subnet.
    - subnet-id - The ID of the subnet.
    - tag :<key> - The key/value combination of a tag assigned to the resource. Use the tag key in the filter name and the tag value as the filter value. For example, to find all resources that have a tag with the key Owner and the value TeamA , specify tag:Owner for the filter name and TeamA for the filter value.
    - tag-key - The key of a tag assigned to the resource. Use this filter to find all resources assigned a tag with a specific key, regardless of the tag value.
    - vpc-id - The ID of the VPC for the subnet.

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **SubnetIds** (_list_) --

    One or more subnet IDs.

    Default: Describes all your subnets.

    - _(string) --_
- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'Subnets': [
        {
            'AvailabilityZone': 'string',
            'AvailabilityZoneId': 'string',
            'AvailableIpAddressCount': 123,
            'CidrBlock': 'string',
            'DefaultForAz': True|False,
            'MapPublicIpOnLaunch': True|False,
            'MapCustomerOwnedIpOnLaunch': True|False,
            'CustomerOwnedIpv4Pool': 'string',
            'State': 'pending'|'available',
            'SubnetId': 'string',
            'VpcId': 'string',
            'OwnerId': 'string',
            'AssignIpv6AddressOnCreation': True|False,
            'Ipv6CidrBlockAssociationSet': [
                {
                    'AssociationId': 'string',
                    'Ipv6CidrBlock': 'string',
                    'Ipv6CidrBlockState': {
                        'State': 'associating'|'associated'|'disassociating'|'disassociated'|'failing'|'failed',
                        'StatusMessage': 'string'
                    }
                },
            ],
            'Tags': [
                {
                    'Key': 'string',
                    'Value': 'string'
                },
            ],
            'SubnetArn': 'string',
            'OutpostArn': 'string'
        },
    ],

}

**Response Structure**

- _(dict) --_

    - **Subnets** _(list) --_

        Information about one or more subnets.

        - _(dict) --_

            Describes a subnet.

            - **AvailabilityZone** _(string) --_

                The Availability Zone of the subnet.

            - **AvailabilityZoneId** _(string) --_

                The AZ ID of the subnet.

            - **AvailableIpAddressCount** _(integer) --_

                The number of unused private IPv4 addresses in the subnet. The IPv4 addresses for any stopped instances are considered unavailable.

            - **CidrBlock** _(string) --_

                The IPv4 CIDR block assigned to the subnet.

            - **DefaultForAz** _(boolean) --_

                Indicates whether this is the default subnet for the Availability Zone.

            - **MapPublicIpOnLaunch** _(boolean) --_

                Indicates whether instances launched in this subnet receive a public IPv4 address.

            - **MapCustomerOwnedIpOnLaunch** _(boolean) --_

                Indicates whether a network interface created in this subnet (including a network interface created by RunInstances ) receives a customer-owned IPv4 address.

            - **CustomerOwnedIpv4Pool** _(string) --_

                The customer-owned IPv4 address pool associated with the subnet.

            - **State** _(string) --_

                The current state of the subnet.

            - **SubnetId** _(string) --_

                The ID of the subnet.

            - **VpcId** _(string) --_

                The ID of the VPC the subnet is in.

            - **OwnerId** _(string) --_

                The ID of the AWS account that owns the subnet.

            - **AssignIpv6AddressOnCreation** _(boolean) --_

                Indicates whether a network interface created in this subnet (including a network interface created by RunInstances ) receives an IPv6 address.

            - **Ipv6CidrBlockAssociationSet** _(list) --_

                Information about the IPv6 CIDR blocks associated with the subnet.

                - _(dict) --_

                    Describes an IPv6 CIDR block associated with a subnet.

                    - **AssociationId** _(string) --_

                        The association ID for the CIDR block.

                    - **Ipv6CidrBlock** _(string) --_

                        The IPv6 CIDR block.

                    - **Ipv6CidrBlockState** _(dict) --_

                        Information about the state of the CIDR block.

                        - **State** _(string) --_

                            The state of a CIDR block.

                        - **StatusMessage** _(string) --_

                            A message about the status of the CIDR block, if applicable.

            - **Tags** _(list) --_

                Any tags assigned to the subnet.

                - _(dict) --_

                    Describes a tag.

                    - **Key** _(string) --_

                        The key of the tag.

                        Constraints: Tag keys are case-sensitive and accept a maximum of 127 Unicode characters. May not begin with aws: .

                    - **Value** _(string) --_

                        The value of the tag.

                        Constraints: Tag values are case-sensitive and accept a maximum of 255 Unicode characters.

            - **SubnetArn** _(string) --_

                The Amazon Resource Name (ARN) of the subnet.

            - **OutpostArn** _(string) --_

                The Amazon Resource Name (ARN) of the Outpost.


_class_ EC2.Paginator.DescribeTags

paginator = client.get_paginator('describe_tags')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_tags()](#EC2.Client.describe_tags "EC2.Client.describe_tags").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeTags)

**Request Syntax**

response_iterator = paginator.paginate(
    DryRun=True|False,
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **Filters** (_list_) --

    The filters.

    - key - The tag key.
    - resource-id - The ID of the resource.
    - resource-type - The resource type (customer-gateway | dedicated-host | dhcp-options | elastic-ip | fleet | fpga-image | host-reservation | image | instance | internet-gateway | key-pair | launch-template | natgateway | network-acl | network-interface | placement-group | reserved-instances | route-table | security-group | snapshot | spot-instances-request | subnet | volume | vpc | vpc-endpoint | vpc-endpoint-service | vpc-peering-connection | vpn-connection | vpn-gateway ).
    - tag :<key> - The key/value combination of the tag. For example, specify "[tag:Owner](tag:Owner)" for the filter name and "TeamA" for the filter value to find resources with the tag "Owner=TeamA".
    - value - The tag value.

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'Tags': [
        {
            'Key': 'string',
            'ResourceId': 'string',
            'ResourceType': 'client-vpn-endpoint'|'customer-gateway'|'dedicated-host'|'dhcp-options'|'egress-only-internet-gateway'|'elastic-ip'|'elastic-gpu'|'export-image-task'|'export-instance-task'|'fleet'|'fpga-image'|'host-reservation'|'image'|'import-image-task'|'import-snapshot-task'|'instance'|'internet-gateway'|'key-pair'|'launch-template'|'local-gateway-route-table-vpc-association'|'natgateway'|'network-acl'|'network-interface'|'network-insights-analysis'|'network-insights-path'|'placement-group'|'reserved-instances'|'route-table'|'security-group'|'snapshot'|'spot-fleet-request'|'spot-instances-request'|'subnet'|'traffic-mirror-filter'|'traffic-mirror-session'|'traffic-mirror-target'|'transit-gateway'|'transit-gateway-attachment'|'transit-gateway-connect-peer'|'transit-gateway-multicast-domain'|'transit-gateway-route-table'|'volume'|'vpc'|'vpc-peering-connection'|'vpn-connection'|'vpn-gateway'|'vpc-flow-log',
            'Value': 'string'
        },
    ]
}

**Response Structure**

- _(dict) --_

    - **Tags** _(list) --_

        The tags.

        - _(dict) --_

            Describes a tag.

            - **Key** _(string) --_

                The tag key.

            - **ResourceId** _(string) --_

                The ID of the resource.

            - **ResourceType** _(string) --_

                The resource type.

            - **Value** _(string) --_

                The tag value.


_class_ EC2.Paginator.DescribeTrafficMirrorFilters

paginator = client.get_paginator('describe_traffic_mirror_filters')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_traffic_mirror_filters()](#EC2.Client.describe_traffic_mirror_filters "EC2.Client.describe_traffic_mirror_filters").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeTrafficMirrorFilters)

**Request Syntax**

response_iterator = paginator.paginate(
    TrafficMirrorFilterIds=[
        'string',
    ],
    DryRun=True|False,
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **TrafficMirrorFilterIds** (_list_) --

    The ID of the Traffic Mirror filter.

    - _(string) --_
- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **Filters** (_list_) --

    One or more filters. The possible values are:

    - description : The Traffic Mirror filter description.
    - traffic-mirror-filter-id : The ID of the Traffic Mirror filter.

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'TrafficMirrorFilters': [
        {
            'TrafficMirrorFilterId': 'string',
            'IngressFilterRules': [
                {
                    'TrafficMirrorFilterRuleId': 'string',
                    'TrafficMirrorFilterId': 'string',
                    'TrafficDirection': 'ingress'|'egress',
                    'RuleNumber': 123,
                    'RuleAction': 'accept'|'reject',
                    'Protocol': 123,
                    'DestinationPortRange': {
                        'FromPort': 123,
                        'ToPort': 123
                    },
                    'SourcePortRange': {
                        'FromPort': 123,
                        'ToPort': 123
                    },
                    'DestinationCidrBlock': 'string',
                    'SourceCidrBlock': 'string',
                    'Description': 'string'
                },
            ],
            'EgressFilterRules': [
                {
                    'TrafficMirrorFilterRuleId': 'string',
                    'TrafficMirrorFilterId': 'string',
                    'TrafficDirection': 'ingress'|'egress',
                    'RuleNumber': 123,
                    'RuleAction': 'accept'|'reject',
                    'Protocol': 123,
                    'DestinationPortRange': {
                        'FromPort': 123,
                        'ToPort': 123
                    },
                    'SourcePortRange': {
                        'FromPort': 123,
                        'ToPort': 123
                    },
                    'DestinationCidrBlock': 'string',
                    'SourceCidrBlock': 'string',
                    'Description': 'string'
                },
            ],
            'NetworkServices': [
                'amazon-dns',
            ],
            'Description': 'string',
            'Tags': [
                {
                    'Key': 'string',
                    'Value': 'string'
                },
            ]
        },
    ],

}

**Response Structure**

- _(dict) --_

    - **TrafficMirrorFilters** _(list) --_

        Information about one or more Traffic Mirror filters.

        - _(dict) --_

            Describes the Traffic Mirror filter.

            - **TrafficMirrorFilterId** _(string) --_

                The ID of the Traffic Mirror filter.

            - **IngressFilterRules** _(list) --_

                Information about the ingress rules that are associated with the Traffic Mirror filter.

                - _(dict) --_

                    Describes the Traffic Mirror rule.

                    - **TrafficMirrorFilterRuleId** _(string) --_

                        The ID of the Traffic Mirror rule.

                    - **TrafficMirrorFilterId** _(string) --_

                        The ID of the Traffic Mirror filter that the rule is associated with.

                    - **TrafficDirection** _(string) --_

                        The traffic direction assigned to the Traffic Mirror rule.

                    - **RuleNumber** _(integer) --_

                        The rule number of the Traffic Mirror rule.

                    - **RuleAction** _(string) --_

                        The action assigned to the Traffic Mirror rule.

                    - **Protocol** _(integer) --_

                        The protocol assigned to the Traffic Mirror rule.

                    - **DestinationPortRange** _(dict) --_

                        The destination port range assigned to the Traffic Mirror rule.

                        - **FromPort** _(integer) --_

                            The start of the Traffic Mirror port range. This applies to the TCP and UDP protocols.

                        - **ToPort** _(integer) --_

                            The end of the Traffic Mirror port range. This applies to the TCP and UDP protocols.

                    - **SourcePortRange** _(dict) --_

                        The source port range assigned to the Traffic Mirror rule.

                        - **FromPort** _(integer) --_

                            The start of the Traffic Mirror port range. This applies to the TCP and UDP protocols.

                        - **ToPort** _(integer) --_

                            The end of the Traffic Mirror port range. This applies to the TCP and UDP protocols.

                    - **DestinationCidrBlock** _(string) --_

                        The destination CIDR block assigned to the Traffic Mirror rule.

                    - **SourceCidrBlock** _(string) --_

                        The source CIDR block assigned to the Traffic Mirror rule.

                    - **Description** _(string) --_

                        The description of the Traffic Mirror rule.

            - **EgressFilterRules** _(list) --_

                Information about the egress rules that are associated with the Traffic Mirror filter.

                - _(dict) --_

                    Describes the Traffic Mirror rule.

                    - **TrafficMirrorFilterRuleId** _(string) --_

                        The ID of the Traffic Mirror rule.

                    - **TrafficMirrorFilterId** _(string) --_

                        The ID of the Traffic Mirror filter that the rule is associated with.

                    - **TrafficDirection** _(string) --_

                        The traffic direction assigned to the Traffic Mirror rule.

                    - **RuleNumber** _(integer) --_

                        The rule number of the Traffic Mirror rule.

                    - **RuleAction** _(string) --_

                        The action assigned to the Traffic Mirror rule.

                    - **Protocol** _(integer) --_

                        The protocol assigned to the Traffic Mirror rule.

                    - **DestinationPortRange** _(dict) --_

                        The destination port range assigned to the Traffic Mirror rule.

                        - **FromPort** _(integer) --_

                            The start of the Traffic Mirror port range. This applies to the TCP and UDP protocols.

                        - **ToPort** _(integer) --_

                            The end of the Traffic Mirror port range. This applies to the TCP and UDP protocols.

                    - **SourcePortRange** _(dict) --_

                        The source port range assigned to the Traffic Mirror rule.

                        - **FromPort** _(integer) --_

                            The start of the Traffic Mirror port range. This applies to the TCP and UDP protocols.

                        - **ToPort** _(integer) --_

                            The end of the Traffic Mirror port range. This applies to the TCP and UDP protocols.

                    - **DestinationCidrBlock** _(string) --_

                        The destination CIDR block assigned to the Traffic Mirror rule.

                    - **SourceCidrBlock** _(string) --_

                        The source CIDR block assigned to the Traffic Mirror rule.

                    - **Description** _(string) --_

                        The description of the Traffic Mirror rule.

            - **NetworkServices** _(list) --_

                The network service traffic that is associated with the Traffic Mirror filter.

                - _(string) --_
            - **Description** _(string) --_

                The description of the Traffic Mirror filter.

            - **Tags** _(list) --_

                The tags assigned to the Traffic Mirror filter.

                - _(dict) --_

                    Describes a tag.

                    - **Key** _(string) --_

                        The key of the tag.

                        Constraints: Tag keys are case-sensitive and accept a maximum of 127 Unicode characters. May not begin with aws: .

                    - **Value** _(string) --_

                        The value of the tag.

                        Constraints: Tag values are case-sensitive and accept a maximum of 255 Unicode characters.


_class_ EC2.Paginator.DescribeTrafficMirrorSessions

paginator = client.get_paginator('describe_traffic_mirror_sessions')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_traffic_mirror_sessions()](#EC2.Client.describe_traffic_mirror_sessions "EC2.Client.describe_traffic_mirror_sessions").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeTrafficMirrorSessions)

**Request Syntax**

response_iterator = paginator.paginate(
    TrafficMirrorSessionIds=[
        'string',
    ],
    DryRun=True|False,
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **TrafficMirrorSessionIds** (_list_) --

    The ID of the Traffic Mirror session.

    - _(string) --_
- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **Filters** (_list_) --

    One or more filters. The possible values are:

    - description : The Traffic Mirror session description.
    - network-interface-id : The ID of the Traffic Mirror session network interface.
    - owner-id : The ID of the account that owns the Traffic Mirror session.
    - packet-length : The assigned number of packets to mirror.
    - session-number : The assigned session number.
    - traffic-mirror-filter-id : The ID of the Traffic Mirror filter.
    - traffic-mirror-session-id : The ID of the Traffic Mirror session.
    - traffic-mirror-target-id : The ID of the Traffic Mirror target.
    - virtual-network-id : The virtual network ID of the Traffic Mirror session.

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'TrafficMirrorSessions': [
        {
            'TrafficMirrorSessionId': 'string',
            'TrafficMirrorTargetId': 'string',
            'TrafficMirrorFilterId': 'string',
            'NetworkInterfaceId': 'string',
            'OwnerId': 'string',
            'PacketLength': 123,
            'SessionNumber': 123,
            'VirtualNetworkId': 123,
            'Description': 'string',
            'Tags': [
                {
                    'Key': 'string',
                    'Value': 'string'
                },
            ]
        },
    ],

}

**Response Structure**

- _(dict) --_

    - **TrafficMirrorSessions** _(list) --_

        Describes one or more Traffic Mirror sessions. By default, all Traffic Mirror sessions are described. Alternatively, you can filter the results.

        - _(dict) --_

            Describes a Traffic Mirror session.

            - **TrafficMirrorSessionId** _(string) --_

                The ID for the Traffic Mirror session.

            - **TrafficMirrorTargetId** _(string) --_

                The ID of the Traffic Mirror target.

            - **TrafficMirrorFilterId** _(string) --_

                The ID of the Traffic Mirror filter.

            - **NetworkInterfaceId** _(string) --_

                The ID of the Traffic Mirror session's network interface.

            - **OwnerId** _(string) --_

                The ID of the account that owns the Traffic Mirror session.

            - **PacketLength** _(integer) --_

                The number of bytes in each packet to mirror. These are the bytes after the VXLAN header. To mirror a subset, set this to the length (in bytes) to mirror. For example, if you set this value to 100, then the first 100 bytes that meet the filter criteria are copied to the target. Do not specify this parameter when you want to mirror the entire packet

            - **SessionNumber** _(integer) --_

                The session number determines the order in which sessions are evaluated when an interface is used by multiple sessions. The first session with a matching filter is the one that mirrors the packets.

                Valid values are 1-32766.

            - **VirtualNetworkId** _(integer) --_

                The virtual network ID associated with the Traffic Mirror session.

            - **Description** _(string) --_

                The description of the Traffic Mirror session.

            - **Tags** _(list) --_

                The tags assigned to the Traffic Mirror session.

                - _(dict) --_

                    Describes a tag.

                    - **Key** _(string) --_

                        The key of the tag.

                        Constraints: Tag keys are case-sensitive and accept a maximum of 127 Unicode characters. May not begin with aws: .

                    - **Value** _(string) --_

                        The value of the tag.

                        Constraints: Tag values are case-sensitive and accept a maximum of 255 Unicode characters.


_class_ EC2.Paginator.DescribeTrafficMirrorTargets

paginator = client.get_paginator('describe_traffic_mirror_targets')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_traffic_mirror_targets()](#EC2.Client.describe_traffic_mirror_targets "EC2.Client.describe_traffic_mirror_targets").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeTrafficMirrorTargets)

**Request Syntax**

response_iterator = paginator.paginate(
    TrafficMirrorTargetIds=[
        'string',
    ],
    DryRun=True|False,
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **TrafficMirrorTargetIds** (_list_) --

    The ID of the Traffic Mirror targets.

    - _(string) --_
- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **Filters** (_list_) --

    One or more filters. The possible values are:

    - description : The Traffic Mirror target description.
    - network-interface-id : The ID of the Traffic Mirror session network interface.
    - network-load-balancer-arn : The Amazon Resource Name (ARN) of the Network Load Balancer that is associated with the session.
    - owner-id : The ID of the account that owns the Traffic Mirror session.
    - traffic-mirror-target-id : The ID of the Traffic Mirror target.

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'TrafficMirrorTargets': [
        {
            'TrafficMirrorTargetId': 'string',
            'NetworkInterfaceId': 'string',
            'NetworkLoadBalancerArn': 'string',
            'Type': 'network-interface'|'network-load-balancer',
            'Description': 'string',
            'OwnerId': 'string',
            'Tags': [
                {
                    'Key': 'string',
                    'Value': 'string'
                },
            ]
        },
    ],

}

**Response Structure**

- _(dict) --_

    - **TrafficMirrorTargets** _(list) --_

        Information about one or more Traffic Mirror targets.

        - _(dict) --_

            Describes a Traffic Mirror target.

            - **TrafficMirrorTargetId** _(string) --_

                The ID of the Traffic Mirror target.

            - **NetworkInterfaceId** _(string) --_

                The network interface ID that is attached to the target.

            - **NetworkLoadBalancerArn** _(string) --_

                The Amazon Resource Name (ARN) of the Network Load Balancer.

            - **Type** _(string) --_

                The type of Traffic Mirror target.

            - **Description** _(string) --_

                Information about the Traffic Mirror target.

            - **OwnerId** _(string) --_

                The ID of the account that owns the Traffic Mirror target.

            - **Tags** _(list) --_

                The tags assigned to the Traffic Mirror target.

                - _(dict) --_

                    Describes a tag.

                    - **Key** _(string) --_

                        The key of the tag.

                        Constraints: Tag keys are case-sensitive and accept a maximum of 127 Unicode characters. May not begin with aws: .

                    - **Value** _(string) --_

                        The value of the tag.

                        Constraints: Tag values are case-sensitive and accept a maximum of 255 Unicode characters.


_class_ EC2.Paginator.DescribeTransitGatewayAttachments

paginator = client.get_paginator('describe_transit_gateway_attachments')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_transit_gateway_attachments()](#EC2.Client.describe_transit_gateway_attachments "EC2.Client.describe_transit_gateway_attachments").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeTransitGatewayAttachments)

**Request Syntax**

response_iterator = paginator.paginate(
    TransitGatewayAttachmentIds=[
        'string',
    ],
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    DryRun=True|False,
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **TransitGatewayAttachmentIds** (_list_) --

    The IDs of the attachments.

    - _(string) --_
- **Filters** (_list_) --

    One or more filters. The possible values are:

    - association.state - The state of the association (associating | associated | disassociating ).
    - association.transit-gateway-route-table-id - The ID of the route table for the transit gateway.
    - resource-id - The ID of the resource.
    - resource-owner-id - The ID of the AWS account that owns the resource.
    - resource-type - The resource type. Valid values are vpc | vpn | direct-connect-gateway | peering | connect .
    - state - The state of the attachment. Valid values are available | deleted | deleting | failed | failing | initiatingRequest | modifying | pendingAcceptance | pending | rollingBack | rejected | rejecting .
    - transit-gateway-attachment-id - The ID of the attachment.
    - transit-gateway-id - The ID of the transit gateway.
    - transit-gateway-owner-id - The ID of the AWS account that owns the transit gateway.

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'TransitGatewayAttachments': [
        {
            'TransitGatewayAttachmentId': 'string',
            'TransitGatewayId': 'string',
            'TransitGatewayOwnerId': 'string',
            'ResourceOwnerId': 'string',
            'ResourceType': 'vpc'|'vpn'|'direct-connect-gateway'|'connect'|'peering'|'tgw-peering',
            'ResourceId': 'string',
            'State': 'initiating'|'initiatingRequest'|'pendingAcceptance'|'rollingBack'|'pending'|'available'|'modifying'|'deleting'|'deleted'|'failed'|'rejected'|'rejecting'|'failing',
            'Association': {
                'TransitGatewayRouteTableId': 'string',
                'State': 'associating'|'associated'|'disassociating'|'disassociated'
            },
            'CreationTime': datetime(2015, 1, 1),
            'Tags': [
                {
                    'Key': 'string',
                    'Value': 'string'
                },
            ]
        },
    ],

}

**Response Structure**

- _(dict) --_

    - **TransitGatewayAttachments** _(list) --_

        Information about the attachments.

        - _(dict) --_

            Describes an attachment between a resource and a transit gateway.

            - **TransitGatewayAttachmentId** _(string) --_

                The ID of the attachment.

            - **TransitGatewayId** _(string) --_

                The ID of the transit gateway.

            - **TransitGatewayOwnerId** _(string) --_

                The ID of the AWS account that owns the transit gateway.

            - **ResourceOwnerId** _(string) --_

                The ID of the AWS account that owns the resource.

            - **ResourceType** _(string) --_

                The resource type. Note that the tgw-peering resource type has been deprecated.

            - **ResourceId** _(string) --_

                The ID of the resource.

            - **State** _(string) --_

                The attachment state. Note that the initiating state has been deprecated.

            - **Association** _(dict) --_

                The association.

                - **TransitGatewayRouteTableId** _(string) --_

                    The ID of the route table for the transit gateway.

                - **State** _(string) --_

                    The state of the association.

            - **CreationTime** _(datetime) --_

                The creation time.

            - **Tags** _(list) --_

                The tags for the attachment.

                - _(dict) --_

                    Describes a tag.

                    - **Key** _(string) --_

                        The key of the tag.

                        Constraints: Tag keys are case-sensitive and accept a maximum of 127 Unicode characters. May not begin with aws: .

                    - **Value** _(string) --_

                        The value of the tag.

                        Constraints: Tag values are case-sensitive and accept a maximum of 255 Unicode characters.


_class_ EC2.Paginator.DescribeTransitGatewayConnectPeers

paginator = client.get_paginator('describe_transit_gateway_connect_peers')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_transit_gateway_connect_peers()](#EC2.Client.describe_transit_gateway_connect_peers "EC2.Client.describe_transit_gateway_connect_peers").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeTransitGatewayConnectPeers)

**Request Syntax**

response_iterator = paginator.paginate(
    TransitGatewayConnectPeerIds=[
        'string',
    ],
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    DryRun=True|False,
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **TransitGatewayConnectPeerIds** (_list_) --

    The IDs of the Connect peers.

    - _(string) --_
- **Filters** (_list_) --

    One or more filters. The possible values are:

    - state - The state of the Connect peer (pending | available | deleting | deleted ).
    - transit-gateway-attachment-id - The ID of the attachment.
    - transit-gateway-connect-peer-id - The ID of the Connect peer.

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'TransitGatewayConnectPeers': [
        {
            'TransitGatewayAttachmentId': 'string',
            'TransitGatewayConnectPeerId': 'string',
            'State': 'pending'|'available'|'deleting'|'deleted',
            'CreationTime': datetime(2015, 1, 1),
            'ConnectPeerConfiguration': {
                'TransitGatewayAddress': 'string',
                'PeerAddress': 'string',
                'InsideCidrBlocks': [
                    'string',
                ],
                'Protocol': 'gre',
                'BgpConfigurations': [
                    {
                        'TransitGatewayAsn': 123,
                        'PeerAsn': 123,
                        'TransitGatewayAddress': 'string',
                        'PeerAddress': 'string',
                        'BgpStatus': 'up'|'down'
                    },
                ]
            },
            'Tags': [
                {
                    'Key': 'string',
                    'Value': 'string'
                },
            ]
        },
    ],

}

**Response Structure**

- _(dict) --_

    - **TransitGatewayConnectPeers** _(list) --_

        Information about the Connect peers.

        - _(dict) --_

            Describes a transit gateway Connect peer.

            - **TransitGatewayAttachmentId** _(string) --_

                The ID of the Connect attachment.

            - **TransitGatewayConnectPeerId** _(string) --_

                The ID of the Connect peer.

            - **State** _(string) --_

                The state of the Connect peer.

            - **CreationTime** _(datetime) --_

                The creation time.

            - **ConnectPeerConfiguration** _(dict) --_

                The Connect peer details.

                - **TransitGatewayAddress** _(string) --_

                    The Connect peer IP address on the transit gateway side of the tunnel.

                - **PeerAddress** _(string) --_

                    The Connect peer IP address on the appliance side of the tunnel.

                - **InsideCidrBlocks** _(list) --_

                    The range of interior BGP peer IP addresses.

                    - _(string) --_
                - **Protocol** _(string) --_

                    The tunnel protocol.

                - **BgpConfigurations** _(list) --_

                    The BGP configuration details.

                    - _(dict) --_

                        The BGP configuration information.

                        - **TransitGatewayAsn** _(integer) --_

                            The transit gateway Autonomous System Number (ASN).

                        - **PeerAsn** _(integer) --_

                            The peer Autonomous System Number (ASN).

                        - **TransitGatewayAddress** _(string) --_

                            The interior BGP peer IP address for the transit gateway.

                        - **PeerAddress** _(string) --_

                            The interior BGP peer IP address for the appliance.

                        - **BgpStatus** _(string) --_

                            The BGP status.

            - **Tags** _(list) --_

                The tags for the Connect peer.

                - _(dict) --_

                    Describes a tag.

                    - **Key** _(string) --_

                        The key of the tag.

                        Constraints: Tag keys are case-sensitive and accept a maximum of 127 Unicode characters. May not begin with aws: .

                    - **Value** _(string) --_

                        The value of the tag.

                        Constraints: Tag values are case-sensitive and accept a maximum of 255 Unicode characters.


_class_ EC2.Paginator.DescribeTransitGatewayConnects

paginator = client.get_paginator('describe_transit_gateway_connects')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_transit_gateway_connects()](#EC2.Client.describe_transit_gateway_connects "EC2.Client.describe_transit_gateway_connects").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeTransitGatewayConnects)

**Request Syntax**

response_iterator = paginator.paginate(
    TransitGatewayAttachmentIds=[
        'string',
    ],
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    DryRun=True|False,
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **TransitGatewayAttachmentIds** (_list_) --

    The IDs of the attachments.

    - _(string) --_
- **Filters** (_list_) --

    One or more filters. The possible values are:

    - options.protocol - The tunnel protocol (gre ).
    - state - The state of the attachment (initiating | initiatingRequest | pendingAcceptance | rollingBack | pending | available | modifying | deleting | deleted | failed | rejected | rejecting | failing ).
    - transit-gateway-attachment-id - The ID of the Connect attachment.
    - transit-gateway-id - The ID of the transit gateway.
    - transport-transit-gateway-attachment-id - The ID of the transit gateway attachment from which the Connect attachment was created.

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'TransitGatewayConnects': [
        {
            'TransitGatewayAttachmentId': 'string',
            'TransportTransitGatewayAttachmentId': 'string',
            'TransitGatewayId': 'string',
            'State': 'initiating'|'initiatingRequest'|'pendingAcceptance'|'rollingBack'|'pending'|'available'|'modifying'|'deleting'|'deleted'|'failed'|'rejected'|'rejecting'|'failing',
            'CreationTime': datetime(2015, 1, 1),
            'Options': {
                'Protocol': 'gre'
            },
            'Tags': [
                {
                    'Key': 'string',
                    'Value': 'string'
                },
            ]
        },
    ],

}

**Response Structure**

- _(dict) --_

    - **TransitGatewayConnects** _(list) --_

        Information about the Connect attachments.

        - _(dict) --_

            Describes a transit gateway Connect attachment.

            - **TransitGatewayAttachmentId** _(string) --_

                The ID of the Connect attachment.

            - **TransportTransitGatewayAttachmentId** _(string) --_

                The ID of the attachment from which the Connect attachment was created.

            - **TransitGatewayId** _(string) --_

                The ID of the transit gateway.

            - **State** _(string) --_

                The state of the attachment.

            - **CreationTime** _(datetime) --_

                The creation time.

            - **Options** _(dict) --_

                The Connect attachment options.

                - **Protocol** _(string) --_

                    The tunnel protocol.

            - **Tags** _(list) --_

                The tags for the attachment.

                - _(dict) --_

                    Describes a tag.

                    - **Key** _(string) --_

                        The key of the tag.

                        Constraints: Tag keys are case-sensitive and accept a maximum of 127 Unicode characters. May not begin with aws: .

                    - **Value** _(string) --_

                        The value of the tag.

                        Constraints: Tag values are case-sensitive and accept a maximum of 255 Unicode characters.


_class_ EC2.Paginator.DescribeTransitGatewayMulticastDomains

paginator = client.get_paginator('describe_transit_gateway_multicast_domains')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_transit_gateway_multicast_domains()](#EC2.Client.describe_transit_gateway_multicast_domains "EC2.Client.describe_transit_gateway_multicast_domains").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeTransitGatewayMulticastDomains)

**Request Syntax**

response_iterator = paginator.paginate(
    TransitGatewayMulticastDomainIds=[
        'string',
    ],
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    DryRun=True|False,
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **TransitGatewayMulticastDomainIds** (_list_) --

    The ID of the transit gateway multicast domain.

    - _(string) --_
- **Filters** (_list_) --

    One or more filters. The possible values are:

    - state - The state of the transit gateway multicast domain. Valid values are pending | available | deleting | deleted .
    - transit-gateway-id - The ID of the transit gateway.
    - transit-gateway-multicast-domain-id - The ID of the transit gateway multicast domain.

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'TransitGatewayMulticastDomains': [
        {
            'TransitGatewayMulticastDomainId': 'string',
            'TransitGatewayId': 'string',
            'TransitGatewayMulticastDomainArn': 'string',
            'OwnerId': 'string',
            'Options': {
                'Igmpv2Support': 'enable'|'disable',
                'StaticSourcesSupport': 'enable'|'disable',
                'AutoAcceptSharedAssociations': 'enable'|'disable'
            },
            'State': 'pending'|'available'|'deleting'|'deleted',
            'CreationTime': datetime(2015, 1, 1),
            'Tags': [
                {
                    'Key': 'string',
                    'Value': 'string'
                },
            ]
        },
    ],

}

**Response Structure**

- _(dict) --_

    - **TransitGatewayMulticastDomains** _(list) --_

        Information about the transit gateway multicast domains.

        - _(dict) --_

            Describes the transit gateway multicast domain.

            - **TransitGatewayMulticastDomainId** _(string) --_

                The ID of the transit gateway multicast domain.

            - **TransitGatewayId** _(string) --_

                The ID of the transit gateway.

            - **TransitGatewayMulticastDomainArn** _(string) --_

                The Amazon Resource Name (ARN) of the transit gateway multicast domain.

            - **OwnerId** _(string) --_

                The ID of the AWS account that owns the transit gateway multiicast domain.

            - **Options** _(dict) --_

                The options for the transit gateway multicast domain.

                - **Igmpv2Support** _(string) --_

                    Indicates whether Internet Group Management Protocol (IGMP) version 2 is turned on for the transit gateway multicast domain.

                - **StaticSourcesSupport** _(string) --_

                    Indicates whether support for statically configuring transit gateway multicast group sources is turned on.

                - **AutoAcceptSharedAssociations** _(string) --_

                    Indicates whether to automatically cross-account subnet associations that are associated with the transit gateway multicast domain.

            - **State** _(string) --_

                The state of the transit gateway multicast domain.

            - **CreationTime** _(datetime) --_

                The time the transit gateway multicast domain was created.

            - **Tags** _(list) --_

                The tags for the transit gateway multicast domain.

                - _(dict) --_

                    Describes a tag.

                    - **Key** _(string) --_

                        The key of the tag.

                        Constraints: Tag keys are case-sensitive and accept a maximum of 127 Unicode characters. May not begin with aws: .

                    - **Value** _(string) --_

                        The value of the tag.

                        Constraints: Tag values are case-sensitive and accept a maximum of 255 Unicode characters.


_class_ EC2.Paginator.DescribeTransitGatewayPeeringAttachmentsdefinition")

paginator = client.get_paginator('describe_transit_gateway_peering_attachments')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_transit_gateway_peering_attachments()](#EC2.Client.describe_transit_gateway_peering_attachments "EC2.Client.describe_transit_gateway_peering_attachments").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeTransitGatewayPeeringAttachments)

**Request Syntax**

response_iterator = paginator.paginate(
    TransitGatewayAttachmentIds=[
        'string',
    ],
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    DryRun=True|False,
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **TransitGatewayAttachmentIds** (_list_) --

    One or more IDs of the transit gateway peering attachments.

    - _(string) --_
- **Filters** (_list_) --

    One or more filters. The possible values are:

    - transit-gateway-attachment-id - The ID of the transit gateway attachment.
    - local-owner-id - The ID of your AWS account.
    - remote-owner-id - The ID of the AWS account in the remote Region that owns the transit gateway.
    - state - The state of the peering attachment. Valid values are available | deleted | deleting | failed | failing | initiatingRequest | modifying | pendingAcceptance | pending | rollingBack | rejected | rejecting ).
    - tag :<key> - The key/value combination of a tag assigned to the resource. Use the tag key in the filter name and the tag value as the filter value. For example, to find all resources that have a tag with the key Owner and the value TeamA , specify tag:Owner for the filter name and TeamA for the filter value.
    - tag-key - The key of a tag assigned to the resource. Use this filter to find all resources that have a tag with a specific key, regardless of the tag value.
    - transit-gateway-id - The ID of the transit gateway.

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'TransitGatewayPeeringAttachments': [
        {
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
        },
    ],

}

**Response Structure**

- _(dict) --_

    - **TransitGatewayPeeringAttachments** _(list) --_

        The transit gateway peering attachments.

        - _(dict) --_

            Describes the transit gateway peering attachment.

            - **TransitGatewayAttachmentId** _(string) --_

                The ID of the transit gateway peering attachment.

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

            - **State** _(string) --_

                The state of the transit gateway peering attachment. Note that the initiating state has been deprecated.

            - **CreationTime** _(datetime) --_

                The time the transit gateway peering attachment was created.

            - **Tags** _(list) --_

                The tags for the transit gateway peering attachment.

                - _(dict) --_

                    Describes a tag.

                    - **Key** _(string) --_

                        The key of the tag.

                        Constraints: Tag keys are case-sensitive and accept a maximum of 127 Unicode characters. May not begin with aws: .

                    - **Value** _(string) --_

                        The value of the tag.

                        Constraints: Tag values are case-sensitive and accept a maximum of 255 Unicode characters.


_class_ EC2.Paginator.DescribeTransitGatewayRouteTables

paginator = client.get_paginator('describe_transit_gateway_route_tables')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_transit_gateway_route_tables()](#EC2.Client.describe_transit_gateway_route_tables "EC2.Client.describe_transit_gateway_route_tables").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeTransitGatewayRouteTables)

**Request Syntax**

response_iterator = paginator.paginate(
    TransitGatewayRouteTableIds=[
        'string',
    ],
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    DryRun=True|False,
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **TransitGatewayRouteTableIds** (_list_) --

    The IDs of the transit gateway route tables.

    - _(string) --_
- **Filters** (_list_) --

    One or more filters. The possible values are:

    - default-association-route-table - Indicates whether this is the default association route table for the transit gateway (true | false ).
    - default-propagation-route-table - Indicates whether this is the default propagation route table for the transit gateway (true | false ).
    - state - The state of the route table (available | deleting | deleted | pending ).
    - transit-gateway-id - The ID of the transit gateway.
    - transit-gateway-route-table-id - The ID of the transit gateway route table.

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'TransitGatewayRouteTables': [
        {
            'TransitGatewayRouteTableId': 'string',
            'TransitGatewayId': 'string',
            'State': 'pending'|'available'|'deleting'|'deleted',
            'DefaultAssociationRouteTable': True|False,
            'DefaultPropagationRouteTable': True|False,
            'CreationTime': datetime(2015, 1, 1),
            'Tags': [
                {
                    'Key': 'string',
                    'Value': 'string'
                },
            ]
        },
    ],

}

**Response Structure**

- _(dict) --_

    - **TransitGatewayRouteTables** _(list) --_

        Information about the transit gateway route tables.

        - _(dict) --_

            Describes a transit gateway route table.

            - **TransitGatewayRouteTableId** _(string) --_

                The ID of the transit gateway route table.

            - **TransitGatewayId** _(string) --_

                The ID of the transit gateway.

            - **State** _(string) --_

                The state of the transit gateway route table.

            - **DefaultAssociationRouteTable** _(boolean) --_

                Indicates whether this is the default association route table for the transit gateway.

            - **DefaultPropagationRouteTable** _(boolean) --_

                Indicates whether this is the default propagation route table for the transit gateway.

            - **CreationTime** _(datetime) --_

                The creation time.

            - **Tags** _(list) --_

                Any tags assigned to the route table.

                - _(dict) --_

                    Describes a tag.

                    - **Key** _(string) --_

                        The key of the tag.

                        Constraints: Tag keys are case-sensitive and accept a maximum of 127 Unicode characters. May not begin with aws: .

                    - **Value** _(string) --_

                        The value of the tag.

                        Constraints: Tag values are case-sensitive and accept a maximum of 255 Unicode characters.


_class_ EC2.Paginator.DescribeTransitGatewayVpcAttachments

paginator = client.get_paginator('describe_transit_gateway_vpc_attachments')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_transit_gateway_vpc_attachments()](#EC2.Client.describe_transit_gateway_vpc_attachments "EC2.Client.describe_transit_gateway_vpc_attachments").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeTransitGatewayVpcAttachments)

**Request Syntax**

response_iterator = paginator.paginate(
    TransitGatewayAttachmentIds=[
        'string',
    ],
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    DryRun=True|False,
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **TransitGatewayAttachmentIds** (_list_) --

    The IDs of the attachments.

    - _(string) --_
- **Filters** (_list_) --

    One or more filters. The possible values are:

    - state - The state of the attachment. Valid values are available | deleted | deleting | failed | failing | initiatingRequest | modifying | pendingAcceptance | pending | rollingBack | rejected | rejecting .
    - transit-gateway-attachment-id - The ID of the attachment.
    - transit-gateway-id - The ID of the transit gateway.
    - vpc-id - The ID of the VPC.

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'TransitGatewayVpcAttachments': [
        {
            'TransitGatewayAttachmentId': 'string',
            'TransitGatewayId': 'string',
            'VpcId': 'string',
            'VpcOwnerId': 'string',
            'State': 'initiating'|'initiatingRequest'|'pendingAcceptance'|'rollingBack'|'pending'|'available'|'modifying'|'deleting'|'deleted'|'failed'|'rejected'|'rejecting'|'failing',
            'SubnetIds': [
                'string',
            ],
            'CreationTime': datetime(2015, 1, 1),
            'Options': {
                'DnsSupport': 'enable'|'disable',
                'Ipv6Support': 'enable'|'disable',
                'ApplianceModeSupport': 'enable'|'disable'
            },
            'Tags': [
                {
                    'Key': 'string',
                    'Value': 'string'
                },
            ]
        },
    ],

}

**Response Structure**

- _(dict) --_

    - **TransitGatewayVpcAttachments** _(list) --_

        Information about the VPC attachments.

        - _(dict) --_

            Describes a VPC attachment.

            - **TransitGatewayAttachmentId** _(string) --_

                The ID of the attachment.

            - **TransitGatewayId** _(string) --_

                The ID of the transit gateway.

            - **VpcId** _(string) --_

                The ID of the VPC.

            - **VpcOwnerId** _(string) --_

                The ID of the AWS account that owns the VPC.

            - **State** _(string) --_

                The state of the VPC attachment. Note that the initiating state has been deprecated.

            - **SubnetIds** _(list) --_

                The IDs of the subnets.

                - _(string) --_
            - **CreationTime** _(datetime) --_

                The creation time.

            - **Options** _(dict) --_

                The VPC attachment options.

                - **DnsSupport** _(string) --_

                    Indicates whether DNS support is enabled.

                - **Ipv6Support** _(string) --_

                    Indicates whether IPv6 support is disabled.

                - **ApplianceModeSupport** _(string) --_

                    Indicates whether appliance mode support is enabled.

            - **Tags** _(list) --_

                The tags for the VPC attachment.

                - _(dict) --_

                    Describes a tag.

                    - **Key** _(string) --_

                        The key of the tag.

                        Constraints: Tag keys are case-sensitive and accept a maximum of 127 Unicode characters. May not begin with aws: .

                    - **Value** _(string) --_

                        The value of the tag.

                        Constraints: Tag values are case-sensitive and accept a maximum of 255 Unicode characters.


_class_ EC2.Paginator.DescribeTransitGateways

paginator = client.get_paginator('describe_transit_gateways')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_transit_gateways()](#EC2.Client.describe_transit_gateways "EC2.Client.describe_transit_gateways").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeTransitGateways)

**Request Syntax**

response_iterator = paginator.paginate(
    TransitGatewayIds=[
        'string',
    ],
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    DryRun=True|False,
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **TransitGatewayIds** (_list_) --

    The IDs of the transit gateways.

    - _(string) --_
- **Filters** (_list_) --

    One or more filters. The possible values are:

    - options.propagation-default-route-table-id - The ID of the default propagation route table.
    - options.amazon-side-asn - The private ASN for the Amazon side of a BGP session.
    - options.association-default-route-table-id - The ID of the default association route table.
    - options.auto-accept-shared-attachments - Indicates whether there is automatic acceptance of attachment requests (enable | disable ).
    - options.default-route-table-association - Indicates whether resource attachments are automatically associated with the default association route table (enable | disable ).
    - options.default-route-table-propagation - Indicates whether resource attachments automatically propagate routes to the default propagation route table (enable | disable ).
    - options.dns-support - Indicates whether DNS support is enabled (enable | disable ).
    - options.vpn-ecmp-support - Indicates whether Equal Cost Multipath Protocol support is enabled (enable | disable ).
    - owner-id - The ID of the AWS account that owns the transit gateway.
    - state - The state of the transit gateway (available | deleted | deleting | modifying | pending ).
    - transit-gateway-id - The ID of the transit gateway.

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'TransitGateways': [
        {
            'TransitGatewayId': 'string',
            'TransitGatewayArn': 'string',
            'State': 'pending'|'available'|'modifying'|'deleting'|'deleted',
            'OwnerId': 'string',
            'Description': 'string',
            'CreationTime': datetime(2015, 1, 1),
            'Options': {
                'AmazonSideAsn': 123,
                'TransitGatewayCidrBlocks': [
                    'string',
                ],
                'AutoAcceptSharedAttachments': 'enable'|'disable',
                'DefaultRouteTableAssociation': 'enable'|'disable',
                'AssociationDefaultRouteTableId': 'string',
                'DefaultRouteTablePropagation': 'enable'|'disable',
                'PropagationDefaultRouteTableId': 'string',
                'VpnEcmpSupport': 'enable'|'disable',
                'DnsSupport': 'enable'|'disable',
                'MulticastSupport': 'enable'|'disable'
            },
            'Tags': [
                {
                    'Key': 'string',
                    'Value': 'string'
                },
            ]
        },
    ],

}

**Response Structure**

- _(dict) --_

    - **TransitGateways** _(list) --_

        Information about the transit gateways.

        - _(dict) --_

            Describes a transit gateway.

            - **TransitGatewayId** _(string) --_

                The ID of the transit gateway.

            - **TransitGatewayArn** _(string) --_

                The Amazon Resource Name (ARN) of the transit gateway.

            - **State** _(string) --_

                The state of the transit gateway.

            - **OwnerId** _(string) --_

                The ID of the AWS account ID that owns the transit gateway.

            - **Description** _(string) --_

                The description of the transit gateway.

            - **CreationTime** _(datetime) --_

                The creation time.

            - **Options** _(dict) --_

                The transit gateway options.

                - **AmazonSideAsn** _(integer) --_

                    A private Autonomous System Number (ASN) for the Amazon side of a BGP session. The range is 64512 to 65534 for 16-bit ASNs and 4200000000 to 4294967294 for 32-bit ASNs.

                - **TransitGatewayCidrBlocks** _(list) --_

                    The transit gateway CIDR blocks.

                    - _(string) --_
                - **AutoAcceptSharedAttachments** _(string) --_

                    Indicates whether attachment requests are automatically accepted.

                - **DefaultRouteTableAssociation** _(string) --_

                    Indicates whether resource attachments are automatically associated with the default association route table.

                - **AssociationDefaultRouteTableId** _(string) --_

                    The ID of the default association route table.

                - **DefaultRouteTablePropagation** _(string) --_

                    Indicates whether resource attachments automatically propagate routes to the default propagation route table.

                - **PropagationDefaultRouteTableId** _(string) --_

                    The ID of the default propagation route table.

                - **VpnEcmpSupport** _(string) --_

                    Indicates whether Equal Cost Multipath Protocol support is enabled.

                - **DnsSupport** _(string) --_

                    Indicates whether DNS support is enabled.

                - **MulticastSupport** _(string) --_

                    Indicates whether multicast is enabled on the transit gateway

            - **Tags** _(list) --_

                The tags for the transit gateway.

                - _(dict) --_

                    Describes a tag.

                    - **Key** _(string) --_

                        The key of the tag.

                        Constraints: Tag keys are case-sensitive and accept a maximum of 127 Unicode characters. May not begin with aws: .

                    - **Value** _(string) --_

                        The value of the tag.

                        Constraints: Tag values are case-sensitive and accept a maximum of 255 Unicode characters.


_class_ EC2.Paginator.DescribeVolumeStatus

paginator = client.get_paginator('describe_volume_status')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_volume_status()](#EC2.Client.describe_volume_status "EC2.Client.describe_volume_status").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeVolumeStatus)

**Request Syntax**

response_iterator = paginator.paginate(
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    VolumeIds=[
        'string',
    ],
    DryRun=True|False,
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **Filters** (_list_) --

    The filters.

    - action.code - The action code for the event (for example, enable-volume-io ).
    - action.description - A description of the action.
    - action.event-id - The event ID associated with the action.
    - availability-zone - The Availability Zone of the instance.
    - event.description - A description of the event.
    - event.event-id - The event ID.
    - event.event-type - The event type (for io-enabled : passed | failed ; for io-performance : io-performance:degraded | io-performance:severely-degraded | io-performance:stalled ).
    - event.not-after - The latest end time for the event.
    - event.not-before - The earliest start time for the event.
    - volume-status.details-name - The cause for volume-status.status (io-enabled | io-performance ).
    - volume-status.details-status - The status of volume-status.details-name (for io-enabled : passed | failed ; for io-performance : normal | degraded | severely-degraded | stalled ).
    - volume-status.status - The status of the volume (ok | impaired | warning | insufficient-data ).

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **VolumeIds** (_list_) --

    The IDs of the volumes.

    Default: Describes all your volumes.

    - _(string) --_
- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'VolumeStatuses': [
        {
            'Actions': [
                {
                    'Code': 'string',
                    'Description': 'string',
                    'EventId': 'string',
                    'EventType': 'string'
                },
            ],
            'AvailabilityZone': 'string',
            'OutpostArn': 'string',
            'Events': [
                {
                    'Description': 'string',
                    'EventId': 'string',
                    'EventType': 'string',
                    'NotAfter': datetime(2015, 1, 1),
                    'NotBefore': datetime(2015, 1, 1),
                    'InstanceId': 'string'
                },
            ],
            'VolumeId': 'string',
            'VolumeStatus': {
                'Details': [
                    {
                        'Name': 'io-enabled'|'io-performance',
                        'Status': 'string'
                    },
                ],
                'Status': 'ok'|'impaired'|'insufficient-data'
            },
            'AttachmentStatuses': [
                {
                    'IoPerformance': 'string',
                    'InstanceId': 'string'
                },
            ]
        },
    ]
}

**Response Structure**

- _(dict) --_

    - **VolumeStatuses** _(list) --_

        Information about the status of the volumes.

        - _(dict) --_

            Describes the volume status.

            - **Actions** _(list) --_

                The details of the operation.

                - _(dict) --_

                    Describes a volume status operation code.

                    - **Code** _(string) --_

                        The code identifying the operation, for example, enable-volume-io .

                    - **Description** _(string) --_

                        A description of the operation.

                    - **EventId** _(string) --_

                        The ID of the event associated with this operation.

                    - **EventType** _(string) --_

                        The event type associated with this operation.

            - **AvailabilityZone** _(string) --_

                The Availability Zone of the volume.

            - **OutpostArn** _(string) --_

                The Amazon Resource Name (ARN) of the Outpost.

            - **Events** _(list) --_

                A list of events associated with the volume.

                - _(dict) --_

                    Describes a volume status event.

                    - **Description** _(string) --_

                        A description of the event.

                    - **EventId** _(string) --_

                        The ID of this event.

                    - **EventType** _(string) --_

                        The type of this event.

                    - **NotAfter** _(datetime) --_

                        The latest end time of the event.

                    - **NotBefore** _(datetime) --_

                        The earliest start time of the event.

                    - **InstanceId** _(string) --_

                        The ID of the instance associated with the event.

            - **VolumeId** _(string) --_

                The volume ID.

            - **VolumeStatus** _(dict) --_

                The volume status.

                - **Details** _(list) --_

                    The details of the volume status.

                    - _(dict) --_

                        Describes a volume status.

                        - **Name** _(string) --_

                            The name of the volume status.

                        - **Status** _(string) --_

                            The intended status of the volume status.

                - **Status** _(string) --_

                    The status of the volume.

            - **AttachmentStatuses** _(list) --_

                Information about the instances to which the volume is attached.

                - _(dict) --_

                    Information about the instances to which the volume is attached.

                    - **IoPerformance** _(string) --_

                        The maximum IOPS supported by the attached instance.

                    - **InstanceId** _(string) --_

                        The ID of the attached instance.


_class_ EC2.Paginator.DescribeVolumes

paginator = client.get_paginator('describe_volumes')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_volumes()](#EC2.Client.describe_volumes "EC2.Client.describe_volumes").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeVolumes)

**Request Syntax**

response_iterator = paginator.paginate(
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    VolumeIds=[
        'string',
    ],
    DryRun=True|False,
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **Filters** (_list_) --

    The filters.

    - attachment.attach-time - The time stamp when the attachment initiated.
    - attachment.delete-on-termination - Whether the volume is deleted on instance termination.
    - attachment.device - The device name specified in the block device mapping (for example, /dev/sda1 ).
    - attachment.instance-id - The ID of the instance the volume is attached to.
    - attachment.status - The attachment state (attaching | attached | detaching ).
    - availability-zone - The Availability Zone in which the volume was created.
    - create-time - The time stamp when the volume was created.
    - encrypted - Indicates whether the volume is encrypted (true | false )
    - multi-attach-enabled - Indicates whether the volume is enabled for Multi-Attach (true | false )
    - fast-restored - Indicates whether the volume was created from a snapshot that is enabled for fast snapshot restore (true | false ).
    - size - The size of the volume, in GiB.
    - snapshot-id - The snapshot from which the volume was created.
    - status - The state of the volume (creating | available | in-use | deleting | deleted | error ).
    - tag :<key> - The key/value combination of a tag assigned to the resource. Use the tag key in the filter name and the tag value as the filter value. For example, to find all resources that have a tag with the key Owner and the value TeamA , specify tag:Owner for the filter name and TeamA for the filter value.
    - tag-key - The key of a tag assigned to the resource. Use this filter to find all resources assigned a tag with a specific key, regardless of the tag value.
    - volume-id - The volume ID.
    - volume-type - The Amazon EBS volume type (gp2 | gp3 | io1 | io2 | st1 | sc1 | standard )

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **VolumeIds** (_list_) --

    The volume IDs.

    - _(string) --_
- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'Volumes': [
        {
            'Attachments': [
                {
                    'AttachTime': datetime(2015, 1, 1),
                    'Device': 'string',
                    'InstanceId': 'string',
                    'State': 'attaching'|'attached'|'detaching'|'detached'|'busy',
                    'VolumeId': 'string',
                    'DeleteOnTermination': True|False
                },
            ],
            'AvailabilityZone': 'string',
            'CreateTime': datetime(2015, 1, 1),
            'Encrypted': True|False,
            'KmsKeyId': 'string',
            'OutpostArn': 'string',
            'Size': 123,
            'SnapshotId': 'string',
            'State': 'creating'|'available'|'in-use'|'deleting'|'deleted'|'error',
            'VolumeId': 'string',
            'Iops': 123,
            'Tags': [
                {
                    'Key': 'string',
                    'Value': 'string'
                },
            ],
            'VolumeType': 'standard'|'io1'|'io2'|'gp2'|'sc1'|'st1'|'gp3',
            'FastRestored': True|False,
            'MultiAttachEnabled': True|False,
            'Throughput': 123
        },
    ],

}

**Response Structure**

- _(dict) --_

    - **Volumes** _(list) --_

        Information about the volumes.

        - _(dict) --_

            Describes a volume.

            - **Attachments** _(list) --_

                Information about the volume attachments.

                - _(dict) --_

                    Describes volume attachment details.

                    - **AttachTime** _(datetime) --_

                        The time stamp when the attachment initiated.

                    - **Device** _(string) --_

                        The device name.

                    - **InstanceId** _(string) --_

                        The ID of the instance.

                    - **State** _(string) --_

                        The attachment state of the volume.

                    - **VolumeId** _(string) --_

                        The ID of the volume.

                    - **DeleteOnTermination** _(boolean) --_

                        Indicates whether the EBS volume is deleted on instance termination.

            - **AvailabilityZone** _(string) --_

                The Availability Zone for the volume.

            - **CreateTime** _(datetime) --_

                The time stamp when volume creation was initiated.

            - **Encrypted** _(boolean) --_

                Indicates whether the volume is encrypted.

            - **KmsKeyId** _(string) --_

                The Amazon Resource Name (ARN) of the AWS Key Management Service (AWS KMS) customer master key (CMK) that was used to protect the volume encryption key for the volume.

            - **OutpostArn** _(string) --_

                The Amazon Resource Name (ARN) of the Outpost.

            - **Size** _(integer) --_

                The size of the volume, in GiBs.

            - **SnapshotId** _(string) --_

                The snapshot from which the volume was created, if applicable.

            - **State** _(string) --_

                The volume state.

            - **VolumeId** _(string) --_

                The ID of the volume.

            - **Iops** _(integer) --_

                The number of I/O operations per second (IOPS). For gp3 , io1 , and io2 volumes, this represents the number of IOPS that are provisioned for the volume. For gp2 volumes, this represents the baseline performance of the volume and the rate at which the volume accumulates I/O credits for bursting.

            - **Tags** _(list) --_

                Any tags assigned to the volume.

                - _(dict) --_

                    Describes a tag.

                    - **Key** _(string) --_

                        The key of the tag.

                        Constraints: Tag keys are case-sensitive and accept a maximum of 127 Unicode characters. May not begin with aws: .

                    - **Value** _(string) --_

                        The value of the tag.

                        Constraints: Tag values are case-sensitive and accept a maximum of 255 Unicode characters.

            - **VolumeType** _(string) --_

                The volume type.

            - **FastRestored** _(boolean) --_

                Indicates whether the volume was created using fast snapshot restore.

            - **MultiAttachEnabled** _(boolean) --_

                Indicates whether Amazon EBS Multi-Attach is enabled.

            - **Throughput** _(integer) --_

                The throughput that the volume supports, in MiB/s.


_class_ EC2.Paginator.DescribeVolumesModifications

paginator = client.get_paginator('describe_volumes_modifications')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_volumes_modifications()](#EC2.Client.describe_volumes_modifications "EC2.Client.describe_volumes_modifications").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeVolumesModifications)

**Request Syntax**

response_iterator = paginator.paginate(
    DryRun=True|False,
    VolumeIds=[
        'string',
    ],
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **VolumeIds** (_list_) --

    The IDs of the volumes.

    - _(string) --_
- **Filters** (_list_) --

    The filters.

    - modification-state - The current modification state (modifying | optimizing | completed | failed).
    - original-iops - The original IOPS rate of the volume.
    - original-size - The original size of the volume, in GiB.
    - original-volume-type - The original volume type of the volume (standard | io1 | io2 | gp2 | sc1 | st1).
    - originalMultiAttachEnabled - Indicates whether Multi-Attach support was enabled (true | false).
    - start-time - The modification start time.
    - target-iops - The target IOPS rate of the volume.
    - target-size - The target size of the volume, in GiB.
    - target-volume-type - The target volume type of the volume (standard | io1 | io2 | gp2 | sc1 | st1).
    - targetMultiAttachEnabled - Indicates whether Multi-Attach support is to be enabled (true | false).
    - volume-id - The ID of the volume.

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'VolumesModifications': [
        {
            'VolumeId': 'string',
            'ModificationState': 'modifying'|'optimizing'|'completed'|'failed',
            'StatusMessage': 'string',
            'TargetSize': 123,
            'TargetIops': 123,
            'TargetVolumeType': 'standard'|'io1'|'io2'|'gp2'|'sc1'|'st1'|'gp3',
            'TargetThroughput': 123,
            'TargetMultiAttachEnabled': True|False,
            'OriginalSize': 123,
            'OriginalIops': 123,
            'OriginalVolumeType': 'standard'|'io1'|'io2'|'gp2'|'sc1'|'st1'|'gp3',
            'OriginalThroughput': 123,
            'OriginalMultiAttachEnabled': True|False,
            'Progress': 123,
            'StartTime': datetime(2015, 1, 1),
            'EndTime': datetime(2015, 1, 1)
        },
    ],

}

**Response Structure**

- _(dict) --_

    - **VolumesModifications** _(list) --_

        Information about the volume modifications.

        - _(dict) --_

            Describes the modification status of an EBS volume.

            If the volume has never been modified, some element values will be null.

            - **VolumeId** _(string) --_

                The ID of the volume.

            - **ModificationState** _(string) --_

                The current modification state. The modification state is null for unmodified volumes.

            - **StatusMessage** _(string) --_

                A status message about the modification progress or failure.

            - **TargetSize** _(integer) --_

                The target size of the volume, in GiB.

            - **TargetIops** _(integer) --_

                The target IOPS rate of the volume.

            - **TargetVolumeType** _(string) --_

                The target EBS volume type of the volume.

            - **TargetThroughput** _(integer) --_

                The target throughput of the volume, in MiB/s.

            - **TargetMultiAttachEnabled** _(boolean) --_

                The target setting for Amazon EBS Multi-Attach.

            - **OriginalSize** _(integer) --_

                The original size of the volume, in GiB.

            - **OriginalIops** _(integer) --_

                The original IOPS rate of the volume.

            - **OriginalVolumeType** _(string) --_

                The original EBS volume type of the volume.

            - **OriginalThroughput** _(integer) --_

                The original throughput of the volume, in MiB/s.

            - **OriginalMultiAttachEnabled** _(boolean) --_

                The original setting for Amazon EBS Multi-Attach.

            - **Progress** _(integer) --_

                The modification progress, from 0 to 100 percent complete.

            - **StartTime** _(datetime) --_

                The modification start time.

            - **EndTime** _(datetime) --_

                The modification completion or failure time.


_class_ EC2.Paginator.DescribeVpcClassicLinkDnsSupport

paginator = client.get_paginator('describe_vpc_classic_link_dns_support')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_vpc_classic_link_dns_support()](#EC2.Client.describe_vpc_classic_link_dns_support "EC2.Client.describe_vpc_classic_link_dns_support").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeVpcClassicLinkDnsSupport)

**Request Syntax**

response_iterator = paginator.paginate(
    VpcIds=[
        'string',
    ],
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **VpcIds** (_list_) --

    One or more VPC IDs.

    - _(string) --_
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'Vpcs': [
        {
            'ClassicLinkDnsSupported': True|False,
            'VpcId': 'string'
        },
    ]
}

**Response Structure**

- _(dict) --_

    - **Vpcs** _(list) --_

        Information about the ClassicLink DNS support status of the VPCs.

        - _(dict) --_

            Describes the ClassicLink DNS support status of a VPC.

            - **ClassicLinkDnsSupported** _(boolean) --_

                Indicates whether ClassicLink DNS support is enabled for the VPC.

            - **VpcId** _(string) --_

                The ID of the VPC.


_class_ EC2.Paginator.DescribeVpcEndpointConnectionNotificationsdefinition")

paginator = client.get_paginator('describe_vpc_endpoint_connection_notifications')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_vpc_endpoint_connection_notifications()](#EC2.Client.describe_vpc_endpoint_connection_notifications "EC2.Client.describe_vpc_endpoint_connection_notifications").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeVpcEndpointConnectionNotifications)

**Request Syntax**

response_iterator = paginator.paginate(
    DryRun=True|False,
    ConnectionNotificationId='string',
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **ConnectionNotificationId** (_string_) -- The ID of the notification.
- **Filters** (_list_) --

    One or more filters.

    - connection-notification-arn - The ARN of the SNS topic for the notification.
    - connection-notification-id - The ID of the notification.
    - connection-notification-state - The state of the notification (Enabled | Disabled ).
    - connection-notification-type - The type of notification (Topic ).
    - service-id - The ID of the endpoint service.
    - vpc-endpoint-id - The ID of the VPC endpoint.

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'ConnectionNotificationSet': [
        {
            'ConnectionNotificationId': 'string',
            'ServiceId': 'string',
            'VpcEndpointId': 'string',
            'ConnectionNotificationType': 'Topic',
            'ConnectionNotificationArn': 'string',
            'ConnectionEvents': [
                'string',
            ],
            'ConnectionNotificationState': 'Enabled'|'Disabled'
        },
    ],

}

**Response Structure**

- _(dict) --_

    - **ConnectionNotificationSet** _(list) --_

        One or more notifications.

        - _(dict) --_

            Describes a connection notification for a VPC endpoint or VPC endpoint service.

            - **ConnectionNotificationId** _(string) --_

                The ID of the notification.

            - **ServiceId** _(string) --_

                The ID of the endpoint service.

            - **VpcEndpointId** _(string) --_

                The ID of the VPC endpoint.

            - **ConnectionNotificationType** _(string) --_

                The type of notification.

            - **ConnectionNotificationArn** _(string) --_

                The ARN of the SNS topic for the notification.

            - **ConnectionEvents** _(list) --_

                The events for the notification. Valid values are Accept , Connect , Delete , and Reject .

                - _(string) --_
            - **ConnectionNotificationState** _(string) --_

                The state of the notification.


_class_ EC2.Paginator.DescribeVpcEndpointConnections

paginator = client.get_paginator('describe_vpc_endpoint_connections')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_vpc_endpoint_connections()](#EC2.Client.describe_vpc_endpoint_connections "EC2.Client.describe_vpc_endpoint_connections").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeVpcEndpointConnections)

**Request Syntax**

response_iterator = paginator.paginate(
    DryRun=True|False,
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **Filters** (_list_) --

    One or more filters.

    - service-id - The ID of the service.
    - vpc-endpoint-owner - The AWS account number of the owner of the endpoint.
    - vpc-endpoint-state - The state of the endpoint (pendingAcceptance | pending | available | deleting | deleted | rejected | failed ).
    - vpc-endpoint-id - The ID of the endpoint.

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'VpcEndpointConnections': [
        {
            'ServiceId': 'string',
            'VpcEndpointId': 'string',
            'VpcEndpointOwner': 'string',
            'VpcEndpointState': 'PendingAcceptance'|'Pending'|'Available'|'Deleting'|'Deleted'|'Rejected'|'Failed'|'Expired',
            'CreationTimestamp': datetime(2015, 1, 1),
            'DnsEntries': [
                {
                    'DnsName': 'string',
                    'HostedZoneId': 'string'
                },
            ],
            'NetworkLoadBalancerArns': [
                'string',
            ],
            'GatewayLoadBalancerArns': [
                'string',
            ]
        },
    ],

}

**Response Structure**

- _(dict) --_

    - **VpcEndpointConnections** _(list) --_

        Information about one or more VPC endpoint connections.

        - _(dict) --_

            Describes a VPC endpoint connection to a service.

            - **ServiceId** _(string) --_

                The ID of the service to which the endpoint is connected.

            - **VpcEndpointId** _(string) --_

                The ID of the VPC endpoint.

            - **VpcEndpointOwner** _(string) --_

                The AWS account ID of the owner of the VPC endpoint.

            - **VpcEndpointState** _(string) --_

                The state of the VPC endpoint.

            - **CreationTimestamp** _(datetime) --_

                The date and time that the VPC endpoint was created.

            - **DnsEntries** _(list) --_

                The DNS entries for the VPC endpoint.

                - _(dict) --_

                    Describes a DNS entry.

                    - **DnsName** _(string) --_

                        The DNS name.

                    - **HostedZoneId** _(string) --_

                        The ID of the private hosted zone.

            - **NetworkLoadBalancerArns** _(list) --_

                The Amazon Resource Names (ARNs) of the network load balancers for the service.

                - _(string) --_
            - **GatewayLoadBalancerArns** _(list) --_

                The Amazon Resource Names (ARNs) of the Gateway Load Balancers for the service.

                - _(string) --_

_class_ EC2.Paginator.DescribeVpcEndpointServiceConfigurationsdefinition")

paginator = client.get_paginator('describe_vpc_endpoint_service_configurations')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_vpc_endpoint_service_configurations()](#EC2.Client.describe_vpc_endpoint_service_configurations "EC2.Client.describe_vpc_endpoint_service_configurations").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeVpcEndpointServiceConfigurations)

**Request Syntax**

response_iterator = paginator.paginate(
    DryRun=True|False,
    ServiceIds=[
        'string',
    ],
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **ServiceIds** (_list_) --

    The IDs of one or more services.

    - _(string) --_
- **Filters** (_list_) --

    One or more filters.

    - service-name - The name of the service.
    - service-id - The ID of the service.
    - service-state - The state of the service (Pending | Available | Deleting | Deleted | Failed ).
    - tag :<key> - The key/value combination of a tag assigned to the resource. Use the tag key in the filter name and the tag value as the filter value. For example, to find all resources that have a tag with the key Owner and the value TeamA , specify tag:Owner for the filter name and TeamA for the filter value.
    - tag-key - The key of a tag assigned to the resource. Use this filter to find all resources assigned a tag with a specific key, regardless of the tag value.

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'ServiceConfigurations': [
        {
            'ServiceType': [
                {
                    'ServiceType': 'Interface'|'Gateway'|'GatewayLoadBalancer'
                },
            ],
            'ServiceId': 'string',
            'ServiceName': 'string',
            'ServiceState': 'Pending'|'Available'|'Deleting'|'Deleted'|'Failed',
            'AvailabilityZones': [
                'string',
            ],
            'AcceptanceRequired': True|False,
            'ManagesVpcEndpoints': True|False,
            'NetworkLoadBalancerArns': [
                'string',
            ],
            'GatewayLoadBalancerArns': [
                'string',
            ],
            'BaseEndpointDnsNames': [
                'string',
            ],
            'PrivateDnsName': 'string',
            'PrivateDnsNameConfiguration': {
                'State': 'pendingVerification'|'verified'|'failed',
                'Type': 'string',
                'Value': 'string',
                'Name': 'string'
            },
            'Tags': [
                {
                    'Key': 'string',
                    'Value': 'string'
                },
            ]
        },
    ],

}

**Response Structure**

- _(dict) --_

    - **ServiceConfigurations** _(list) --_

        Information about one or more services.

        - _(dict) --_

            Describes a service configuration for a VPC endpoint service.

            - **ServiceType** _(list) --_

                The type of service.

                - _(dict) --_

                    Describes the type of service for a VPC endpoint.

                    - **ServiceType** _(string) --_

                        The type of service.

            - **ServiceId** _(string) --_

                The ID of the service.

            - **ServiceName** _(string) --_

                The name of the service.

            - **ServiceState** _(string) --_

                The service state.

            - **AvailabilityZones** _(list) --_

                The Availability Zones in which the service is available.

                - _(string) --_
            - **AcceptanceRequired** _(boolean) --_

                Indicates whether requests from other AWS accounts to create an endpoint to the service must first be accepted.

            - **ManagesVpcEndpoints** _(boolean) --_

                Indicates whether the service manages its VPC endpoints. Management of the service VPC endpoints using the VPC endpoint API is restricted.

            - **NetworkLoadBalancerArns** _(list) --_

                The Amazon Resource Names (ARNs) of the Network Load Balancers for the service.

                - _(string) --_
            - **GatewayLoadBalancerArns** _(list) --_

                The Amazon Resource Names (ARNs) of the Gateway Load Balancers for the service.

                - _(string) --_
            - **BaseEndpointDnsNames** _(list) --_

                The DNS names for the service.

                - _(string) --_
            - **PrivateDnsName** _(string) --_

                The private DNS name for the service.

            - **PrivateDnsNameConfiguration** _(dict) --_

                Information about the endpoint service private DNS name configuration.

                - **State** _(string) --_

                    The verification state of the VPC endpoint service.

                    >Consumers of the endpoint service can use the private name only when the state is verified .

                - **Type** _(string) --_

                    The endpoint service verification type, for example TXT.

                - **Value** _(string) --_

                    The value the service provider adds to the private DNS name domain record before verification.

                - **Name** _(string) --_

                    The name of the record subdomain the service provider needs to create. The service provider adds the value text to the name .

            - **Tags** _(list) --_

                Any tags assigned to the service.

                - _(dict) --_

                    Describes a tag.

                    - **Key** _(string) --_

                        The key of the tag.

                        Constraints: Tag keys are case-sensitive and accept a maximum of 127 Unicode characters. May not begin with aws: .

                    - **Value** _(string) --_

                        The value of the tag.

                        Constraints: Tag values are case-sensitive and accept a maximum of 255 Unicode characters.


_class_ EC2.Paginator.DescribeVpcEndpointServicePermissions

paginator = client.get_paginator('describe_vpc_endpoint_service_permissions')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_vpc_endpoint_service_permissions()](#EC2.Client.describe_vpc_endpoint_service_permissions "EC2.Client.describe_vpc_endpoint_service_permissions").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeVpcEndpointServicePermissions)

**Request Syntax**

response_iterator = paginator.paginate(
    DryRun=True|False,
    ServiceId='string',
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **ServiceId** (_string_) --

    **[REQUIRED]**

    The ID of the service.

- **Filters** (_list_) --

    One or more filters.

    - principal - The ARN of the principal.
    - principal-type - The principal type (All | Service | OrganizationUnit | Account | User | Role ).

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'AllowedPrincipals': [
        {
            'PrincipalType': 'All'|'Service'|'OrganizationUnit'|'Account'|'User'|'Role',
            'Principal': 'string'
        },
    ],

}

**Response Structure**

- _(dict) --_

    - **AllowedPrincipals** _(list) --_

        Information about one or more allowed principals.

        - _(dict) --_

            Describes a principal.

            - **PrincipalType** _(string) --_

                The type of principal.

            - **Principal** _(string) --_

                The Amazon Resource Name (ARN) of the principal.


_class_ EC2.Paginator.DescribeVpcEndpointServices

paginator = client.get_paginator('describe_vpc_endpoint_services')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_vpc_endpoint_services()](#EC2.Client.describe_vpc_endpoint_services "EC2.Client.describe_vpc_endpoint_services").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeVpcEndpointServices)

**Request Syntax**

response_iterator = paginator.paginate(
    DryRun=True|False,
    ServiceNames=[
        'string',
    ],
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **ServiceNames** (_list_) --

    One or more service names.

    - _(string) --_
- **Filters** (_list_) --

    One or more filters.

    - service-name - The name of the service.
    - tag :<key> - The key/value combination of a tag assigned to the resource. Use the tag key in the filter name and the tag value as the filter value. For example, to find all resources that have a tag with the key Owner and the value TeamA , specify tag:Owner for the filter name and TeamA for the filter value.
    - tag-key - The key of a tag assigned to the resource. Use this filter to find all resources assigned a tag with a specific key, regardless of the tag value.

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'ServiceNames': [
        'string',
    ],
    'ServiceDetails': [
        {
            'ServiceName': 'string',
            'ServiceId': 'string',
            'ServiceType': [
                {
                    'ServiceType': 'Interface'|'Gateway'|'GatewayLoadBalancer'
                },
            ],
            'AvailabilityZones': [
                'string',
            ],
            'Owner': 'string',
            'BaseEndpointDnsNames': [
                'string',
            ],
            'PrivateDnsName': 'string',
            'PrivateDnsNames': [
                {
                    'PrivateDnsName': 'string'
                },
            ],
            'VpcEndpointPolicySupported': True|False,
            'AcceptanceRequired': True|False,
            'ManagesVpcEndpoints': True|False,
            'Tags': [
                {
                    'Key': 'string',
                    'Value': 'string'
                },
            ],
            'PrivateDnsNameVerificationState': 'pendingVerification'|'verified'|'failed'
        },
    ],

}

**Response Structure**

- _(dict) --_

    Contains the output of DescribeVpcEndpointServices.

    - **ServiceNames** _(list) --_

        A list of supported services.

        - _(string) --_
    - **ServiceDetails** _(list) --_

        Information about the service.

        - _(dict) --_

            Describes a VPC endpoint service.

            - **ServiceName** _(string) --_

                The Amazon Resource Name (ARN) of the service.

            - **ServiceId** _(string) --_

                The ID of the endpoint service.

            - **ServiceType** _(list) --_

                The type of service.

                - _(dict) --_

                    Describes the type of service for a VPC endpoint.

                    - **ServiceType** _(string) --_

                        The type of service.

            - **AvailabilityZones** _(list) --_

                The Availability Zones in which the service is available.

                - _(string) --_
            - **Owner** _(string) --_

                The AWS account ID of the service owner.

            - **BaseEndpointDnsNames** _(list) --_

                The DNS names for the service.

                - _(string) --_
            - **PrivateDnsName** _(string) --_

                The private DNS name for the service.

            - **PrivateDnsNames** _(list) --_

                The private DNS names assigned to the VPC endpoint service.

                - _(dict) --_

                    Information about the Private DNS name for interface endpoints.

                    - **PrivateDnsName** _(string) --_

                        The private DNS name assigned to the VPC endpoint service.

            - **VpcEndpointPolicySupported** _(boolean) --_

                Indicates whether the service supports endpoint policies.

            - **AcceptanceRequired** _(boolean) --_

                Indicates whether VPC endpoint connection requests to the service must be accepted by the service owner.

            - **ManagesVpcEndpoints** _(boolean) --_

                Indicates whether the service manages its VPC endpoints. Management of the service VPC endpoints using the VPC endpoint API is restricted.

            - **Tags** _(list) --_

                Any tags assigned to the service.

                - _(dict) --_

                    Describes a tag.

                    - **Key** _(string) --_

                        The key of the tag.

                        Constraints: Tag keys are case-sensitive and accept a maximum of 127 Unicode characters. May not begin with aws: .

                    - **Value** _(string) --_

                        The value of the tag.

                        Constraints: Tag values are case-sensitive and accept a maximum of 255 Unicode characters.

            - **PrivateDnsNameVerificationState** _(string) --_

                The verification state of the VPC endpoint service.

                Consumers of the endpoint service cannot use the private name when the state is not verified .


_class_ EC2.Paginator.DescribeVpcEndpoints

paginator = client.get_paginator('describe_vpc_endpoints')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_vpc_endpoints()](#EC2.Client.describe_vpc_endpoints "EC2.Client.describe_vpc_endpoints").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeVpcEndpoints)

**Request Syntax**

response_iterator = paginator.paginate(
    DryRun=True|False,
    VpcEndpointIds=[
        'string',
    ],
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **VpcEndpointIds** (_list_) --

    One or more endpoint IDs.

    - _(string) --_
- **Filters** (_list_) --

    One or more filters.

    - service-name - The name of the service.
    - vpc-id - The ID of the VPC in which the endpoint resides.
    - vpc-endpoint-id - The ID of the endpoint.
    - vpc-endpoint-state - The state of the endpoint (pendingAcceptance | pending | available | deleting | deleted | rejected | failed ).
    - vpc-endpoint-type - The type of VPC endpoint (Interface | Gateway | GatewayLoadBalancer ).
    - tag :<key> - The key/value combination of a tag assigned to the resource. Use the tag key in the filter name and the tag value as the filter value. For example, to find all resources that have a tag with the key Owner and the value TeamA , specify tag:Owner for the filter name and TeamA for the filter value.
    - tag-key - The key of a tag assigned to the resource. Use this filter to find all resources assigned a tag with a specific key, regardless of the tag value.

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'VpcEndpoints': [
        {
            'VpcEndpointId': 'string',
            'VpcEndpointType': 'Interface'|'Gateway'|'GatewayLoadBalancer',
            'VpcId': 'string',
            'ServiceName': 'string',
            'State': 'PendingAcceptance'|'Pending'|'Available'|'Deleting'|'Deleted'|'Rejected'|'Failed'|'Expired',
            'PolicyDocument': 'string',
            'RouteTableIds': [
                'string',
            ],
            'SubnetIds': [
                'string',
            ],
            'Groups': [
                {
                    'GroupId': 'string',
                    'GroupName': 'string'
                },
            ],
            'PrivateDnsEnabled': True|False,
            'RequesterManaged': True|False,
            'NetworkInterfaceIds': [
                'string',
            ],
            'DnsEntries': [
                {
                    'DnsName': 'string',
                    'HostedZoneId': 'string'
                },
            ],
            'CreationTimestamp': datetime(2015, 1, 1),
            'Tags': [
                {
                    'Key': 'string',
                    'Value': 'string'
                },
            ],
            'OwnerId': 'string',
            'LastError': {
                'Message': 'string',
                'Code': 'string'
            }
        },
    ],

}

**Response Structure**

- _(dict) --_

    Contains the output of DescribeVpcEndpoints.

    - **VpcEndpoints** _(list) --_

        Information about the endpoints.

        - _(dict) --_

            Describes a VPC endpoint.

            - **VpcEndpointId** _(string) --_

                The ID of the VPC endpoint.

            - **VpcEndpointType** _(string) --_

                The type of endpoint.

            - **VpcId** _(string) --_

                The ID of the VPC to which the endpoint is associated.

            - **ServiceName** _(string) --_

                The name of the service to which the endpoint is associated.

            - **State** _(string) --_

                The state of the VPC endpoint.

            - **PolicyDocument** _(string) --_

                The policy document associated with the endpoint, if applicable.

            - **RouteTableIds** _(list) --_

                (Gateway endpoint) One or more route tables associated with the endpoint.

                - _(string) --_
            - **SubnetIds** _(list) --_

                (Interface endpoint) One or more subnets in which the endpoint is located.

                - _(string) --_
            - **Groups** _(list) --_

                (Interface endpoint) Information about the security groups that are associated with the network interface.

                - _(dict) --_

                    Describes a security group.

                    - **GroupId** _(string) --_

                        The ID of the security group.

                    - **GroupName** _(string) --_

                        The name of the security group.

            - **PrivateDnsEnabled** _(boolean) --_

                (Interface endpoint) Indicates whether the VPC is associated with a private hosted zone.

            - **RequesterManaged** _(boolean) --_

                Indicates whether the VPC endpoint is being managed by its service.

            - **NetworkInterfaceIds** _(list) --_

                (Interface endpoint) One or more network interfaces for the endpoint.

                - _(string) --_
            - **DnsEntries** _(list) --_

                (Interface endpoint) The DNS entries for the endpoint.

                - _(dict) --_

                    Describes a DNS entry.

                    - **DnsName** _(string) --_

                        The DNS name.

                    - **HostedZoneId** _(string) --_

                        The ID of the private hosted zone.

            - **CreationTimestamp** _(datetime) --_

                The date and time that the VPC endpoint was created.

            - **Tags** _(list) --_

                Any tags assigned to the VPC endpoint.

                - _(dict) --_

                    Describes a tag.

                    - **Key** _(string) --_

                        The key of the tag.

                        Constraints: Tag keys are case-sensitive and accept a maximum of 127 Unicode characters. May not begin with aws: .

                    - **Value** _(string) --_

                        The value of the tag.

                        Constraints: Tag values are case-sensitive and accept a maximum of 255 Unicode characters.

            - **OwnerId** _(string) --_

                The ID of the AWS account that owns the VPC endpoint.

            - **LastError** _(dict) --_

                The last error that occurred for VPC endpoint.

                - **Message** _(string) --_

                    The error message for the VPC endpoint error.

                - **Code** _(string) --_

                    The error code for the VPC endpoint error.


_class_ EC2.Paginator.DescribeVpcPeeringConnections

paginator = client.get_paginator('describe_vpc_peering_connections')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_vpc_peering_connections()](#EC2.Client.describe_vpc_peering_connections "EC2.Client.describe_vpc_peering_connections").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeVpcPeeringConnections)

**Request Syntax**

response_iterator = paginator.paginate(
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    DryRun=True|False,
    VpcPeeringConnectionIds=[
        'string',
    ],
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **Filters** (_list_) --

    One or more filters.

    - accepter-vpc-info.cidr-block - The IPv4 CIDR block of the accepter VPC.
    - accepter-vpc-info.owner-id - The AWS account ID of the owner of the accepter VPC.
    - accepter-vpc-info.vpc-id - The ID of the accepter VPC.
    - expiration-time - The expiration date and time for the VPC peering connection.
    - requester-vpc-info.cidr-block - The IPv4 CIDR block of the requester's VPC.
    - requester-vpc-info.owner-id - The AWS account ID of the owner of the requester VPC.
    - requester-vpc-info.vpc-id - The ID of the requester VPC.
    - status-code - The status of the VPC peering connection (pending-acceptance | failed | expired | provisioning | active | deleting | deleted | rejected ).
    - status-message - A message that provides more information about the status of the VPC peering connection, if applicable.
    - tag :<key> - The key/value combination of a tag assigned to the resource. Use the tag key in the filter name and the tag value as the filter value. For example, to find all resources that have a tag with the key Owner and the value TeamA , specify tag:Owner for the filter name and TeamA for the filter value.
    - tag-key - The key of a tag assigned to the resource. Use this filter to find all resources assigned a tag with a specific key, regardless of the tag value.
    - vpc-peering-connection-id - The ID of the VPC peering connection.

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **VpcPeeringConnectionIds** (_list_) --

    One or more VPC peering connection IDs.

    Default: Describes all your VPC peering connections.

    - _(string) --_
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'VpcPeeringConnections': [
        {
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
                    'AllowEgressFromLocalClassicLinkToRemoteVpc'#39;: True|False,
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
        },
    ],

}

**Response Structure**

- _(dict) --_

    - **VpcPeeringConnections** _(list) --_

        Information about the VPC peering connections.

        - _(dict) --_

            Describes a VPC peering connection.

            - **AccepterVpcInfo** _(dict) --_

                Information about the accepter VPC. CIDR block information is only returned when describing an active VPC peering connection.

                - **CidrBlock** _(string) --_

                    The IPv4 CIDR block for the VPC.

                - **Ipv6CidrBlockSet** _(list) --_

                    The IPv6 CIDR block for the VPC.

                    - _(dict) --_

                        Describes an IPv6 CIDR block.

                        - **Ipv6CidrBlock** _(string) --_

                            The IPv6 CIDR block.

                - **CidrBlockSet** _(list) --_

                    Information about the IPv4 CIDR blocks for the VPC.

                    - _(dict) --_

                        Describes an IPv4 CIDR block.

                        - **CidrBlock** _(string) --_

                            The IPv4 CIDR block.

                - **OwnerId** _(string) --_

                    The AWS account ID of the VPC owner.

                - **PeeringOptions** _(dict) --_

                    Information about the VPC peering connection options for the accepter or requester VPC.

                    - **AllowDnsResolutionFromRemoteVpc** _(boolean) --_

                        Indicates whether a local VPC can resolve public DNS hostnames to private IP addresses when queried from instances in a peer VPC.

                    - **AllowEgressFromLocalClassicLinkToRemoteVpc** _(boolean) --_

                        Indicates whether a local ClassicLink connection can communicate with the peer VPC over the VPC peering connection.

                    - **AllowEgressFromLocalVpcToRemoteClassicLink** _(boolean) --_

                        Indicates whether a local VPC can communicate with a ClassicLink connection in the peer VPC over the VPC peering connection.

                - **VpcId** _(string) --_

                    The ID of the VPC.

                - **Region** _(string) --_

                    The Region in which the VPC is located.

            - **ExpirationTime** _(datetime) --_

                The time that an unaccepted VPC peering connection will expire.

            - **RequesterVpcInfo** _(dict) --_

                Information about the requester VPC. CIDR block information is only returned when describing an active VPC peering connection.

                - **CidrBlock** _(string) --_

                    The IPv4 CIDR block for the VPC.

                - **Ipv6CidrBlockSet** _(list) --_

                    The IPv6 CIDR block for the VPC.

                    - _(dict) --_

                        Describes an IPv6 CIDR block.

                        - **Ipv6CidrBlock** _(string) --_

                            The IPv6 CIDR block.

                - **CidrBlockSet** _(list) --_

                    Information about the IPv4 CIDR blocks for the VPC.

                    - _(dict) --_

                        Describes an IPv4 CIDR block.

                        - **CidrBlock** _(string) --_

                            The IPv4 CIDR block.

                - **OwnerId** _(string) --_

                    The AWS account ID of the VPC owner.

                - **PeeringOptions** _(dict) --_

                    Information about the VPC peering connection options for the accepter or requester VPC.

                    - **AllowDnsResolutionFromRemoteVpc** _(boolean) --_

                        Indicates whether a local VPC can resolve public DNS hostnames to private IP addresses when queried from instances in a peer VPC.

                    - **AllowEgressFromLocalClassicLinkToRemoteVpc** _(boolean) --_

                        Indicates whether a local ClassicLink connection can communicate with the peer VPC over the VPC peering connection.

                    - **AllowEgressFromLocalVpcToRemoteClassicLink** _(boolean) --_

                        Indicates whether a local VPC can communicate with a ClassicLink connection in the peer VPC over the VPC peering connection.

                - **VpcId** _(string) --_

                    The ID of the VPC.

                - **Region** _(string) --_

                    The Region in which the VPC is located.

            - **Status** _(dict) --_

                The status of the VPC peering connection.

                - **Code** _(string) --_

                    The status of the VPC peering connection.

                - **Message** _(string) --_

                    A message that provides more information about the status, if applicable.

            - **Tags** _(list) --_

                Any tags assigned to the resource.

                - _(dict) --_

                    Describes a tag.

                    - **Key** _(string) --_

                        The key of the tag.

                        Constraints: Tag keys are case-sensitive and accept a maximum of 127 Unicode characters. May not begin with aws: .

                    - **Value** _(string) --_

                        The value of the tag.

                        Constraints: Tag values are case-sensitive and accept a maximum of 255 Unicode characters.

            - **VpcPeeringConnectionId** _(string) --_

                The ID of the VPC peering connection.


_class_ EC2.Paginator.DescribeVpcs

paginator = client.get_paginator('describe_vpcs')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.describe_vpcs()](#EC2.Client.describe_vpcs "EC2.Client.describe_vpcs").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/DescribeVpcs)

**Request Syntax**

response_iterator = paginator.paginate(
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    VpcIds=[
        'string',
    ],
    DryRun=True|False,
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **Filters** (_list_) --

    One or more filters.

    - cidr - The primary IPv4 CIDR block of the VPC. The CIDR block you specify must exactly match the VPC's CIDR block for information to be returned for the VPC. Must contain the slash followed by one or two digits (for example, /28 ).
    - cidr-block-association.cidr-block - An IPv4 CIDR block associated with the VPC.
    - cidr-block-association.association-id - The association ID for an IPv4 CIDR block associated with the VPC.
    - cidr-block-association.state - The state of an IPv4 CIDR block associated with the VPC.
    - dhcp-options-id - The ID of a set of DHCP options.
    - ipv6-cidr-block-association.ipv6-cidr-block - An IPv6 CIDR block associated with the VPC.
    - ipv6-cidr-block-association.ipv6-pool - The ID of the IPv6 address pool from which the IPv6 CIDR block is allocated.
    - ipv6-cidr-block-association.association-id - The association ID for an IPv6 CIDR block associated with the VPC.
    - ipv6-cidr-block-association.state - The state of an IPv6 CIDR block associated with the VPC.
    - isDefault - Indicates whether the VPC is the default VPC.
    - owner-id - The ID of the AWS account that owns the VPC.
    - state - The state of the VPC (pending | available ).
    - tag :<key> - The key/value combination of a tag assigned to the resource. Use the tag key in the filter name and the tag value as the filter value. For example, to find all resources that have a tag with the key Owner and the value TeamA , specify tag:Owner for the filter name and TeamA for the filter value.
    - tag-key - The key of a tag assigned to the resource. Use this filter to find all resources assigned a tag with a specific key, regardless of the tag value.
    - vpc-id - The ID of the VPC.

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **VpcIds** (_list_) --

    One or more VPC IDs.

    Default: Describes all your VPCs.

    - _(string) --_
- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'Vpcs': [
        {
            'CidrBlock': 'string',
            'DhcpOptionsId': 'string',
            'State': 'pending'|'available',
            'VpcId': 'string',
            'OwnerId': 'string',
            'InstanceTenancy': 'default'|'dedicated'|'host',
            'Ipv6CidrBlockAssociationSet': [
                {
                    'AssociationId': 'string',
                    'Ipv6CidrBlock': 'string',
                    'Ipv6CidrBlockState': {
                        'State': 'associating'|'associated'|'disassociating'|'disassociated'|'failing'|'failed',
                        'StatusMessage': 'string'
                    },
                    'NetworkBorderGroup': 'string',
                    'Ipv6Pool': 'string'
                },
            ],
            'CidrBlockAssociationSet': [
                {
                    'AssociationId': 'string',
                    'CidrBlock': 'string',
                    'CidrBlockState': {
                        'State': 'associating'|'associated'|'disassociating'|'disassociated'|'failing'|'failed',
                        'StatusMessage': 'string'
                    }
                },
            ],
            'IsDefault': True|False,
            'Tags': [
                {
                    'Key': 'string',
                    'Value': 'string'
                },
            ]
        },
    ],

}

**Response Structure**

- _(dict) --_

    - **Vpcs** _(list) --_

        Information about one or more VPCs.

        - _(dict) --_

            Describes a VPC.

            - **CidrBlock** _(string) --_

                The primary IPv4 CIDR block for the VPC.

            - **DhcpOptionsId** _(string) --_

                The ID of the set of DHCP options you've associated with the VPC.

            - **State** _(string) --_

                The current state of the VPC.

            - **VpcId** _(string) --_

                The ID of the VPC.

            - **OwnerId** _(string) --_

                The ID of the AWS account that owns the VPC.

            - **InstanceTenancy** _(string) --_

                The allowed tenancy of instances launched into the VPC.

            - **Ipv6CidrBlockAssociationSet** _(list) --_

                Information about the IPv6 CIDR blocks associated with the VPC.

                - _(dict) --_

                    Describes an IPv6 CIDR block associated with a VPC.

                    - **AssociationId** _(string) --_

                        The association ID for the IPv6 CIDR block.

                    - **Ipv6CidrBlock** _(string) --_

                        The IPv6 CIDR block.

                    - **Ipv6CidrBlockState** _(dict) --_

                        Information about the state of the CIDR block.

                        - **State** _(string) --_

                            The state of the CIDR block.

                        - **StatusMessage** _(string) --_

                            A message about the status of the CIDR block, if applicable.

                    - **NetworkBorderGroup** _(string) --_

                        The name of the unique set of Availability Zones, Local Zones, or Wavelength Zones from which AWS advertises IP addresses, for example, us-east-1-wl1-bos-wlz-1 .

                    - **Ipv6Pool** _(string) --_

                        The ID of the IPv6 address pool from which the IPv6 CIDR block is allocated.

            - **CidrBlockAssociationSet** _(list) --_

                Information about the IPv4 CIDR blocks associated with the VPC.

                - _(dict) --_

                    Describes an IPv4 CIDR block associated with a VPC.

                    - **AssociationId** _(string) --_

                        The association ID for the IPv4 CIDR block.

                    - **CidrBlock** _(string) --_

                        The IPv4 CIDR block.

                    - **CidrBlockState** _(dict) --_

                        Information about the state of the CIDR block.

                        - **State** _(string) --_

                            The state of the CIDR block.

                        - **StatusMessage** _(string) --_

                            A message about the status of the CIDR block, if applicable.

            - **IsDefault** _(boolean) --_

                Indicates whether the VPC is the default VPC.

            - **Tags** _(list) --_

                Any tags assigned to the VPC.

                - _(dict) --_

                    Describes a tag.

                    - **Key** _(string) --_

                        The key of the tag.

                        Constraints: Tag keys are case-sensitive and accept a maximum of 127 Unicode characters. May not begin with aws: .

                    - **Value** _(string) --_

                        The value of the tag.

                        Constraints: Tag values are case-sensitive and accept a maximum of 255 Unicode characters.


_class_ EC2.Paginator.GetAssociatedIpv6PoolCidrs

paginator = client.get_paginator('get_associated_ipv6_pool_cidrs')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.get_associated_ipv6_pool_cidrs()](#EC2.Client.get_associated_ipv6_pool_cidrs "EC2.Client.get_associated_ipv6_pool_cidrs").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/GetAssociatedIpv6PoolCidrs)

**Request Syntax**

response_iterator = paginator.paginate(
    PoolId='string',
    DryRun=True|False,
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **PoolId** (_string_) --

    **[REQUIRED]**

    The ID of the IPv6 address pool.

- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'Ipv6CidrAssociations': [
        {
            'Ipv6Cidr': 'string',
            'AssociatedResource': 'string'
        },
    ],

}

**Response Structure**

- _(dict) --_

    - **Ipv6CidrAssociations** _(list) --_

        Information about the IPv6 CIDR block associations.

        - _(dict) --_

            Describes an IPv6 CIDR block association.

            - **Ipv6Cidr** _(string) --_

                The IPv6 CIDR block.

            - **AssociatedResource** _(string) --_

                The resource that's associated with the IPv6 CIDR block.


_class_ EC2.Paginator.GetGroupsForCapacityReservation

paginator = client.get_paginator('get_groups_for_capacity_reservation')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.get_groups_for_capacity_reservation()](#EC2.Client.get_groups_for_capacity_reservation "EC2.Client.get_groups_for_capacity_reservation").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/GetGroupsForCapacityReservation)

**Request Syntax**

response_iterator = paginator.paginate(
    CapacityReservationId='string',
    DryRun=True|False,
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **CapacityReservationId** (_string_) --

    **[REQUIRED]**

    The ID of the Capacity Reservation.

- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'CapacityReservationGroups': [
        {
            'GroupArn': 'string',
            'OwnerId': 'string'
        },
    ]
}

**Response Structure**

- _(dict) --_

    - **CapacityReservationGroups** _(list) --_

        Information about the resource groups to which the Capacity Reservation has been added.

        - _(dict) --_

            Describes a resource group to which a Capacity Reservation has been added.

            - **GroupArn** _(string) --_

                The ARN of the resource group.

            - **OwnerId** _(string) --_

                The ID of the AWS account that owns the resource group.


_class_ EC2.Paginator.GetManagedPrefixListAssociations

paginator = client.get_paginator('get_managed_prefix_list_associations')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.get_managed_prefix_list_associations()](#EC2.Client.get_managed_prefix_list_associations "EC2.Client.get_managed_prefix_list_associations").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/GetManagedPrefixListAssociations)

**Request Syntax**

response_iterator = paginator.paginate(
    DryRun=True|False,
    PrefixListId='string',
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **PrefixListId** (_string_) --

    **[REQUIRED]**

    The ID of the prefix list.

- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'PrefixListAssociations': [
        {
            'ResourceId': 'string',
            'ResourceOwner': 'string'
        },
    ],

}

**Response Structure**

- _(dict) --_

    - **PrefixListAssociations** _(list) --_

        Information about the associations.

        - _(dict) --_

            Describes the resource with which a prefix list is associated.

            - **ResourceId** _(string) --_

                The ID of the resource.

            - **ResourceOwner** _(string) --_

                The owner of the resource.


_class_ EC2.Paginator.GetManagedPrefixListEntries

paginator = client.get_paginator('get_managed_prefix_list_entries')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.get_managed_prefix_list_entries()](#EC2.Client.get_managed_prefix_list_entries "EC2.Client.get_managed_prefix_list_entries").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/GetManagedPrefixListEntries)

**Request Syntax**

response_iterator = paginator.paginate(
    DryRun=True|False,
    PrefixListId='string',
    TargetVersion=123,
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **PrefixListId** (_string_) --

    **[REQUIRED]**

    The ID of the prefix list.

- **TargetVersion** (_integer_) -- The version of the prefix list for which to return the entries. The default is the current version.
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'Entries': [
        {
            'Cidr': 'string',
            'Description': 'string'
        },
    ],

}

**Response Structure**

- _(dict) --_

    - **Entries** _(list) --_

        Information about the prefix list entries.

        - _(dict) --_

            Describes a prefix list entry.

            - **Cidr** _(string) --_

                The CIDR block.

            - **Description** _(string) --_

                The description.


_class_ EC2.Paginator.GetTransitGatewayAttachmentPropagationsdefinition")

paginator = client.get_paginator('get_transit_gateway_attachment_propagations')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.get_transit_gateway_attachment_propagations()](#EC2.Client.get_transit_gateway_attachment_propagations "EC2.Client.get_transit_gateway_attachment_propagations").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/GetTransitGatewayAttachmentPropagations)

**Request Syntax**

response_iterator = paginator.paginate(
    TransitGatewayAttachmentId='string',
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    DryRun=True|False,
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **TransitGatewayAttachmentId** (_string_) --

    **[REQUIRED]**

    The ID of the attachment.

- **Filters** (_list_) --

    One or more filters. The possible values are:

    - transit-gateway-route-table-id - The ID of the transit gateway route table.

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'TransitGatewayAttachmentPropagations': [
        {
            'TransitGatewayRouteTableId': 'string',
            'State': 'enabling'|'enabled'|'disabling'|'disabled'
        },
    ],

}

**Response Structure**

- _(dict) --_

    - **TransitGatewayAttachmentPropagations** _(list) --_

        Information about the propagation route tables.

        - _(dict) --_

            Describes a propagation route table.

            - **TransitGatewayRouteTableId** _(string) --_

                The ID of the propagation route table.

            - **State** _(string) --_

                The state of the propagation route table.


_class_ EC2.Paginator.GetTransitGatewayMulticastDomainAssociationsdefinition")

paginator = client.get_paginator('get_transit_gateway_multicast_domain_associations')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.get_transit_gateway_multicast_domain_associations()](#EC2.Client.get_transit_gateway_multicast_domain_associations "EC2.Client.get_transit_gateway_multicast_domain_associations").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/GetTransitGatewayMulticastDomainAssociations)

**Request Syntax**

response_iterator = paginator.paginate(
    TransitGatewayMulticastDomainId='string',
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    DryRun=True|False,
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **TransitGatewayMulticastDomainId** (_string_) -- The ID of the transit gateway multicast domain.
- **Filters** (_list_) --

    One or more filters. The possible values are:

    - resource-id - The ID of the resource.
    - resource-type - The type of resource. The valid value is: vpc .
    - state - The state of the subnet association. Valid values are associated | associating | disassociated | disassociating .
    - subnet-id - The ID of the subnet.
    - transit-gateway-attachment-id - The id of the transit gateway attachment.

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'MulticastDomainAssociations': [
        {
            'TransitGatewayAttachmentId': 'string',
            'ResourceId': 'string',
            'ResourceType': 'vpc'|'vpn'|'direct-connect-gateway'|'connect'|'peering'|'tgw-peering',
            'ResourceOwnerId': 'string',
            'Subnet': {
                'SubnetId': 'string',
                'State': 'pendingAcceptance'|'associating'|'associated'|'disassociating'|'disassociated'|'rejected'|'failed'
            }
        },
    ],

}

**Response Structure**

- _(dict) --_

    - **MulticastDomainAssociations** _(list) --_

        Information about the multicast domain associations.

        - _(dict) --_

            Describes the resources associated with the transit gateway multicast domain.

            - **TransitGatewayAttachmentId** _(string) --_

                The ID of the transit gateway attachment.

            - **ResourceId** _(string) --_

                The ID of the resource.

            - **ResourceType** _(string) --_

                The type of resource, for example a VPC attachment.

            - **ResourceOwnerId** _(string) --_

                The ID of the AWS account that owns the transit gateway multicast domain association resource.

            - **Subnet** _(dict) --_

                The subnet associated with the transit gateway multicast domain.

                - **SubnetId** _(string) --_

                    The ID of the subnet.

                - **State** _(string) --_

                    The state of the subnet association.


_class_ EC2.Paginator.GetTransitGatewayPrefixListReferences

paginator = client.get_paginator('get_transit_gateway_prefix_list_references')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.get_transit_gateway_prefix_list_references()](#EC2.Client.get_transit_gateway_prefix_list_references "EC2.Client.get_transit_gateway_prefix_list_references").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/GetTransitGatewayPrefixListReferences)

**Request Syntax**

response_iterator = paginator.paginate(
    TransitGatewayRouteTableId='string',
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    DryRun=True|False,
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **TransitGatewayRouteTableId** (_string_) --

    **[REQUIRED]**

    The ID of the transit gateway route table.

- **Filters** (_list_) --

    One or more filters. The possible values are:

    - attachment.resource-id - The ID of the resource for the attachment.
    - attachment.resource-type - The type of resource for the attachment. Valid values are vpc | vpn | direct-connect-gateway | peering .
    - attachment.transit-gateway-attachment-id - The ID of the attachment.
    - is-blackhole - Whether traffic matching the route is blocked (true | false ).
    - prefix-list-id - The ID of the prefix list.
    - prefix-list-owner-id - The ID of the owner of the prefix list.
    - state - The state of the prefix list reference (pending | available | modifying | deleting ).

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'TransitGatewayPrefixListReferences': [
        {
            'TransitGatewayRouteTableId': 'string',
            'PrefixListId': 'string',
            'PrefixListOwnerId': 'string',
            'State': 'pending'|'available'|'modifying'|'deleting',
            'Blackhole': True|False,
            'TransitGatewayAttachment': {
                'TransitGatewayAttachmentId': 'string',
                'ResourceType': 'vpc'|'vpn'|'direct-connect-gateway'|'connect'|'peering'|'tgw-peering',
                'ResourceId': 'string'
            }
        },
    ],

}

**Response Structure**

- _(dict) --_

    - **TransitGatewayPrefixListReferences** _(list) --_

        Information about the prefix list references.

        - _(dict) --_

            Describes a prefix list reference.

            - **TransitGatewayRouteTableId** _(string) --_

                The ID of the transit gateway route table.

            - **PrefixListId** _(string) --_

                The ID of the prefix list.

            - **PrefixListOwnerId** _(string) --_

                The ID of the prefix list owner.

            - **State** _(string) --_

                The state of the prefix list reference.

            - **Blackhole** _(boolean) --_

                Indicates whether traffic that matches this route is dropped.

            - **TransitGatewayAttachment** _(dict) --_

                Information about the transit gateway attachment.

                - **TransitGatewayAttachmentId** _(string) --_

                    The ID of the attachment.

                - **ResourceType** _(string) --_

                    The resource type. Note that the tgw-peering resource type has been deprecated.

                - **ResourceId** _(string) --_

                    The ID of the resource.


_class_ EC2.Paginator.GetTransitGatewayRouteTableAssociationsdefinition")

paginator = client.get_paginator('get_transit_gateway_route_table_associations')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.get_transit_gateway_route_table_associations()](#EC2.Client.get_transit_gateway_route_table_associations "EC2.Client.get_transit_gateway_route_table_associations").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/GetTransitGatewayRouteTableAssociations)

**Request Syntax**

response_iterator = paginator.paginate(
    TransitGatewayRouteTableId='string',
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    DryRun=True|False,
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **TransitGatewayRouteTableId** (_string_) --

    **[REQUIRED]**

    The ID of the transit gateway route table.

- **Filters** (_list_) --

    One or more filters. The possible values are:

    - resource-id - The ID of the resource.
    - resource-type - The resource type. Valid values are vpc | vpn | direct-connect-gateway | peering | connect .
    - transit-gateway-attachment-id - The ID of the attachment.

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'Associations': [
        {
            'TransitGatewayAttachmentId': 'string',
            'ResourceId': 'string',
            'ResourceType': 'vpc'|'vpn'|'direct-connect-gateway'|'connect'|'peering'|'tgw-peering',
            'State': 'associating'|'associated'|'disassociating'|'disassociated'
        },
    ],

}

**Response Structure**

- _(dict) --_

    - **Associations** _(list) --_

        Information about the associations.

        - _(dict) --_

            Describes an association between a route table and a resource attachment.

            - **TransitGatewayAttachmentId** _(string) --_

                The ID of the attachment.

            - **ResourceId** _(string) --_

                The ID of the resource.

            - **ResourceType** _(string) --_

                The resource type. Note that the tgw-peering resource type has been deprecated.

            - **State** _(string) --_

                The state of the association.


_class_ EC2.Paginator.GetTransitGatewayRouteTablePropagationsdefinition")

paginator = client.get_paginator('get_transit_gateway_route_table_propagations')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.get_transit_gateway_route_table_propagations()](#EC2.Client.get_transit_gateway_route_table_propagations "EC2.Client.get_transit_gateway_route_table_propagations").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/GetTransitGatewayRouteTablePropagations)

**Request Syntax**

response_iterator = paginator.paginate(
    TransitGatewayRouteTableId='string',
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    DryRun=True|False,
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **TransitGatewayRouteTableId** (_string_) --

    **[REQUIRED]**

    The ID of the transit gateway route table.

- **Filters** (_list_) --

    One or more filters. The possible values are:

    - resource-id - The ID of the resource.
    - resource-type - The resource type. Valid values are vpc | vpn | direct-connect-gateway | peering | connect .
    - transit-gateway-attachment-id - The ID of the attachment.

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'TransitGatewayRouteTablePropagations': [
        {
            'TransitGatewayAttachmentId': 'string',
            'ResourceId': 'string',
            'ResourceType': 'vpc'|'vpn'|'direct-connect-gateway'|'connect'|'peering'|'tgw-peering',
            'State': 'enabling'|'enabled'|'disabling'|'disabled'
        },
    ],

}

**Response Structure**

- _(dict) --_

    - **TransitGatewayRouteTablePropagations** _(list) --_

        Information about the route table propagations.

        - _(dict) --_

            Describes a route table propagation.

            - **TransitGatewayAttachmentId** _(string) --_

                The ID of the attachment.

            - **ResourceId** _(string) --_

                The ID of the resource.

            - **ResourceType** _(string) --_

                The type of resource. Note that the tgw-peering resource type has been deprecated.

            - **State** _(string) --_

                The state of the resource.


_class_ EC2.Paginator.SearchLocalGatewayRoutes

paginator = client.get_paginator('search_local_gateway_routes')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.search_local_gateway_routes()](#EC2.Client.search_local_gateway_routes "EC2.Client.search_local_gateway_routes").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/SearchLocalGatewayRoutes)

**Request Syntax**

response_iterator = paginator.paginate(
    LocalGatewayRouteTableId='string',
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    DryRun=True|False,
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **LocalGatewayRouteTableId** (_string_) --

    **[REQUIRED]**

    The ID of the local gateway route table.

- **Filters** (_list_) --

    **[REQUIRED]**

    One or more filters.

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'Routes': [
        {
            'DestinationCidrBlock': 'string',
            'LocalGatewayVirtualInterfaceGroupId': 'string',
            'Type': 'static'|'propagated',
            'State': 'pending'|'active'|'blackhole'|'deleting'|'deleted',
            'LocalGatewayRouteTableId': 'string',
            'LocalGatewayRouteTableArn': 'string',
            'OwnerId': 'string'
        },
    ],

}

**Response Structure**

- _(dict) --_

    - **Routes** _(list) --_

        Information about the routes.

        - _(dict) --_

            Describes a route for a local gateway route table.

            - **DestinationCidrBlock** _(string) --_

                The CIDR block used for destination matches.

            - **LocalGatewayVirtualInterfaceGroupId** _(string) --_

                The ID of the virtual interface group.

            - **Type** _(string) --_

                The route type.

            - **State** _(string) --_

                The state of the route.

            - **LocalGatewayRouteTableId** _(string) --_

                The ID of the local gateway route table.

            - **LocalGatewayRouteTableArn** _(string) --_

                The Amazon Resource Name (ARN) of the local gateway route table.

            - **OwnerId** _(string) --_

                The AWS account ID that owns the local gateway route.


_class_ EC2.Paginator.SearchTransitGatewayMulticastGroups

paginator = client.get_paginator('search_transit_gateway_multicast_groups')

paginate(kwargs_)

Creates an iterator that will paginate through responses from [EC2.Client.search_transit_gateway_multicast_groups()](#EC2.Client.search_transit_gateway_multicast_groups "EC2.Client.search_transit_gateway_multicast_groups").

See also: [AWS API Documentation](https://docs.aws.amazon.com/goto/WebAPI/ec2-2016-11-15/SearchTransitGatewayMulticastGroups)

**Request Syntax**

response_iterator = paginator.paginate(
    TransitGatewayMulticastDomainId='string',
    Filters=[
        {
            'Name': 'string',
            'Values': [
                'string',
            ]
        },
    ],
    DryRun=True|False,
    PaginationConfig={
        'MaxItems': 123,
        'PageSize': 123,
        'StartingToken': 'string'
    }
)

Parameters

- **TransitGatewayMulticastDomainId** (_string_) -- The ID of the transit gateway multicast domain.
- **Filters** (_list_) --

    One or more filters. The possible values are:

    - group-ip-address - The IP address of the transit gateway multicast group.
    - is-group-member - The resource is a group member. Valid values are true | false .
    - is-group-source - The resource is a group source. Valid values are true | false .
    - member-type - The member type. Valid values are igmp | static .
    - resource-id - The ID of the resource.
    - resource-type - The type of resource. Valid values are vpc | vpn | direct-connect-gateway | tgw-peering .
    - source-type - The source type. Valid values are igmp | static .
    - state - The state of the subnet association. Valid values are associated | associated | disassociated | disassociating .
    - subnet-id - The ID of the subnet.
    - transit-gateway-attachment-id - The id of the transit gateway attachment.

    - _(dict) --_

        A filter name and value pair that is used to return a more specific list of results from a describe operation. Filters can be used to match a set of resources by specific criteria, such as tags, attributes, or IDs. The filters supported by a describe operation are documented with the describe operation. For example:

        - DescribeAvailabilityZones
        - DescribeImages
        - DescribeInstances
        - DescribeKeyPairs
        - DescribeSecurityGroups
        - DescribeSnapshots
        - DescribeSubnets
        - DescribeTags
        - DescribeVolumes
        - DescribeVpcs

        - **Name** _(string) --_

            The name of the filter. Filter names are case-sensitive.

        - **Values** _(list) --_

            The filter values. Filter values are case-sensitive.

            - _(string) --_
- **DryRun** (_boolean_) -- Checks whether you have the required permissions for the action, without actually making the request, and provides an error response. If you have the required permissions, the error response is DryRunOperation . Otherwise, it is UnauthorizedOperation .
- **PaginationConfig** (_dict_) --

    A dictionary that provides parameters to control pagination.

    - **MaxItems** _(integer) --_

        The total number of items to return. If the total number of items available is more than the value specified in max-items then a NextToken will be provided in the output that you can use to resume pagination.

    - **PageSize** _(integer) --_

        The size of each page.

    - **StartingToken** _(string) --_

        A token to specify where to start paginating. This is the NextToken from a previous response.


Return type

dict

Returns

**Response Syntax**

{
    'MulticastGroups': [
        {
            'GroupIpAddress': 'string',
            'TransitGatewayAttachmentId': 'string',
            'SubnetId': 'string',
            'ResourceId': 'string',
            'ResourceType': 'vpc'|'vpn'|'direct-connect-gateway'|'connect'|'peering'|'tgw-peering',
            'ResourceOwnerId': 'string',
            'NetworkInterfaceId': 'string',
            'GroupMember': True|False,
            'GroupSource': True|False,
            'MemberType': 'static'|'igmp',
            'SourceType': 'static'|'igmp'
        },
    ],

}

**Response Structure**

- _(dict) --_

    - **MulticastGroups** _(list) --_

        Information about the transit gateway multicast group.

        - _(dict) --_

            Describes the transit gateway multicast group resources.

            - **GroupIpAddress** _(string) --_

                The IP address assigned to the transit gateway multicast group.

            - **TransitGatewayAttachmentId** _(string) --_

                The ID of the transit gateway attachment.

            - **SubnetId** _(string) --_

                The ID of the subnet.

            - **ResourceId** _(string) --_

                The ID of the resource.

            - **ResourceType** _(string) --_

                The type of resource, for example a VPC attachment.

            - **ResourceOwnerId** _(string) --_

                The ID of the AWS account that owns the transit gateway multicast domain group resource.

            - **NetworkInterfaceId** _(string) --_

                The ID of the transit gateway attachment.

            - **GroupMember** _(boolean) --_

                Indicates that the resource is a transit gateway multicast group member.

            - **GroupSource** _(boolean) --_

                Indicates that the resource is a transit gateway multicast group member.

            - **MemberType** _(string) --_

                The member type (for example, static ).

            - **SourceType** _(string) --_

                The source type.
