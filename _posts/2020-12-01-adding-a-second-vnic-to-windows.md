---
layout: post
title: Adding A Second VNIC To Windows
date: 2020-12-01 13:01
comments: true
category: 
 - powershell
author: Eric Marquez
tags: [productivity, windows, powershell]
summary: How to add a second virtual network interface to windows.
---

This week I took the opportunity to finally setup my server and make it truly useful.  This server is going to act as a jump box where I can connect from my management network to the lab and operate some virtual machines on.  Long ago in this type of situation, it would require two different network cards where a physical link would be dedicated for each network.  A better way to perform this same design is to use switch independent teaming and virtual NICs.  When switch independent teaming is used it allows you to bond both interfaces, this gives you a redundant link, and additional bandwidth.  In my case, I’m working with two 40G NICs and this will give me a combined links speed of 80G, who wouldn’t want to work with 80G of bandwidth!

To setup this environment the switch that connects to the server needs to be configured in a way to take advantage of this setup. The switch setup is very simple, it only needs a native VLAN and a VLAN tag.  In this example VLAN 5 would be my management network and VLAN 3000 is the lab.

```
Interface Ethernet1/1
  switchport
  switchport mode trunk
  switchport trunk native vlan 5
  switchport trunk allowed vlan 5, 3000

```

VLAN 5 being a management network, it’s setup with DHCP.  VLAN 3000 is the secondary network and it  requires a static IP address.
To configure windows, the following set of commands needs to be performed.

The first thing we need is the network adapter names.  This server has more than two network interfaces installed. To ensure I'm working with the correct interfaces I’ll need to gather the actual interface names.  I happen to know the interfaces names contain the word "Slot".

```powershell
$nicName = Get-NetAdapter | Where-Object {$_.Name -match "Slot"}
$nicName.Name
SLOT 3 2
SLOT 3

```

Next, setup windows [Switch Independent teaming](https://docs.microsoft.com/en-us/windows-server/networking/technologies/nic-teaming/nic-teaming-settings).  The name of the team will be 
“LBFOTeam”, with a Load Balancing Algorithm of HyperVport.  I'm planning to use VM's on this host, HyperV Port is the best match.

```powershell
New-NetLbfoTeam -Name LBFOTeam –TeamMembers $nicName -TeamingMode SwitchIndependent -LoadBalancingAlgorithm HyperVPort -Confirm:$false
```

At this point the team is setup and should come online within a few minutes.  The next three steps will setup the virtual switch (VSwitch), attach a virtual network (VNIC) adapter to the VSwitch, and rename the VNIC. If you used the GUI this is where the GUI ends.  The GUI does not allow you to add multiple VNIC's, but PowerShell does.  As I stated above, my management network is already setup with DHCP so once the Management adapter is added it will pick up an IP address.

```powershell
# Virutal Switch setup
New-VMSwitch -Name HVSwitch –NetAdapterName "LBFOTeam" –MinimumBandwidthMode Weight –AllowManagementOS $false
# Attach a virtual NIC.
Add-VMNetworkAdapter –ManagementOS –Name Management –SwitchName HVSwitch
# Rename the adapter.
Get-NetAdapter -Name *Management* | Rename-NetAdapter -NewName Management
```

Next, the secondary VNIC needs to be setup and connected to my lab network.  To do this, some of the same steps need to be performed with one additional step, attaching the NIC to a VLAN id.

```powershell
# Add the additional VNIC
Add-VMNetworkAdapter -ManagementOS -Name Lab -SwitchName HVSwitch
# Rename it to a proper name
Get-NetAdapter -Name *Lab* | Rename-NetAdapter -NewName "Lab"
# Attach a VLAN ID to the VNIC
Set-VMNetworkAdapterVlan -ManagementOS -VMNetworkAdapterName Lab -Access -VlanId 3000
```

At this point the VNIC is established and connected to the lab network with the VLAN id of 3000.  When a host has two different networks attached, there is one rule.  There can only be one default gateway.  Keeping to this rule, when the static ip is configured, the gateway is not going to be configured.  Static routes will be used to route beyond the subnet attached subnet.

```powershell
# Add a Static IP to the new VNIC
New-NetIPAddress -InterfaceAlias Lab -IPAddress 192.168.1.10 -AddressFamily IPv4 -PrefixLength 24
# setup the static route to allow access to the lab network.
New-NetRoute -DestinationPrefix 10.16.0.0/12 -InterfaceAlias Lab -AddressFamily IPv4 -NextHop 192.168.1.1
New-NetRoute -DestinationPrefix 192.168.1.0/24 -InterfaceAlias Lab -AddressFamily IPv4 -NextHop 192.168.1.1
```

All of these steps provided a brief walk though of setting up Switch Independent teaming, using a VSwitch and two VNIC's that are attached to the management os on Windows Server.  Enjoy!
