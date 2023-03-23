
# make files

[toc]

---

## AWSTemplateFormatVersion
`AWSTemplateFormatVersion: 2010-09-09`

---

## Description
`Description: Linux Academy - SAAC01 - RDS - Adrian Cantrill`

---

## Parameters

```yaml
Parameters:
  LatestAmiId:
    Type: 'AWS::SSM::Parameter::Value<AWS::EC2::Image::Id>'
    Default: '/aws/service/ami-amazon-linux-latest/amzn2-ami-hvm-x86_64-gp2'
  SSHKeyPair:
      Description: SSH Key Pair for Bastion and App Instances
      Type: AWS::EC2::KeyPair::KeyName
```


---


## resource


```yaml
Resources:

  AAA:
    Type: xxx

  BBB:
    Type: xxx
```

---

### network:vpc

```yaml
  VPC:
    Type: AWS::EC2::VPC
    Properties:
      CidrBlock: 10.0.0.0/16
      EnableDnsSupport: true
      EnableDnsHostnames: true
      InstanceTenancy: default
      Tags:
        - Key: Name
          Value: VPC1
```

---

### network:subnet

```yaml
  subnetXX:
    Type: AWS::EC2::Subnet
    Properties:
      AvailabilityZone:
        Fn::Select:
          - 0
          - Fn::GetAZs: ""
      VpcId:
        Ref: xxxx
      CidrBlock: x.x.x.x/24
      Tags:
        - Key: Name
          Value: subnet-name
```


```yaml
  subnetpublicA:
    Type: AWS::EC2::Subnet
    Properties:
      AvailabilityZone:
        Fn::Select:
          - 0
          - Fn::GetAZs: ""
      VpcId:
        Ref: VPC
      CidrBlock: 10.0.1.0/24
      Tags:
        - Key: Name
          Value: subnet-public-A

  subnetpublicB:
    Type: AWS::EC2::Subnet
    Properties:
      AvailabilityZone:
        Fn::Select:
          - 1
          - Fn::GetAZs: ""
      VpcId:
        Ref: VPC
      CidrBlock: 10.0.2.0/24
      Tags:
        - Key: Name
          Value: subnet-public-B

  subnetappA:
    Type: AWS::EC2::Subnet
    Properties:
      AvailabilityZone:
        Fn::Select:
          - 0
          - Fn::GetAZs: ""
      VpcId:
        Ref: VPC
      CidrBlock: 10.0.11.0/24
      Tags:
        - Key: Name
          Value: subnet-app-A

  subnetappB:
    Type: AWS::EC2::Subnet
    Properties:
      AvailabilityZone:
        Fn::Select:
          - 1
          - Fn::GetAZs: ""
      VpcId:
        Ref: VPC
      CidrBlock: 10.0.12.0/24
      Tags:
        - Key: Name
          Value: subnet-app-B

  subnetdbA:
    Type: AWS::EC2::Subnet
    Properties:
      AvailabilityZone:
        Fn::Select:
          - 0
          - Fn::GetAZs: ""
      VpcId:
        Ref: VPC
      CidrBlock: 10.0.21.0/24
      Tags:
        - Key: Name
          Value: subnet-db-A

  subnetdbB:
    Type: AWS::EC2::Subnet
    Properties:
      AvailabilityZone:
        Fn::Select:
          - 1
          - Fn::GetAZs: ""
      VpcId:
        Ref: VPC
      CidrBlock: 10.0.22.0/24
      Tags:
        - Key: Name
          Value: subnet-db-B

```


---

### network:Gateway

```yaml
  InternetGateway:
    Type: 'AWS::EC2::InternetGateway'
    Properties:
      Tags:
      - Key: Name
        Value: vpc1-igw

  InternetGatewayAttachment:
    Type: 'AWS::EC2::VPCGatewayAttachment'
    Properties:
      VpcId:
        Ref: VPC
      InternetGatewayId:
        Ref: InternetGateway

  RouteTablePublic:
    Type: 'AWS::EC2::RouteTable'
    Properties:
      VpcId:
        Ref: VPC
      Tags:
      - Key: Name
        Value: rt-public

  RouteTablePrivateA:
    Type: 'AWS::EC2::RouteTable'
    Properties:
      VpcId:
        Ref: VPC
      Tags:
      - Key: Name
        Value: rt-private-A

  RouteTablePrivateB:
    Type: 'AWS::EC2::RouteTable'
    Properties:
      VpcId:
        Ref: VPC
      Tags:
      - Key: Name
        Value: rt-private-B

  RouteTableAssociationPublicA:
    Type: 'AWS::EC2::SubnetRouteTableAssociation'
    Properties:
      SubnetId:
        Ref: subnetpublicA
      RouteTableId:
        Ref: RouteTablePublic

  RouteTableAssociationPublicB:
    Type: 'AWS::EC2::SubnetRouteTableAssociation'
    Properties:
      SubnetId:
        Ref: subnetpublicB
      RouteTableId:
        Ref: RouteTablePublic

  RouteTableAssociationAppA:
    Type: 'AWS::EC2::SubnetRouteTableAssociation'
    Properties:
      SubnetId:
        Ref: subnetappA
      RouteTableId:
        Ref: RouteTablePrivateA

  RouteTableAssociationAppB:
    Type: 'AWS::EC2::SubnetRouteTableAssociation'
    Properties:
      SubnetId:
        Ref: subnetappB
      RouteTableId:
        Ref: RouteTablePrivateB

  RouteTableAssociationDBA:
    Type: 'AWS::EC2::SubnetRouteTableAssociation'
    Properties:
      SubnetId:
        Ref: subnetdbA
      RouteTableId:
        Ref: RouteTablePrivateA

  RouteTableAssociationDBB:
    Type: 'AWS::EC2::SubnetRouteTableAssociation'
    Properties:
      SubnetId:
        Ref: subnetdbB
      RouteTableId:
        Ref: RouteTablePrivateB


  RouteTablePublicInternetRoute:
    Type: 'AWS::EC2::Route'
    DependsOn: InternetGatewayAttachment
    Properties:
      RouteTableId:
        Ref: RouteTablePublic
      DestinationCidrBlock: '0.0.0.0/0'
      GatewayId:
        Ref: InternetGateway
```

---

### NACL

```yaml
  NetworkAclPublic:
    Type: 'AWS::EC2::NetworkAcl'
    Properties:
      VpcId:
        Ref: VPC
      Tags:
      - Key: Name
        Value: nacl-public

  NetworkAclPrivate:
    Type: 'AWS::EC2::NetworkAcl'
    Properties:
      VpcId:
        Ref: VPC
      Tags:
      - Key: Name
        Value: nacl-private

  SubnetNetworkAclAssociationPublicA:
    Type: 'AWS::EC2::SubnetNetworkAclAssociation'
    Properties:
      SubnetId:
        Ref: subnetpublicA
      NetworkAclId:
        Ref: NetworkAclPublic

  SubnetNetworkAclAssociationPublicB:
    Type: 'AWS::EC2::SubnetNetworkAclAssociation'
    Properties:
      SubnetId:
        Ref: subnetpublicB
      NetworkAclId:
        Ref: NetworkAclPublic

  SubnetNetworkAclAssociationAppA:
    Type: 'AWS::EC2::SubnetNetworkAclAssociation'
    Properties:
      SubnetId:
        Ref: subnetappA
      NetworkAclId:
        Ref: NetworkAclPrivate

  SubnetNetworkAclAssociationAppB:
    Type: 'AWS::EC2::SubnetNetworkAclAssociation'
    Properties:
      SubnetId:
        Ref: subnetappB
      NetworkAclId:
        Ref: NetworkAclPrivate

  SubnetNetworkAclAssociationDBA:
    Type: 'AWS::EC2::SubnetNetworkAclAssociation'
    Properties:
      SubnetId:
        Ref: subnetdbA
      NetworkAclId:
        Ref: NetworkAclPrivate

  SubnetNetworkAclAssociationDBB:
    Type: 'AWS::EC2::SubnetNetworkAclAssociation'
    Properties:
      SubnetId:
        Ref: subnetdbB
      NetworkAclId:
        Ref: NetworkAclPrivate

  NetworkAclEntryInPublicAllowAll:
    Type: 'AWS::EC2::NetworkAclEntry'
    Properties:
      NetworkAclId:
        Ref: NetworkAclPublic
      RuleNumber: 99
      Protocol: -1
      RuleAction: allow
      Egress: false
      CidrBlock: '0.0.0.0/0'

  NetworkAclEntryOutPublicAllowAll:
    Type: 'AWS::EC2::NetworkAclEntry'
    Properties:
      NetworkAclId:
        Ref: NetworkAclPublic
      RuleNumber: 99
      Protocol: -1
      RuleAction: allow
      Egress: true
      CidrBlock: '0.0.0.0/0'

  NetworkAclEntryInPrivateAllowVPC:
    Type: 'AWS::EC2::NetworkAclEntry'
    Properties:
      NetworkAclId:
        Ref: NetworkAclPrivate
      RuleNumber: 99
      Protocol: -1
      RuleAction: allow
      Egress: false
      CidrBlock: '0.0.0.0/0'

  NetworkAclEntryOutPrivateAllowVPC:
    Type: 'AWS::EC2::NetworkAclEntry'
    Properties:
      NetworkAclId:
        Ref: NetworkAclPrivate
      RuleNumber: 99
      Protocol: -1
      RuleAction: allow
      Egress: true
      CidrBlock: '0.0.0.0/0'
```

---

### EIPA


```yaml
  EIPA:
    Type: 'AWS::EC2::EIP'
    Properties:
      Domain: vpc

  EIPB:
    Type: 'AWS::EC2::EIP'
    Properties:
      Domain: vpc

  NatGatewayA:
    Type: 'AWS::EC2::NatGateway'
    Properties:
      AllocationId: !GetAtt EIPA.AllocationId
      SubnetId:
        Ref: subnetpublicA

  NatGatewayB:
    Type: 'AWS::EC2::NatGateway'
    Properties:
      AllocationId: !GetAtt EIPB.AllocationId
      SubnetId:
        Ref: subnetpublicB

  RouteTablePrivateANATGWRoute:
    Type: 'AWS::EC2::Route'
    Properties:
      RouteTableId:
        Ref: RouteTablePrivateA
      DestinationCidrBlock: '0.0.0.0/0'
      NatGatewayId:
        Ref: NatGatewayA

  RouteTablePrivateBNATGWRoute:
    Type: 'AWS::EC2::Route'
    Properties:
      RouteTableId:
        Ref: RouteTablePrivateB
      DestinationCidrBlock: '0.0.0.0/0'
      NatGatewayId:
        Ref: NatGatewayB


  EIPBASTION:
    Type: 'AWS::EC2::EIP'
    Properties:
      Domain: vpc

  BastionSG:
    Type: 'AWS::EC2::SecurityGroup'
    Properties:
      GroupDescription: BastionSG
      SecurityGroupIngress:
      - IpProtocol: tcp
        FromPort: 22
        ToPort: 22
        CidrIp: '0.0.0.0/0'
      - IpProtocol: tcp
        FromPort: 80
        ToPort: 80
        CidrIp: '0.0.0.0/0'
      - IpProtocol: tcp
        FromPort: 443
        ToPort: 443
        CidrIp: '0.0.0.0/0'
      VpcId:
        Ref: VPC

  IAMRole:
    Type: 'AWS::IAM::Role'
    Properties:
      AssumeRolePolicyDocument:
        Version: '2012-10-17'
        Statement:
        - Effect: Allow
          Principal:
            Service: 'ec2.amazonaws.com'
          Action: 'sts:AssumeRole'
      Policies:
      - PolicyName: 's3'
        PolicyDocument:
          Version: '2012-10-17'
          Statement:
          - Effect: Allow
            Action:
            - 's3:*'
            Resource:
            - '*'

  BastionInstance:
    Type: AWS::EC2::Instance
    Properties:
      KeyName:
        Ref: SSHKeyPair
      ImageId:
        Ref: LatestAmiId
      InstanceType: t3.micro
      NetworkInterfaces:
        - AssociatePublicIpAddress: "true"
          DeviceIndex: "0"
          GroupSet:
            - Ref: "BastionSG"
          SubnetId:
            Ref: "subnetpublicA"
      Tags:
        - Key: Name
          Value: BastionHostAndWeb
      UserData:
        'Fn::Base64': !Sub |
          #!/bin/bash -ex
          sudo yum update -y
          sudo amazon-linux-extras install -y lamp-mariadb10.2-php7.2 php7.2
          sudo yum install -y httpd
          sudo systemctl start httpd
          sudo systemctl enable httpd
          cd /tmp
          sudo wget https://wordpress.org/latest.tar.gz
          sudo tar -xzf latest.tar.gz
          sudo cp -r ./wordpress/* /var/www/html/
          sudo chown -R apache /var/www
          sudo chgrp -R apache /var/www
          sudo chmod 2775 /var/www
          find /var/www -type d -exec sudo chmod 2775 {} \;
          find /var/www -type f -exec sudo chmod 0664 {} \;
          sudo systemctl restart httpd
```
