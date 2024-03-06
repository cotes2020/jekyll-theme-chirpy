



- [RAM](#ram)
- [RDS](#rds)


---


# RAM


```bash

$ aliyun ram ListPolicies | grep PolicyName

$ aliyun ram ListPolicies \
--PolicyType Custom | grep PolicyName



$ aliyun ram AttachPolicyToRole \
--PolicyName AdministratorAccess \
--PolicyType System \
--RoleName adminalicloud


$ aliyun ram AttachPolicyToRole \
--PolicyName sec-test-grace \
--PolicyType Custom \
--RoleName adminalicloud



$ aliyun ram DetachPolicyFromRole \
--PolicyName ct-boundary-testcopy-grace \
--PolicyType Custom \
--RoleName adminalicloud


$ aliyun ram ListPoliciesForRole \
--RoleName adminalicloud


$ aliyun ram DeletePolicy \
--PolicyName ct-boundary-testcopy-grace


```





# RDS

```bash

aliyun rds DescribeDBInstances \
--RegionId cn-beijing | grep DBInstanceId

aliyun rds DescribeDBInstances \
--RegionId cn-beijing
{
	"Items": {
		"DBInstance": [
			{
				"ConnectionMode": "Standard",
				"ConnectionString": "rm-dj16n21vme51m0j9o.mysql.rds.aliyuncs.com",
				"CreateTime": "2022-02-02T22:06:56Z",
				"DBInstanceClass": "mysql.n2.large.25",
				"DBInstanceId": "rm-dj16n21vme51m0j9o",
				"DBInstanceNetType": "Intranet",
				"DBInstanceStatus": "Running",
				"DBInstanceStorageType": "local_ssd",
				"DBInstanceType": "Primary",
				"Engine": "MySQL",
				"EngineVersion": "8.0",
				"ExpireTime": "",
				"InsId": 1,
				"InstanceNetworkType": "Classic",
				"LockMode": "Unlock",
				"LockReason": "",
				"MutriORsingle": false,
				"PayType": "Postpaid",
				"ReadOnlyDBInstanceIds": {
					"ReadOnlyDBInstanceId": []
				},
				"RegionId": "cn-beijing",
				"ResourceGroupId": "rg-acfnxlnj6dcw2ay",
				"TipsLevel": 0,
				"VpcCloudInstanceId": "",
				"ZoneId": "cn-beijing-a"
			}
		]
	},
	"NextToken": "o7PORW52PYRg8NUW9EJ7Yw",
	"PageNumber": 1,
	"PageRecordCount": 1,
	"RequestId": "286DD4D1-D701-5717-8880-FF5A7AD742A1",
	"TotalRecordCount": 1
}



aliyun rds ModifyActionEventPolicy




# ModifyBackupPolicy
aliyun rds ModifyBackupPolicy \
--DBInstanceId rm-dj16n21vme51m0j9o \
--BackupLog Enable

aliyun rds ModifyBackupPolicy \
--DBInstanceId rm-dj16n21vme51m0j9o \
--BackupLog Disabled


aliyun rds ModifyInstanceCrossBackupPolicy \
--DBInstanceId rm-dj16n21vme51m0j9o \
--LogBackupEnabled 0


# ModifyDBInstanceTDE
aliyun rds ModifyDBInstanceTDE \
--DBInstanceId rm-dj16n21vme51m0j9o \
--TDEStatus Disabled

aliyun rds ModifyDBInstanceTDE \
--DBInstanceId rm-dj16n21vme51m0j9o \
--TDEStatus Enabled



# SSL
aliyun rds DescribeDBInstanceSSL \
--DBInstanceId rm-dj16n21vme51m0j9o
{
	"ConnectionString": "",
	"RequestId": "6995A99E-11EC-5E8E-8BFF-7B54CF28317D",
	"RequireUpdate": "No",
	"RequireUpdateReason": "",
	"SSLEnabled": "No",
	"SSLExpireTime": ""
}


aliyun rds ModifyDBInstanceSSL \
--DBInstanceId rm-dj16n21vme51m0j9o \
--ConnectionString "" \
--SSLEnabled 0



# ModifyActionEventPolicy
aliyun rds ModifyActionEventPolicy \
--EnableEventLog False
{
	"EnableEventLog": "False",
	"RegionId": "cn-beijing",
	"RequestId": "0C8A5071-8A64-58F3-B1D6-DB159D9D86BB"
}

aliyun rds ModifyActionEventPolicy \
--EnableEventLog True



aliyun rds CreateDBInstance \
--RegionId cn-beijing \
--Engine MySQL \
--EngineVersion 8.0 \
--DBInstanceClass mysql.n2.large.25 \
--DBInstanceStorage local_ssd \
--DBInstanceNetType Internet \
--SecurityIPList "" \
--PayType Postpaid \
--EncryptionKey "" \
--InstanceNetworkType Classic






```


















.
