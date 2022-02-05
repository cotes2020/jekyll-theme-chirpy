



- [RAM](#ram)
- [RDS](#rds)




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



```


















.