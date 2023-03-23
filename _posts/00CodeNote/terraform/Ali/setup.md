



```
unset ALICLOUD_ACCESS_KEY
unset ALICLOUD_SECRET_KEY
unset ALICLOUD_SECURITY_TOKEN


aliyun configure set \
  --profile dev-admin \
  --mode StsToken \
  --region cn-beijing \
  --access-key-id $ALICLOUD_ACCESS_KEY \
  --access-key-secret $ALICLOUD_SECRET_KEY \
  --sts-token $ALICLOUD_SECURITY_TOKEN

aliyun sts GetCallerIdentity \
    --endpoint sts.us-west-1.aliyuncs.com

echo "Now login as admin"'

export ALICLOUD_ACCESS_KEY=$(echo $my_role | jq -r .access_key_id)
export ALICLOUD_SECRET_KEY=$(echo $my_role | jq -r .access_key_secret)
export ALICLOUD_SECURITY_TOKEN=$(echo $my_role | jq -r .sts_token)

export LAN=en
export REGION=cn-beijing

arc-tool config set access_key_id $ALICLOUD_ACCESS_KEY
arc-tool config set access_key_secret $ALICLOUD_SECRET_KEY
arc-tool config set region $REGION
arc-tool config set language $LAN

```
