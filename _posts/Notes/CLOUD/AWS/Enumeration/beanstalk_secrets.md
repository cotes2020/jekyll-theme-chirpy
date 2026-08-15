--------------

### Enumerating Beanstalk

--------------

- Using [PACU](https://github.com/rhinosecuritylabs/pacu)

```bash
#kali
# Kali doesn't have the new updates
pipx install git+https://github.com/RhinoSecurityLabs/pacu.git
```

- Setting up Pacu and set profile-:
  
```bash
#run
pacu
```

<img width="931" height="661" alt="image" src="https://github.com/user-attachments/assets/f43c2c92-9af2-4526-a719-9fb56eb75531" />

- Import keys with `import_keys`-:

```bash
#import keys with import_keys
import_keys profile
```
<img width="621" height="122" alt="image" src="https://github.com/user-attachments/assets/3fb1f79d-012d-4eb4-9ba6-b06dde07d461" />

- Use `search` to search through modules-:

```pacu
search
```

- Using Pacu's `elasticbeanstalk__enum` module-:

```bash
elasticbeanstalk__enum --region us-east-1
```

<img width="1893" height="411" alt="image" src="https://github.com/user-attachments/assets/d10acf20-4d89-48b4-b1cc-a3ae095d3778" />

- Bruteforce for iam permissions with Pacu-:

```bash
run iam__bruteforce_permissions

#for privesc scan check
run iam__privesc_scan
```

<img width="1280" height="633" alt="image" src="https://github.com/user-attachments/assets/96bc16a3-c65b-48ec-abde-6619e41044f1" />

- Backdooring the keys-:

<img width="1656" height="796" alt="image" src="https://github.com/user-attachments/assets/ab875987-b145-4e99-8b6b-93cb47d79612" />

- More Pacu-:

```bash
# Setting up Pacu profile
#to insert keys
set_keys
```

<img width="837" height="269" alt="image" src="https://github.com/user-attachments/assets/94648c21-365c-450d-9eb2-dca826259bf5" />

-  Secrets enumeration-:

```pacu
run secrets__enum --region
```

<img width="972" height="173" alt="image" src="https://github.com/user-attachments/assets/3d20d0f6-5c8a-4d59-a04e-daa5f8e77b45" />

- Read it at-:
```bash
cat ~/.local/share/pacu/<session name>/downloads/secrets/
```
<img width="1044" height="90" alt="image" src="https://github.com/user-attachments/assets/da00fde5-202e-40bd-8d72-59a47287de5a" />

------------


--------------
