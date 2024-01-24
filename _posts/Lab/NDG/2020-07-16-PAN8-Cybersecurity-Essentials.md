---
title: Lab - NDG - PAN8 Cybersecurity Essentials
date: 2020-07-16 11:11:11 -0400
description: Learning Path
categories: [Lab, NDG]
img: /assets/img/sample/rabbit.png
tags: [Lab, NDG]
---


# PAN8 Cybersecurity Essentials

[toc]

From the NDG: [PAN8 Cybersecurity Essentials](https://portal.netdevgroup.com/learn/pan8-ce-pilot/wb79h483WM)

![Screen Shot 2020-05-27 at 23.24.24](https://i.imgur.com/AvhpUOu.png)

---

## Lab 1: Creating a Zero Trust Environment

![Screen Shot 2020-05-27 at 23.27.20](https://i.imgur.com/TxK1iKz.png)

1. Google Chrome: https://192.168.1.254
2. Login to the Firewall web interface
3. Device > Setup > Operations > Configuration Management > `Load named configuration snapshot` > `pan8-ce-lab-01`
4.  Click the Commit link located at the top-right of the web interface.

### 1.1 Create `Zones` and Associate the Zones to Interfaces
In this section, you will create three basic zones: inside, outside, and dmz.
- A security zone segregate traffic in the Firewall so can apply security policies later to limit the traffic between zones.
- Next, you will associate them to the appropriate interfaces.

1. `Network` > `Zones`.

![Screen Shot 2020-05-27 at 23.37.00](https://i.imgur.com/NYhoanJ.png)

2. `Network` > `Interfaces`: add zone to each interface
    - ethernet1/1: Outside
    - ethernet1/2: inside
    - ethernet1/3: dmz

![Screen Shot 2020-05-27 at 23.41.34](https://i.imgur.com/PKifD7y.png)

### 1.2 Create a `Security Policy Rule`

1. `Policies` > `Security`: Add
    -   Security Policy Rule window
    -   Name: `Allow-Inside-Out`

    -   Source tab:
        -   Add Source Zone section:
        -   select `inside` from Source Zone column.
        -   allows any address in the inside zone to pass through

    -   Destination tab:
        -   Add Destination Zone section:
        -   select `outside` from Destination Zone column.
        -   allows the source traffic to communicate with any address in the outside zone.

    -   Application tab: `Any`
        -   select predefined applications to allow through the Firewall.
        -   The Any checkbox allows any application through.

    -   Service/URL Category tab: `application-default`
        -   select predefined services or preset groups to allow through the Firewall.
        -   The application-default selection means that the selected applications are allowed or denied only on their default ports defined by Palo Alto Networks.
        -   This option is recommended for allowing policies because it prevents applications from running on unusual ports and protocols, which if not intentional, can be a sign of undesired application behavior and usage. When you use this option, the device still checks for all applications on all ports, but with this configuration, applications are only allowed on their default ports/protocols. For example, if a web server is running on the standard port 80, traffic will be allowed to pass. However, if the web server is running on a non-standard port such as 5000, traffic will be blocked.

![Screen Shot 2020-05-27 at 23.42.52](https://i.imgur.com/73vhjKL.png)

![Screen Shot 2020-05-29 at 14.59.51](https://i.imgur.com/jht9HmD.png)

![Screen Shot 2020-05-29 at 15.00.04](https://i.imgur.com/iDY3aYL.png)
![Screen Shot 2020-05-29 at 15.00.15](https://i.imgur.com/WkTcJ4l.png)

![Screen Shot 2020-05-29 at 15.21.01](https://i.imgur.com/40rGbxk.png)


2.   Click on the number 3, to select but not open the interzone-default security policy.

3.   With the interzone-default policy selected, click on the `Override` button at the bottom of the center section.

4.  **Security Policy Rule – predefined** window > Actions tab:
    - checkbox: `Log at Session End`


### 1.3 Create a `NAT Policy`
1. Policies > **NAT**: Add
    - **NAT Policy Rule** window
    - Name: `Inside-NAT-Outside`
    - Original Packet tab:
      - Add of the Source Zone section.
      - the Source Zone column: `inside`
      - the Destination Zone: `outside`
    - Translated Packet tab.

![Screen Shot 2020-05-29 at 15.25.02](https://i.imgur.com/3jFp4Bs.png)

### 1.4 Commit and Test the Rules and Policies
create a basic NAT policy to NAT traffic from the inside zone to the outside zone.
1. Click the Commit link
2. Internet Explorer: https://www.facebook.com
3. Click the X in the upper-right to close Internet Explorer.
4. Monitor > Logs > **Traffic**.
5. filter: `rule eq ‘Allow-Inside-Out’`
6. see log entries allowing the facebook-base application.

![Screen Shot 2020-05-27 at 23.55.36](https://i.imgur.com/CIt1oqc.png)


---

## Lab 2: Configuring Authentication

1. Google Chrome: https://192.168.1.254
2. Login to the Firewall web interface
3. Device > Setup > Operations > Configuration Management > `Load named configuration snapshot` > `pan8-ce-lab-01`
4.  Click the Commit link located at the top-right of the web interface.


### 2.1 Configure a `Local User Account` and `Authentication Profile`
configure a `local user account`. Then, create a `local authentication profile` which will later be assigned to a security policy.

1. Network > Zones > **inside zone**.
   - **Zone** window:
     - User Identification ACL. checkbox: `Enable User Identification`
     - enable the inside zone to use a Username for authentication.

2. Device > **Local User Database** > **Users**: Add.
   - **Local User** window, type lab-web in the
     - Name: `lab-web`
     - Password: `Pal0Alt0`

3. Device > **Authentication Profile**: Add.
   - **Authentication Profile** window
     - Name: `Local-Auth-Profile`
     - Type: `Local Database`![Screen Shot 2020-05-30 at 14.14.25](https://i.imgur.com/h28iibz.png)
     - the Advanced tab: Add button > select `all`


### 2.2 Enable the `Captive Portal` and Enable `Web-Form based Logins`
enable a captive portal. In that captive portal, you will use a webform for login.

1. Device > **User Identification** > **Captive Portal Settings**.
   - the **Captive Portal Settings** tab, click on the gear icon.
   - **Captive Portal** window
     - checkbox: `Enable Captive Portal`
     - Authentication Profile: `Local-Auth-Profile`
     - This will turn on the Captive Portal for web-form logins and associate it with the Local-Auth-Profile  created earlier.

2. Objects > **Authentication**
   - checkbox: `default-web-form` > click `Clone`.
   - the **Clone** window, click the OK button to confirm the clone.

3. a new entry `default-web-form-1` has been created, click on default-web-form-1.
   - the **Authentication Enforcement** window
   - Name: `local-web-form-auth`
   - Authentication Profile: `Local-Auth-Profile`
   - OK button.


### 2.3 Create an `Authentication Policy`
enable a `captive portal`.
- A captive portal redirects web requests that match the authentication policy and forces the user to use a login to continue.
- typically seen in corporate guest networks, hotels and Wi-Fi hotspots.
- In captive portal, use a web-form for login.

1. Policies > **Authentication**: Add.
   - **Authentication Policy Rule** window
   - Name: `web-form-policy`
   - **Source** tab:
     - `Add` button in the Source Zone section > select `inside`.
   - **Destination** tab
     - `Add` button in the Destination Zone section > select `outside`.
   - **Service/URL Category** tab
     - `Add` button in the Service section > select `service-https`.
   - **Actions** tab
     - Authentication Enforcement: `local-web-form-auth`
     - OK button.


### 2.4 Commit and Test Authentication Policy
commit changes and test the authentication policy with the captive portal.
1. Click the Commit link
2. Internet Explorer: https://www.facebook.com
3. confirm the certificate error, and click `Continue to this website (not recommended)`.
   - You are seeing this error because the Firewall is intercepting traffic coming from the inside zone to the outside zone. The Firewall serves as a man-in-the-middle until authenticated.
   - You will see a vsys1 in Warnings, which refers to a virtual system in the Firewall. You can ignore it in this lab environment.
4. You will see a **web-form login**
   - lab-web as the username
   - Pal0Alt0 as the password
   - Login button.
5. You will then see Facebook after you successfully `authenticate to the Firewall as lab-web`.
6. Monitor > Logs > **Traffic**.
7. logs of entries to facebook-base are associated to the labweb user.
![Screen Shot 2020-05-29 at 23.54.46](https://i.imgur.com/stTgzcR.png)


---

## Lab 3: Using Two-Factor Authentication to Secure the Firewall

### 3.1 `Create Local-User Account`
create a local user account, lab-user. This account will be used for authentication against the Firewall.

1. Device > **Administrators** > Add.
   - **Administrator** window
     - type `lab-user` in the Name field.
     - type `Pal0Alt0` in the Password and Confirm
     - OK

![Screen Shot 2020-05-29 at 23.59.37](https://i.imgur.com/0idZtBP.png)

### 3.2 `Generate Certificates`
generate two certificates.
- The first is a self-signed Root Certificate Authority (CA) certificate, the top-most certificate in the certificate chain.
- The Firewall can use this certificate to automatically issue certificates for other uses.
- use the Root CA certificate to generate a certificate for use on the Client machine that is associated with local user account, lab-user.

1. Device > **Certificate Management** > **Certificates**:

2. `Generate` button
   - In the **Generate Certificate** window,
     - type `lab-local` in the Certificate Name field.
     - type `192.168.1.254` in the Common Name field.
     - Next, click the `Certificate Authority` checkbox.
     - Finally, click the `Generate` button.
     - OK

> This will generate a certificate for the `Firewall to act as a Certificate Authority (CA)`.
> Firewall being a CA, can now issue a certificate for the local account, lab-user.

3. `Generate` button
   - In the **Generate Certificate** window
   - type `lab-user` in the Certificate Name field.
   - type `lab-user` in the Common Name field.
   - select `lab-local` in the Signed By dropdown.
   - Finally, click the `Generate` button.
   - click OK to continue.

> Common Name as lab-user, will later be used as an authentication field, to map to the local user account, lab-user.
> now using the previous root CA certificate, lab-local, to sign this certificate.


### 3.3 Create a `Certificate Profile`
create a certificate profile
- defines user and device authentication for multiple services on the Firewall.
- The profile specifies which certificates to use, how to verify certificate revocation status and how that status constrains access.
- In this lab, the certificate profile is created to tell the Firewall to use the common-name of the certificate as a username.
- Then, you will tell the Firewall to use this Certificate Profile to authenticate users.


1. Device > **Certificate Management** > **Certificate Profile**: Add
   - In the **Certification Profile** window
     - type `Cert-Local-Profile` in the Name field.
     - select `Subject` in the Username Field dropdown.
     - Next, `Add` button.
       - select `lab-local` in the CA Certificate dropdown.
       - Then, click the OK button.
     - click the OK button.

> Username Field set to Subject, will use the common-name as the default. The Firewall will now use that common-name as the username.
> The lab-user certificate you generated earlier has a common-name of lab-user, and will therefore use lab-user to authenticate the client machine

> Firewall will use lab-local CA certificate to verify the authenticity of the client supplied certificate, lab-user.


2. Device > **Setup** > **Management**.
   - Click the gear icon on the **Authentication Settings** section, located in the center.
   - In the **Authentication Settings** window
     - select `Cert-Local-Profile` for the Certification Profile dropdown.
     - click on the OK button.


### 3.4 Export Certificate and Commit
export the lab-user certificate generated on the Firewall.
commit changes, causing the Firewall to start using certificates for authentication.

1. Device > **Certificate Management** > **Certificates**: `lab-user` certificate > Export button.
   - In the **Export Certificate - lab-user** window,
     - File Format: `Encrypted Private Key and Certificate (PKCS12)`
     - type `paloalto` for the Passphrase and Confirm Passphrase fields,
     - OK button.

![Screen Shot 2020-05-30 at 01.05.46](https://i.imgur.com/NsjHyDR.png)

> By using an Encrypted Private Key and Certificate, this creates an additional security measure, as the passphrase is required to install the certificate on a client machine.

2. The `cert_lab-user.p12 file` will download to the Client machine’s Downloads folder.

3. Click the Commit link


### 3.5 Test Connectivity and Import Certificate on the Client
test connectivity to the Firewall. import the lab-user certificate on the Client machine and try again.

1. Open Internet Explorer: https://192.168.1.254

> This message is displayed because the Firewall has a `self-signed certificate` by default.
> The client does not have a Certificate Authority that can validate the certificate.

2. Click the `Continue to this website (not recommended)`.

3. You will see a “The webpage cannot be found” message. close Internet Explorer.

> Notice you get a HTTP 400 Bad Request error.
> This is because the lab-user certificate is not installed on the Client machine.
> The Firewall administrators are not allowed to login unless they have the certificate installed and have an account and password.
> These two factors make up the Two-Factor Authentication in this lab.

4. To install the lab-user certificate, double-click `certificates.msc` on the Desktop.

> This launches the Client’s Certificate Manager snap-in for the Management Console.

5. File > `Add/Remove Snap-in`
   - select Certificates on the left.
   - click the Add button to move Certificates to the right.


6. In the Certificates snap-in window, make sure `My user account` is selected. Then click the Finish button.

7.  In the Add or Remove Snap-ins window
    - `Certificates (Local Computer)` and `Certificates – Current User`, under Selected snap-ins. Then, click the OK button.


8.  Click `Certificates – Current User` in the left side pane, to expand the section.


9.  Right-click on the Personal folder > All Tasks > `Import`

10. In the **Certificate Import Wizard** window
    - click the Next button.
    - click the Browse… button.
    - select Personal Information Exchange (*.pfx;*.p12) from the File Type
    - select the `cert_lab-user.p12` file.
    - Finally, click the Open button.
    - click the Next button.
    - type `paloalto`. the passphrase entered when you exported the certificate from the Firewall.
    - Then, click the Next button.
    - leave the default Personal in the **Certificate Store** field.
    - Then, click the Next button.
    - click the Finish Button.
    - click the OK button.
    - close the Certificate Manager.
    - In the **Microsoft Management Console** window, click the Yes button to save console settings.

11.  Open Internet Explorer: https://192.168.1.254

12.  In the **Windows Security** window, click the OK button to confirm the certificate just installed on the Client machine.

13. Click the `Continue to this website (not recommended)`.

14. The Confirm Certificate window will pop-up again > OK to confirm.

15. The Firewall login will be displayed.
    - Username: `lab-user`
    - Type `Pal0Alt0` for the Password field.
    - Then, click the Log In button.

> lab-user is pre-populated for the Username because the Certificate Profile you created earlier used the subject, common-name for the Username field.


16.  On the Welcome window, click the Close button.

17.  now at the Palo Alto Networks Web GUI, logged on as lab-user. Notice the username in the lower-left.

---

## Lab 4: Allowing Only Trusted Applications

### 4.1 Create an `Application Group`
To simplify the creation of security policies, applications requiring the same security settings can be combined by creating an application group.

1. Objects > Application Groups: Add
   - Application Group window,
     - type Trusted-Apps in the Name field.
     - click the Add button
     - type facebook in the search box
     - type dns in the search box
     - click the OK button.

![Screen Shot 2020-05-30 at 14.15.03](https://i.imgur.com/PcM0UPf.png)

### 4.2 Modify Security Policy
modify the Allow-Inside-Out security policy to only allow the applications in the application group, Trusted-Apps

1. Policies > Security: `Allow-Inside-Out` policy.
   - **Security Policy Rule** window
   - Application tab: Add button.
     - select `Trusted-Apps` under Application group.


### 4.3 Commit and Test
commit changes to the Firewall.
Then, test the security policy
Next, add an additional application to the application group, Trusted-Apps.
verify the additional application is allowed.

1. Click the Commit link
2. Internet Explorer: https://www.facebook.com
3. the address bar: https://www.google.com

> The Firewall recognizes the traffic and matches it to the application, google. As google is NOT part of the application group, Trusted-Apps, the Firewall blocks the traffic based on the security policy

4. Monitor > Logs > Traffic.
   - the application `facebook-base` has the action of allow based on rule `Allow-Inside-Out`.
   - the application `web- browsing` has the action reset-both based on the rule `interzone-default`, which has a session end reason of policy-deny.
   - the application `dns` has the action of allow based on the rule `Allow-Inside-Out`.

> the traffic to www.google.com is not part of the Trusted-Apps application group, which applied to the Allow-Inside-Out security policy,
> the Firewall matches that traffic to the next appropriate policy. match to interzone-default, which has an explicit deny action.


5.  Objects > Application Groups: Trusted-Apps Application Group
   - Add button.
   - type google in the search box
   - click on google-base under Application.
   - OK button.
6.  Click the Commit link


7.  Internet Explorer: https://www.google.com

> www.google.com now works because it was added to the Trusted-Apps application group.

---

## Lab 5: Managing Certificates

### 5.1 Generate Certificates

generate two certificates.
- The first is a `self-signed` `Root Certificate Authority (CA) certificate`, the top-most certificate in the certificate chain.
- The Firewall can use this certificate to automatically issue certificates for other uses.
- use the Root CA certificate to generate a new certificate for the Firewall to use for Inbound Management Traffic, replacing the default certificate issued specifically for this lab environment.


1. Device > **Certificate Management** > **Certificates**: `Generate` button at the bottom-center of the center section.
   - **Generate Certificate** window,
     - type `lab-firewall` in the Certificate Name field.
     - type `203.0.113.20` in the Common Name field.
     - Next, click the `Certificate Authority` checkbox.
     - Then, select sha512 in the Digest dropdown.
     - Next, type 1095 in the Expiration (days) field.
     - Finally, click the Generate button.

![Screen Shot 2020-05-30 at 20.40.10](https://i.imgur.com/GXOzyAm.png)

> generate a certificate for the Firewall to act as a root Certificate Authority (CA).
> The IP address, `203.0.113.20, used in the Common Name field is the Firewall’s outside IP address`. It is best practice that a digest algorithm of sha256 or higher is used for enhanced security. By increasing the default digest to sha512, you have created a much stronger certificate.
> The Expiration (days) field is equivalent to 3 years (365 days x 3 years = 1,095 days).

   - **Generate Certificate** window
     - type `lab-management` in the Certificate Name field.
     - type `192.168.1.254` in the Common Name field.
     - select `lab-firewall` in the Signed By dropdown.
       - click the Add button in the Certificate Attributes section.
         - Type: `Organization`
         - Value: `Palo Alto Networks`
       - click the Add button in the Certificate Attributes section.
         - Type: `Email`
         - Value: `support@paloaltonetworks.com`
       - click the Add button in the Certificate Attributes section.
         - Type: `Department`
         - Value: `Management Interface`
     - click the Generate button.

> The IP address, `192.168.1.254, used in the Common Name field is the Firewall’s inside IP address`.
> selected the previously created root CA certificate, lab-firewall, to sign this certificate.
> Client certificates that are used when requesting firewall services that rely on TLSv1.2 (such as administrator access to the web interface) cannot have sha512 as a digest algorithm, therefore you will leave the default sha256.
>
> Certificate Attributes are used to uniquely identify the firewall and the service that will use the certificate.




1.  In the Generate Certificate window, review the settings. Then, click the Generate button.
2.  In the Generate Certificate window, click OK to continue.
  7/2/2018 Copyright © 2018 Network Development Group, Inc. www.netdevgroup.com Page 13

> Palo Alto Networks Firewalls use certificates in the following applications:
> • **User authentication** for Captive Portal, GlobalProtectTM, Mobile Security Manager, and web interface access to a firewall or Panorama.
> • **Device authentication** for GlobalProtect VPN (remote user- to-site or large scale).
> • **Device authentication** for IPSec site-to-site VPN with Internet Key Exchange (IKE).
> • **Decrypting inbound and outbound SSL traffic**. As a best practice, it is recommended you use different certificates for each usage.

> In a real-world scenario, you can simplify your certificate deployment by using a certificate that the client systems already trust. It is recommended that you import a certificate and private key from your enterprise certificate authority (CA) or obtain a certificate from an external CA. The trusted root certificate store of the client systems is likely to already have the associated root CA certificate that ensures trust. This prevents you from having to create a root CA certificate and install it on every client system to prevent a certificate error.


### 5.2 Replace the Certificate for Inbound Management Traffic
replace the certificate for inbound management traffic.
- boot the Firewall for the first time, it automatically generates a default certificate that enables HTTPS access to the web interface over the management (MGT) interface.

- To improve the security of inbound management traffic,
  - configure a SSL/TLS Service Profile
  - replace the default certificate with the lab-management certificate created
  - apply the SSL/TLS Service Profile to inbound management traffic.

1. Device > Certificate Management > **SSL/TLS Service Profile** > Add.
   - **SSL/TLS Service Profile** window,
     - type `Management` in the Name field.
     - select `lab-management` from the Certificate
     - select `TLSv1.1` from the Min Version
     - Finally, click the OK button.

2. Device > Setup > Management: the gear icon on the General Settings section, located in the center.
   - **General Settings** window,
     - select `Management` from the SSL/TLS Service Profile dropdown
     - click the OK button.


### 5.3 Export Certificate and Commit
export the root CA certificate, lab-firewall. Then, you will commit your changes to the Firewall.

1. Device > Certificate Management > Certificates: `lab-firewall`: `Export` button at the bottom.
   - the **Export Certificate - lab-firewall** window,
     - select `Encrypted Private Key and Certificate (PKCS12)` in the File Format
     - type `paloalto` for the Passphrase
     - OK button.

> By using an Encrypted Private Key and Certificate, this creates an additional security measure, as the passphrase is required to install the certificate on a client machine.

2. The cert_lab-firewall.p12 file will download to the Client machine’s Downloads folder.
3. Click the Commit link

> Notice the warning about the Web server being restarted, this is because of the authentication changes you made. You will need to click the Close button when it gets to 99%, since the web server is restarting, you will not see it get to 100%.

4. Click the X in the upper-right to close Google Chrome.


### 5.4 Test Connectivity and Import Certificate on the Client

test connectivity to the Firewall.
- When establishing a secure connection with the Firewall, the client must trust the root CA that issued the certificate.
- Otherwise, the client browser will display a warning that the certificate is invalid and might (depending on security settings) block the connection.
- To prevent this, you will import the root CA certificate on the Client, creating a trust relationship between the Firewall and the Client machine. Then, you will test connectivity again.


1. Google Chrome: https://192.168.1.254
2. You will see a “Your connection is not private” message.
   - This is because the Client cannot verify the certificate from the Firewall. To view the certificate, click the ! icon (triangle exclamation) in the address bar.

![Screen Shot 2020-05-30 at 20.53.04](https://i.imgur.com/JGlTaKC.png)

3. In the popup > Details link > View certificate under the Certificate Error section > Details tab > Subject field.

> Notice the details match the lab-management certificate you created earlier in section 5.1. The sha256 algorithm is being used. The certificate was issued by 203.0.113.20, which is the common-name of the root CA certificate, lab-firewall, you created. The Valid from and Valid to fields indicate the certificate is valid for 365 days. In the center, you will see the certificate attributes you set when you generated the certificate. The Public key is using RSA (2048 Bits).

4. To install the lab-firewall certificate, double-click `certificates.msc` on the Desktop. This launches the Client’s Certificate Manager snap-in for the Management Console.
   - In the **User Account Control** window, click the Yes button.
   - In the **certificates – [Console Root]** window,
     - click `Certificates (Local Computer)` in the left side pane, to expand the section.
     - right-click the Trusted Root Certification Authorities folder. Then, click on All Tasks > Import...
     - Certificate Import Wizard window,
       - click the Browse... button.
       - select Personal Information Exchange (*.pfx;*.p12) from the File Type dropdown in the lower- right.
       - select the cert_lab-firewall.p12 file. Finally, click the Open button.
       - click the Next button.
       - type paloalto: passphrase
       - Next button.
       - leave the default `Trusted Root Certification Authorities`, in the Certificate Store field. Then, click the Next button.
       - click the Finish Button.OK button.


5. Google Chrome: https://192.168.1.254
6. login prompt from the Firewall.
   - Notice the ! icon in the address bar from before is now showing a secured padlock icon. Click on the padlock icon.
   - In the popup, click the Details link
   - In the Developer Tools pane on the right, notice the message “This page is secure (valid HTTPS)”. Below, under the Valid Certificate section, you will see the message “The connection to this site is using a valid, trusted server certificate.”


---

## Lab 6: Using the Application Command Center to Find Threats

6.1 Generate Malware Traffic to the Firewall

1. PuTTY: `traffic-generator` > Load button
   - type Pal0Alt0
   - sh /tg/malware.sh
   - Wait 10 minutes to let the script generate malware traffic.

### 6.2 Find Malware Threat in the Application Command Center
review Threat Activity and Blocked Activity in the Application Command Center.

1. ACC > **Threat Activity**
2. a bar graph of the types of threats identified and list of threats under the Threat Name.

3. Blocked Activity tab: Blocked User Activity section.


---

## Lab 7: Decrypting SSL Inbound Traffic

### 7.1 `Download the SSL Certificate` from DMZ Server
use `WinSCP` to download the certificate and key that is being used on the DMZ server.
- WinSCP is a free, open-source tool, used to transfer secure files between clients.


1. open WinSCP
2. In the Login window,
   - click on `edl-webserver` on the left. Then, click the `Login` button.
   - Select `/ <root>` from the dropdown in the top-middle.
   - Then, double-click on `ssl-inbound`.
   - Press CTRL + A to highlight `dmz-server.key` and `dmz-server.crt`.
   - Then, click the Download button at the top.

3. terminate the edl-webserver session.


### 7.2 `Import SSL Certificate`
import the SSL Certificate you downloaded from the DMZ server to the Firewall. This will later be used to create a decryption profile.


1. Device > Certificate Management > Certificates: `Import` button at the bottom-center of the center section.
   - **Import Certificate** window
     - name: type `SSL Inbound Cert`.
     - Browse: `dmz-server.crt`.
     - Click the checkbox for Import private key.
       - Browse: `dmz-server.key`.
     - type paloalto for the Passphrase
     - OK button.

2. Verify the SSL Inbound Cert is showing a status of valid.


### 7.3 Create a `Decryption Profile`
create a decryption profile.
- Decryption profiles allow administrators to perform checks on both decrypted traffic and traffic that would have been excluded from decryption.
- After a decryption profile is created, it can then be attached to a decryption policy rule that will enforce the profile settings.

1. Objects > Decryption Profile > Add
   - **Decryption Profile** window,
   - name: type `SSL Inbound Inspection`.
   - Then, click the OK button.
2. Verify the SSL Inbound Inspection Decryption Profile was created.


### 7.4 Create a `Decryption Policy`
create a decryption policy. Decryption Policies allow administrators to stop threats that would otherwise remain hidden in encrypted traffic and help prevent sensitive content from leaving an organization.

1. Policies > **Decryption**: Add.
   - Decryption Policy Rule window,
     - type `Decrypt SSL Inbound Inspection` in the Name field.
     - Source tab.
       - Add in the Source Zone section: `inside`.
     - Destination tab
       - Add in the Destination Zone section: `dmz`.
     - Service/URL Category tab.
       - Add in the Service column: `service-http`.
       - Add in the Service column: `service-https`.
     - Options tab.
       - select Decrypt for the Action.
       - select `SSL Inbound Inspection` in the Type
       - select `SSL Inbound Cert` in the Certificate
       - select `SSL Inbound Inspection` in the Decryption Profile field.
       - Finally, click the OK button.

### 7.5 Commit and Test Decryption Policy
test the decryption policy you created earlier.

1. Click the Commit link
2. Monitor > Logs > Traffic.
   - search box, type ( addr.dst in 192.168.50.10 ) and press Enter.
   - add Columns and select Decrypted.

3. Google Chrome: https://192.168.50.10
4.  You will see a “Your connection is not private” message. Click on the ADVANCED link.
5.  Click on Proceed to 192.168.50.10 (unsafe)

6.  Notice that the Apache HTTP Server Test page is working properly. Click on the X on the tab to close it.


7.  Monitor > Logs > **Traffic**: refresh icon.
8.  Look for traffic associated with the application of ssl and the Decrypted column is set to yes.
    - Open the Detailed Log View of the traffic to analyze the traffic from the Client machine of 192.168.1.20 to the DMZ server of 192.168.50.10.
    - In the **Detailed Log View** window
      - the Destination section: 192.168.50.10 and Port 443 to the dmz zone of the DMZ server.
      - Then, in the Flags section, notice the flag Decrypted is set and click the Close button.

### 7.6 Disable Decryption Policy
disable the decryption policy you created earlier. Then, after committing the changes to the Firewall, you will monitor traffic logs to determine if traffic is still being decrypted.


1. Policies > Decryption: the **Decrypt SSL Inbound Inspection policy**: Disable button.
2. Click the Commit link
3. Google Chrome: https://192.168.50.10
4. the Apache HTTP Server Test page is working properly.


5. Monitor > Logs > Traffic. Then, click the refresh icon.
6. Look for traffic associated with the application of ssl and the Decrypted column is set to no.
   - Open the Detailed Log View of the traffic to analyze the traffic from the Client machine of 192.168.1.20 to the DMZ server of 192.168.50.10.
   - Detailed Log View window,
   - Destination section: 192.168.50.10 and Port 443 to the dmz zone of the DMZ server.
   - Flags section, notice the flag for Decrypted is not set.





















.
