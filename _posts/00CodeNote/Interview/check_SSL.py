""" Alibaba Cloud Function Compute Example """

import datetime
import json
import socket
import ssl
import sys
import time

import myemail
import myhtml

PROGRAM_MODE_CMDLINE = 0  # The program operates from the command line
# The program operates as a Alibaba Cloud Function Compute function
PROGRAM_MODE_ACS_FUNC = 1

g_program_mode = PROGRAM_MODE_ACS_FUNC
# g_program_mode = PROGRAM_MODE_CMDLINE

g_days_left = 14  # Warn if a certificate will expire in less than this number of days

g_no_send = False  # if set, don't actually send an email. This is used for debugging

# If set, only send emails if a certificate will expire soon or on error
g_only_send_notices = False

# This is set during processing if a warning or error was detected
g_email_required = False

g_hostnames = [
    "neoprime.xyz",
    "api.neoprime.xyz",
    "cdn.neoprime.xyz",
    "www.neoprime.xyz",
]

email_params = {"To": "", "Subject": "", "Body": "", "BodyText": ""}
email_params["To"] = "someone@example.com"
email_params["Subject"] = "NeoPrime SSL Certificate Status Report"

dm_account = {
    "Debug": 0,  # Debug flag
    "Account": "",  # DirectMail account
    "Alias": "",  # DirectMail alias
    "host": "",  # HTTP Host header
    "url": "",  # URL for POST
}
# From the DirectMail Console
dm_account["Debug"] = 0
dm_account["Account"] = ""
dm_account["Alias"] = ""
dm_account["host"] = "dm.ap-southeast-1.aliyuncs.com"
dm_account["url"] = "https://dm.ap-southeast-1.aliyuncs.com/"


def add_row(body, domain, status, expires, issuerName, names, flag_hl):
    """Add a row to the HTML table"""
    # build the url
    url = '<a href="https://' + domain + '">' + domain + "</a>"
    # begin a new table row
    if flag_hl is False:
        body += "<tr>\n"
    else:
        body += '<tr bgcolor="#FFFF00">\n'  # yellow
    body += "<td>" + url + "</td>\n"
    body += "<td>" + status + "</td>\n"
    body += "<td>" + expires + "</td>\n"
    body += "<td>" + issuerName + "</td>\n"
    body += "<td>" + names + "</td>\n"
    return body + "</tr>\n"


# Email specific
def send(account, credentials, params):
    """email send function"""
    # pylint: disable=global-statement
    global g_only_send_notices
    global g_email_required
    # If set, only send emails if a certificate will expire soon or on error
    if g_only_send_notices is True:
        if g_email_required is False:
            print("")
            print("All hosts have valid certificates")
            print("Sending an email is not required")
            return
    myemail.sendEmail(credentials, account, params, g_no_send)


def ssl_get_cert(hostname):
    """
    This function returns an SSL certificate from a host
    This SSL certificate contains information about the certificate
    such as the domain name, and expiration date.
    """
    context = ssl.create_default_context()
    conn = context.wrap_socket(socket.socket(socket.AF_INET), server_hostname=hostname)
    # 3 second timeout because Function Compute has runtime limitations
    conn.settimeout(3.0)
    try:
        conn.connect((hostname, 443))
    except Exception as ex:
        print(f"{hostname}: Exception: {ex}", file=sys.stderr)
        return False, str(ex)
    host_ssl_info = conn.getpeercert()
    return host_ssl_info, ""


def get_ssl_info(host):
    """
    This function retrieves the SSL certificate for host
    If we receive an error, retry up to three times waiting 10 seconds each time.
    """
    retry = 0
    err = ""
    while retry < 3:
        ssl_info, err = ssl_get_cert(host)
        if ssl_info is not False:
            return ssl_info, ""
        retry += 1
        print("    retrying ...")
        time.sleep(10)
    return False, err


def get_ssl_issuer_name(ssl_info):
    """Return the IssuerName from the SSL certificate"""
    issuerName = ""
    issuer = ssl_info["issuer"]
    # pylint: disable=line-too-long
    # issuer looks like this:
    # This is a set of a set of a set of key / value pairs.
    # ((('countryName', 'US'),), (('organizationName', "Let's Encrypt"),), (('commonName', "Let's Encrypt Authority X3"),))
    for item in issuer:
        # item will look like this as it goes thru the issuer set
        # Note that this is a set of a set
        #
        # (('countryName', 'US'),)
        # (('organizationName', "Let's Encrypt"),)
        # (('commonName', "Let's Encrypt Authority X3"),)
        s = item[0]
        # s will look like this as it goes thru the isser set
        # Note that this is now a set
        # ('countryName', 'US')
        # ('organizationName', "Let's Encrypt")
        # ('commonName', "Let's Encrypt Authority X3")
        # break the set into "key" and "value" pairs
        k = s[0]
        v = s[1]
        if k == "organizationName":
            if v != "":
                issuerName = v
                continue
        if k == "commonName":
            if v != "":
                issuerName = v
    return issuerName


def get_ssl_subject_alt_names(ssl_info):
    """Return the Subject Alt Names"""
    altNames = ""
    subjectAltNames = ssl_info["subjectAltName"]
    index = 0
    for item in subjectAltNames:
        altNames += item[1]
        index += 1
        if index < len(subjectAltNames):
            altNames += ", "
    return altNames


def process_hostnames(msg_body, hostnames):
    """
    Process the SSL certificate for each hostname
    """

    # pylint: disable=global-statement
    global g_email_required

    ssl_date_fmt = r"%b %d %H:%M:%S %Y %Z"

    for host in hostnames:
        f_expired = False
        print("Processing host:", host)

        ssl_info, err = get_ssl_info(host)
        if ssl_info is False:
            msg_body = add_row(msg_body, host, err, "", "", "", True)
            g_email_required = True
            continue

        # print(ssl_info)
        issuerName = get_ssl_issuer_name(ssl_info)
        altNames = get_ssl_subject_alt_names(ssl_info)
        l_expires = datetime.datetime.strptime(ssl_info["notAfter"], ssl_date_fmt)
        remaining = l_expires - datetime.datetime.utcnow()
        if remaining < datetime.timedelta(days=0):
            # cert has already expired - uhoh!
            cert_status = "Expired"
            f_expired = True
            g_email_required = True
        elif remaining < datetime.timedelta(days=g_days_left):
            # expires sooner than the buffer
            cert_status = "Time to Renew"
            f_expired = True
            g_email_required = True
        else:
            # everything is fine
            cert_status = "OK"
            f_expired = False
        msg_body = add_row(
            msg_body, host, cert_status, str(l_expires), issuerName, altNames, f_expired
        )
    return msg_body


def main_cmdline():
    """This is the main function"""

    # My library for processing Alibaba Cloud Services (ACS) credentials
    # This library is only used when running from the desktop and not from the cloud
    import mycred_acs

    # Load the Alibaba Cloud Credentials (AccessKey)
    cred = mycred_acs.LoadCredentials()

    if cred is False:
        print("Error: Cannot load credentials", file=sys.stderr)
        sys.exit(1)

    now = datetime.datetime.utcnow()
    date = now.strftime("%a, %d %b %Y %H:%M:%S GMT")

    msg_body = ""
    msg_body = myhtml.build_body_top()
    msg_body += "<h4>NeoPrime SSL Cerificate Status Report</h4>"
    msg_body += date + "<br />"
    msg_body += "<br />"
    msg_body = myhtml.build_table_top(msg_body)

    # This is where the SSL processing happens
    msg_body = process_hostnames(msg_body, g_hostnames)
    msg_body = myhtml.build_table_bottom(msg_body)
    msg_body = myhtml.build_body_bottom(msg_body)
    email_params["Body"] = msg_body
    email_params["BodyText"] = ""
    # print(msg_body)
    send(dm_account, cred, email_params)


def main_acs_func(event, context):
    """This is the main function"""
    cred = {"accessKeyId": "", "accessKeySecret": "", "securityToken": "", "Region": ""}
    cred["accessKeyId"] = context.credentials.accessKeyId
    cred["accessKeySecret"] = context.credentials.accessKeySecret
    cred["securityToken"] = context.credentials.securityToken

    now = datetime.datetime.utcnow()
    date = now.strftime("%a, %d %b %Y %H:%M:%S GMT")

    msg_body = ""
    msg_body = myhtml.build_body_top()
    msg_body += "<h4>NeoPrime SSL Cerificate Status Report</h4>"
    msg_body += date + "<br />"
    msg_body += "<br />"
    msg_body = myhtml.build_table_top(msg_body)

    # This is where the SSL processing happens
    msg_body = process_hostnames(msg_body, g_hostnames)
    msg_body = myhtml.build_table_bottom(msg_body)
    msg_body = myhtml.build_body_bottom(msg_body)
    email_params["Body"] = msg_body
    email_params["BodyText"] = ""
    # print(msg_body)
    send(dm_account, cred, email_params)
    return msg_body


def handler(event, context):
    """This is the Function Compute entry point"""
    body = ""
    body = main_acs_func(event, context)
    res = {
        "isBase64Encoded": False,
        "statusCode": 200,
        "headers": {"content-type": "text/html"},
        "body": body,
    }
    return json.dumps(res)


# Main Program
if g_program_mode == PROGRAM_MODE_CMDLINE:
    main_cmdline()
