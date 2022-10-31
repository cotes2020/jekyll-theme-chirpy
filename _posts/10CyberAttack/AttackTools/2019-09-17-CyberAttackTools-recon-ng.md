---
title: Meow's Testing Tools - Recon-ng
date: 2019-09-17 11:11:11 -0400
categories: [10CyberAttack, CyberAttackTools]
tags: [CyberAttack, CyberAttackTools]
toc: true
image:
---

# Recon-ng

[toc]

---

# Recon-ng

```bash

git clone https://github.com/lanmaster53/recon-ng.git
cd recon-ng
pip install -r REQUIREMENTS

recon-ng
marketplace install all i

workspaces create tesla
workspaces load tesla

modules load hackertarget
modules load recon/domains-hosts/hackertarget

options set SOURCE tesla.com



[recon-ng][Mycompany] > show companies

  +------------------------------------------------------+
  | rowid | company | description | notes |    module    |
  +------------------------------------------------------+
  | 1     | Mycompany   |             |       | user_defined |
  +------------------------------------------------------+

[*] 1 rows returned
[recon-ng][Mycompany] > marketplace search

  +---------------------------------------------------------------------------------------------------+
  |                        Path                        | Version |     Status    |  Updated   | D | K |
  +---------------------------------------------------------------------------------------------------+
  | dev/spyse_subdomains                               | 1.0     | not installed | 2020-07-07 |   | * |
  | discovery/info_disclosure/cache_snoop              | 1.0     | installed     | 2019-06-24 |   |   |
  | discovery/info_disclosure/interesting_files        | 1.1     | installed     | 2020-01-13 |   |   |
  | exploitation/injection/command_injector            | 1.0     | installed     | 2019-06-24 |   |   |
  | exploitation/injection/xpath_bruter                | 1.2     | installed     | 2019-10-08 |   |   |
  | import/csv_file                                    | 1.1     | installed     | 2019-08-09 |   |   |
  | import/list                                        | 1.1     | installed     | 2019-06-24 |   |   |
  | import/masscan                                     | 1.0     | installed     | 2020-04-07 |   |   |
  | import/nmap                                        | 1.0     | installed     | 2019-06-24 |   |   |
  | recon/companies-contacts/bing_linkedin_cache       | 1.0     | installed     | 2019-06-24 |   | * |
  | recon/companies-contacts/censys_email_address      | 1.0     | disabled      | 2019-08-22 |   | * |
  | recon/companies-contacts/pen                       | 1.1     | installed     | 2019-10-15 |   |   |
  | recon/companies-domains/censys_subdomains          | 1.0     | disabled      | 2019-08-22 |   | * |
  | recon/companies-domains/pen                        | 1.1     | installed     | 2019-10-15 |   |   |
  | recon/companies-domains/viewdns_reverse_whois      | 1.0     | installed     | 2019-08-08 |   |   |
  | recon/companies-domains/whoxy_dns                  | 1.1     | installed     | 2020-06-17 |   | * |
    | recon/domains-companies/censys_companies           | 1.0     | disabled      | 2019-08-22 |   | * |
    | recon/domains-companies/pen                        | 1.1     | installed     | 2019-10-15 |   |   |
    | recon/domains-companies/whoxy_whois                | 1.1     | installed     | 2020-06-24 |   | * |
    | recon/domains-contacts/hunter_io                   | 1.3     | installed     | 2020-04-14 |   | * |
    | recon/domains-contacts/metacrawler                 | 1.1     | disabled      | 2019-06-24 | * |   |
    | recon/domains-contacts/pen                         | 1.1     | installed     | 2019-10-15 |   |   |
    | recon/domains-contacts/pgp_search                  | 1.4     | installed     | 2019-10-16 |   |   |
    | recon/domains-contacts/whois_pocs                  | 1.0     | installed     | 2019-06-24 |   |   |
    | recon/domains-contacts/wikileaker                  | 1.0     | installed     | 2020-04-08 |   |   |
    | recon/domains-credentials/pwnedlist/account_creds  | 1.0     | disabled      | 2019-06-24 | * | * |
    | recon/domains-credentials/pwnedlist/api_usage      | 1.0     | installed     | 2019-06-24 |   | * |
    | recon/domains-credentials/pwnedlist/domain_creds   | 1.0     | disabled      | 2019-06-24 | * | * |
    | recon/domains-credentials/pwnedlist/domain_ispwned | 1.0     | installed     | 2019-06-24 |   | * |
    | recon/domains-credentials/pwnedlist/leak_lookup    | 1.0     | installed     | 2019-06-24 |   |   |
    | recon/domains-credentials/pwnedlist/leaks_dump     | 1.0     | installed     | 2019-06-24 |   | * |
    | recon/domains-credentials/scylla                   | 1.3     | installed     | 2020-09-25 |   |   |
    | recon/domains-domains/brute_suffix                 | 1.1     | installed     | 2020-05-17 |   |   |
    | recon/domains-hosts/binaryedge                     | 1.2     | installed     | 2020-06-18 |   | * |
    | recon/domains-hosts/bing_domain_api                | 1.0     | installed     | 2019-06-24 |   | * |
    | recon/domains-hosts/bing_domain_web                | 1.1     | installed     | 2019-07-04 |   |   |
    | recon/domains-hosts/brute_hosts                    | 1.0     | installed     | 2019-06-24 |   |   |
    | recon/domains-hosts/builtwith                      | 1.0     | installed     | 2019-06-24 |   | * |
    | recon/domains-hosts/censys_domain                  | 1.0     | disabled      | 2019-08-22 |   | * |
    | recon/domains-hosts/certificate_transparency       | 1.2     | installed     | 2019-09-16 |   |   |
    | recon/domains-hosts/google_site_web                | 1.0     | installed     | 2019-06-24 |   |   |
    | recon/domains-hosts/hackertarget                   | 1.1     | installed     | 2020-05-17 |   |   |
    | recon/domains-hosts/mx_spf_ip                      | 1.0     | installed     | 2019-06-24 |   |   |
    | recon/domains-hosts/netcraft                       | 1.1     | installed     | 2020-02-05 |   |   |
    | recon/domains-hosts/shodan_hostname                | 1.1     | installed     | 2020-07-01 | * | * |
    | recon/domains-hosts/ssl_san                        | 1.0     | not installed | 2019-06-24 |   |   |
    | recon/domains-hosts/threatcrowd                    | 1.0     | not installed | 2019-06-24 |   |   |
    | recon/domains-hosts/threatminer                    | 1.0     | not installed | 2019-06-24 |   |   |
    | recon/domains-vulnerabilities/ghdb                 | 1.1     | not installed | 2019-06-26 |   |   |
    | recon/domains-vulnerabilities/xssed                | 1.0     | not installed | 2019-06-24 |   |   |
  | recon/companies-hosts/censys_org                   | 1.0     | disabled      | 2019-08-22 |   | * |
  | recon/companies-hosts/censys_tls_subjects          | 1.0     | disabled      | 2019-08-22 |   | * |
  | recon/companies-multi/github_miner                 | 1.1     | installed     | 2020-05-15 |   | * |
  | recon/companies-multi/shodan_org                   | 1.1     | installed     | 2020-07-01 | * | * |
  | recon/companies-multi/whois_miner                  | 1.1     | installed     | 2019-10-15 |   |   |
  | recon/contacts-contacts/abc                        | 1.0     | installed     | 2019-10-11 | * |   |
  | recon/contacts-contacts/mailtester                 | 1.0     | installed     | 2019-06-24 |   |   |
  | recon/contacts-contacts/mangle                     | 1.0     | installed     | 2019-06-24 |   |   |
  | recon/contacts-contacts/unmangle                   | 1.1     | installed     | 2019-10-27 |   |   |
  | recon/contacts-credentials/hibp_breach             | 1.2     | installed     | 2019-09-10 |   | * |
  | recon/contacts-credentials/hibp_paste              | 1.1     | installed     | 2019-09-10 |   | * |
  | recon/contacts-credentials/scylla                  | 1.3     | installed     | 2020-09-14 |   |   |
  | recon/contacts-domains/migrate_contacts            | 1.1     | installed     | 2020-05-17 |   |   |
  | recon/contacts-profiles/fullcontact                | 1.1     | installed     | 2019-07-24 |   | * |
  | recon/credentials-credentials/adobe                | 1.0     | installed     | 2019-06-24 |   |   |
  | recon/credentials-credentials/bozocrack            | 1.0     | installed     | 2019-06-24 |   |   |
  | recon/credentials-credentials/hashes_org           | 1.0     | installed     | 2019-06-24 |   | * |
  | recon/hosts-domains/migrate_hosts                  | 1.1     | not installed | 2020-05-17 |   |   |
  | recon/hosts-hosts/bing_ip                          | 1.0     | not installed | 2019-06-24 |   | * |
  | recon/hosts-hosts/censys_hostname                  | 1.0     | not installed | 2019-08-22 |   | * |
  | recon/hosts-hosts/censys_ip                        | 1.0     | not installed | 2019-08-22 |   | * |
  | recon/hosts-hosts/censys_query                     | 1.0     | not installed | 2019-08-22 |   | * |
  | recon/hosts-hosts/ipinfodb                         | 1.1     | not installed | 2020-06-08 |   | * |
  | recon/hosts-hosts/ipstack                          | 1.0     | not installed | 2019-06-24 |   | * |
  | recon/hosts-hosts/resolve                          | 1.0     | not installed | 2019-06-24 |   |   |
  | recon/hosts-hosts/reverse_resolve                  | 1.0     | not installed | 2019-06-24 |   |   |
  | recon/hosts-hosts/ssltools                         | 1.0     | not installed | 2019-06-24 |   |   |
  | recon/hosts-hosts/virustotal                       | 1.0     | not installed | 2019-06-24 |   | * |
  | recon/hosts-locations/migrate_hosts                | 1.0     | not installed | 2019-06-24 |   |   |
  | recon/hosts-ports/binaryedge                       | 1.0     | not installed | 2019-06-24 |   | * |
  | recon/hosts-ports/shodan_ip                        | 1.2     | not installed | 2020-07-01 | * | * |
  | recon/locations-locations/geocode                  | 1.0     | not installed | 2019-06-24 |   | * |
  | recon/locations-locations/reverse_geocode          | 1.0     | not installed | 2019-06-24 |   | * |
  | recon/locations-pushpins/flickr                    | 1.0     | not installed | 2019-06-24 |   | * |
  | recon/locations-pushpins/shodan                    | 1.1     | not installed | 2020-07-07 | * | * |
  | recon/locations-pushpins/twitter                   | 1.1     | not installed | 2019-10-17 |   | * |
  | recon/locations-pushpins/youtube                   | 1.2     | not installed | 2020-09-02 |   | * |
  | recon/netblocks-companies/censys_netblock_company  | 1.0     | not installed | 2019-08-22 |   | * |
  | recon/netblocks-companies/whois_orgs               | 1.0     | not installed | 2019-06-24 |   |   |
  | recon/netblocks-hosts/censys_netblock              | 1.0     | not installed | 2019-08-22 |   | * |
  | recon/netblocks-hosts/reverse_resolve              | 1.0     | not installed | 2019-06-24 |   |   |
  | recon/netblocks-hosts/shodan_net                   | 1.2     | not installed | 2020-07-21 | * | * |
  | recon/netblocks-hosts/virustotal                   | 1.0     | not installed | 2019-06-24 |   | * |
  | recon/netblocks-ports/census_2012                  | 1.0     | not installed | 2019-06-24 |   |   |
  | recon/netblocks-ports/censysio                     | 1.0     | not installed | 2019-06-24 |   | * |
  | recon/ports-hosts/migrate_ports                    | 1.0     | not installed | 2019-06-24 |   |   |
  | recon/ports-hosts/ssl_scan                         | 1.0     | not installed | 2020-04-13 |   |   |
  | recon/profiles-contacts/bing_linkedin_contacts     | 1.1     | not installed | 2019-10-08 |   | * |
  | recon/profiles-contacts/dev_diver                  | 1.1     | not installed | 2020-05-15 |   |   |
  | recon/profiles-contacts/github_users               | 1.0     | not installed | 2019-06-24 |   | * |
  | recon/profiles-profiles/namechk                    | 1.0     | not installed | 2019-06-24 |   | * |
  | recon/profiles-profiles/profiler                   | 1.0     | not installed | 2019-06-24 |   |   |
  | recon/profiles-profiles/twitter_mentioned          | 1.0     | not installed | 2019-06-24 |   | * |
  | recon/profiles-profiles/twitter_mentions           | 1.0     | not installed | 2019-06-24 |   | * |
  | recon/profiles-repositories/github_repos           | 1.1     | not installed | 2020-05-15 |   | * |
  | recon/repositories-profiles/github_commits         | 1.0     | not installed | 2019-06-24 |   | * |
  | recon/repositories-vulnerabilities/gists_search    | 1.0     | not installed | 2019-06-24 |   |   |
  | recon/repositories-vulnerabilities/github_dorks    | 1.0     | not installed | 2019-06-24 |   | * |
  | reporting/csv                                      | 1.0     | not installed | 2019-06-24 |   |   |
  | reporting/html                                     | 1.0     | not installed | 2019-06-24 |   |   |
  | reporting/json                                     | 1.0     | not installed | 2019-06-24 |   |   |
  | reporting/list                                     | 1.0     | not installed | 2019-06-24 |   |   |
  | reporting/proxifier                                | 1.0     | not installed | 2019-06-24 |   |   |
  | reporting/pushpin                                  | 1.0     | not installed | 2019-06-24 |   | * |
  | reporting/xlsx                                     | 1.0     | not installed | 2019-06-24 |   |   |
  | reporting/xml                                      | 1.1     | not installed | 2019-06-24 |   |   |
  +---------------------------------------------------------------------------------------------------+

  D = Has dependencies. See info for details.
  K = Requires keys. See info for details.

```


## automation


1. Domain list:
   - a text file containing the list of top domains in the working directory of auto-recon-ng.
   - For example: bing.com microsoft.com

2. Modules list:
   - `Subdomain Enumeration`:
     - Create a text file containing the list of modules, for subdomain enumeration use the below list.

```
When using the below list the “domain” option must be used with auto-recon-ng.

recon/domains-domains/brute_suffix
recon/domains-hosts/bing_domain_api
recon/domains-hosts/bing_domain_web
recon/domains-hosts/brute_hosts
recon/domains-hosts/builtwith
recon/domains-hosts/certificate_transparency
recon/domains-hosts/google_site_api
recon/domains-hosts/google_site_web
recon/domains-hosts/hackertarget
recon/domains-hosts/mx_spf_ip
recon/domains-hosts/netcraft
recon/domains-hosts/shodan_hostname
recon/domains-hosts/ssl_san
recon/domains-hosts/theharvester_xml
recon/domains-hosts/threatcrowd
recon/hosts-hosts/bing_ip
recon/hosts-hosts/ssltools
recon/hosts-ports/shodan_ip
```

   - Netblock to host discovery:
     - Create a text file containing the list of modules, for host enumeration use the below list.

```
When using the below list the “netblock” option must be used with auto-recon-ng.

recon/netblocks-hosts/reverse_resolve
recon/netblocks-hosts/shodan_net
recon/netblocks-ports/census_2012
recon/netblocks-ports/censysio

```


py file


```py

$ auto_recon-ng.py [-h] -w WORKSPACE [-i FILENAME] [-m MODULENAME] [-company DBNAME1] [-domain DBNAME2] [-netblock DBNAME3]

#!/usr/bin/python
import sys
import subprocess
import re
import itertools
import os
import platform
import argparse
import datetime
import time



def _readfile(dbname):
	if dbname:
		lines = dbname
		lines = [line.rstrip('\n') for line in lines]
		dblist = lines
		return dblist
	else:
		dblist = None


def _reconsetup():

	print "\n"
	print "---------------------------------------------------------------------------------------------------------------"
	print "AUTO RECON-NG - Automated script to run all modules for a specified list of domains, netblocks or company name"
	print "---------------------------------------------------------------------------------------------------------------"
	print "\n"

	wspace = "-w"

	parser = argparse.ArgumentParser()
	parser.add_argument('-w', '--workspace', type=str, help="Workspace name", required=True)
	parser.add_argument('-i', dest='filename', type=argparse.FileType('r'), help="Set the recon-ng source using this list", default=None)
	parser.add_argument('-m', dest='modulename', type=argparse.FileType('r'), help="Specify the modules list", default=None)
	parser.add_argument('-company', dest='dbname1', type=argparse.FileType('r'), help="Specify the file containing company names", default=None)
	parser.add_argument('-domain', dest='dbname2', type=argparse.FileType('r'), help="Specify the file containing domain names", default=None)
	parser.add_argument('-netblock', dest='dbname3', type=argparse.FileType('r'), help="Specify the file containing netblocks", default=None)
	args = parser.parse_args()

	wspace += args.workspace

	if args.dbname1:
		dblist= _readfile(args.dbname1.readlines())
		_db_companies(dblist,wspace)

	if args.dbname2:
		dblist = _readfile(args.dbname2.readlines())
		_db_domains(dblist,wspace)

	if args.dbname3:
		dblist = _readfile(args.dbname3.readlines())
		_db_netblocks(dblist,wspace)

	if args.filename:
		lines = args.filename.readlines()
		lines = [line.rstrip('\n') for line in lines]
		domainList = lines
	else:
		domainList = None
		print "Domain file not specified, recon-ng will run with existing database"

    # import the module to list
	if args.modulename:
		lines2 = args.modulename.readlines()
		lines2 = [line.rstrip('\n') for line in lines2]
		moduleList = lines2
	else:
		moduleList = None

        if domainList is not None:
                for src in domainList:
                        for mod in moduleList:
				_reconmod(wspace,mod,src)
        else:
        	print "No sources specified, recon-ng will run with the default settings!"

    # if list not none
	if moduleList is not None:
		for mod in moduleList:
			_reconmod(wspace,mod,"default")
		else:
			print "No modules specified, recon-ng report is being generated!"
	_reportgen(wspace)



def _reconmod(wspace,mod,src):
	modarg = "-m" + mod
	if src:
		srcarg = "-o source=" + src
		proc = subprocess.call(["recon-cli", wspace, modarg, srcarg,"-x"])
        # recon-cli , wspace, -m mod, o source=sr, -x
	else:
		print "No modules specified, recon-ng report for the current workspace is being generated!"


def _reportgen(wspace):
	report_list = ["reporting/csv", "reporting/html", "reporting/json", "reporting/list", "reporting/xlsx", "reporting/xml"]
	reportfiles ="/root/"
	stamp = wspace
	stamp += datetime.datetime.now().strftime('%M_%H-%m_%d_%Y')
	for rep in report_list:
		modarg= "-m" + rep
		ext =  re.split('/',rep)[1]
		srcarg = "-o FILENAME=" + reportfiles + stamp + "." + ext
		reportarg1 = "-o CREATOR = AutoRecon-ng"
		reportarg2 = "-o CUSTOMER = haxorhead"
		proc = subprocess.call(["recon-cli", wspace, modarg, reportarg1, reportarg2, srcarg,"-x"])


def _db_companies(dblist,wspace):
	print "Loading database with companies"
	for i in dblist:
		proc = subprocess.call(["recon-cli", wspace, "-C query insert into companies (company) values ('" + i + "');" ,"-x"])

def _db_domains(dblist,wspace):
	for i in dblist:
        	proc = subprocess.call(["recon-cli", wspace, "-C query insert into domains (domain) values ('" + i + "');" ,"-x"])

def _db_netblocks(dblist,wspace):
	for i in dblist:
        	proc = subprocess.call(["recon-cli", wspace, "-C query insert into netblocks (netblock) values ('" + i + "');" ,"-x"])


if __name__== "__main__":

	 _reconsetup()

```













.
