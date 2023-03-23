---
title: Linux - Create Own RPM package in CentOS 8
date: 2020-07-16 11:11:11 -0400
categories: [30System, CentOS8]
tags: [Linux, CentOS8]
math: true
image:
---


# Create Own RPM package in CentOS 8

[toc]

Assignment under:

LFCE: Advanced Network and System Administration / Advanced Package Management - [LFCEbyPluralsight](https://app.pluralsight.com/library/courses/advanced-network-system-administration-lfce/table-of-contents)

---

## build a RPM package from source.

http://vault.centos.org/ -> source -> binary RPM
centos 8.1.1911 -> os/ -> source/ -> Spackages/ -> rpm -> save link as


1. install "Development Tools"
    - include all requirement needed to compile and build RPM from source code

```c
$ sudo yum group install "Development Tools"
```

2. get the source rpm

```c
$ ls
procps-ng-3.3.15-1.el8.src.rpm
```

3. build from source rpm

```c
$ rpmbuild --rebuild procps-ng-3.3.15-1.el8.src.rpm
// install the rpm
// going though and start to complie the software
Installing procps-ng-3.3.15-1.el8.src.rpm
error: Failed build dependencies:
	systemd-devel is needed by procps-ng-3.3.15-1.el8.x86_64

  $ sudo yum install systemd-devel
```

4. get the rpmbuild directory

```c
$ ls
demo     Documents  index.html  Pictures  rpmbuild   Videos
Desktop  Downloads  Music       Public    Templates
// prefer to use user to install not root.
```

>rpmbuild/:
BUILD  BUILDROOT  RPMS  SOURCES  SPECS  SRPMS

>rpmbuild/BUILD:

>rpmbuild/BUILDROOT:

>rpmbuild/RPMS:
noarch  x86_64

>rpmbuild/RPMS/noarch:
procps-ng-i18n-3.3.15-1.el8.noarch.rpm

>// build constructed RPM
rpmbuild/RPMS/x86_64:
procps-ng-3.3.15-1.el8.x86_64.rpm
procps-ng-debuginfo-3.3.15-1.el8.x86_64.rpm
procps-ng-debugsource-3.3.15-1.el8.x86_64.rpm
procps-ng-devel-3.3.15-1.el8.x86_64.rpm

>// source file
rpmbuild/SOURCES:
procps-ng-3.3.15-1.el8.x86_64.tar.gz

>// spec file
rpmbuild/SPECS:
procps.spec

>rpmbuild/SRPMS:


## build a RPM package from code

1. prepare the file

rpm.spec:

```c
// under the SPEC path, temple for sepc
$ vi newfile.spec


1.

Name:       hello
Version:    1
Release:    1%{?dist}
Summary:    simple hello world program in a package
License:    MIT
URL:        www.pluralsight.com
source0:    /home/server1/rpmbuild/SOURCES/hello-1.tar.gz
//...name...
// configuration sections
// special directives that tell RPM what to do at certain parts of the build phase.

%description
simple hello world program in a package

%prep
%setup -q
// prep section
// caused the command setup -q
// creates the data directories in build process and uncompresses source tarball

%build
#make // either one words
gcc -o hello hello.c
// actually builds the software

%install
rm -rf $RPM_BUILD_ROOT/usr/local/bin/
mkdir -p $RPM_BUILD_ROOT/usr/local/bin/
install -m 755 hello $RPM_BUILD_ROOT/usr/local/bin/hello
// install section
// are the directories on what to do while building software package in our build directory.
// kind of a temporary working space where software is compiled
// first, delete a directory if it exists.
// Then recreate that directory, give a blank space to work in
// and then finally et up some permissions on particular binary that compiling and distributing.

%clean
rm -rf $RPM_BUILD_ROOT
// clean the working place

%files
/usr/local/bin/hello
// files that need to be package inside the rpm


2.

Name:           hello
Version:        2.10
Release:        1%{?dist}
Summary:        The "Hello World" program from GNU

License:        GPLv3+
URL:            http://ftp.gnu.org/gnu/%{name}
Source0:        http://ftp.gnu.org/gnu/%{name}/%{name}-%{version}.tar.gz

BuildRequires: gettext

Requires(post): info
Requires(preun): info

%description
The "Hello World" program package

%prep
%autosetup

%build
%configure
make %{make_build}

%install
%make_install
%find_lang %{name}
rm -f %{buildroot}/%{_infodir}/dir

%post
/sbin/install-info %{_infodir}/%{name}.info %{_infodir}/dir || :

%preun
if [ $1 = 0 ] ; then
/sbin/install-info --delete %{_infodir}/%{name}.info %{_infodir}/dir || :
fi

%files -f %{name}.lang
%{_mandir}/man1/hello.1.*
%{_infodir}/hello.info.*
%{_bindir}/hello

%doc AUTHORS ChangeLog NEWS README THANKS TODO
%license COPYING

#%changelog
#* Tue May 28 2019 Aaron Kili
```

hello.c

```c
#include <hellomake.h>
int main() {
  // call a function in another file
  myPrintHelloMake();

  return(0);
}
```

Makefile

```c
hellomake: hellomake.c hellofunc.c
     gcc -o hellomake hellomake.c hellofunc.c -I.
```

---

2. create the source code.

```c
$ mkdir hello-1
// 2 file inside
hello-1/
hello-1/hello.c    // the c program
hello-1/Makefile   // makefile to compile it

$ tar -czvf hello-1.tar.gz /home/server1https://github.com/ocholuo/language/tree/master/0.project/webdemo/m4/hello-1


$ ls
hello-1.tar.gz   // the source file
hello.spec       // the metadata, name, version, url, source, configuration


// put in the right place
$ mv hello-1.tar.gz ~/rpmbuild/SOURCES/
$ mv hello.spec ~/rpmbuild/SPECS/

```

3. install from the source code

```c
$ cd ~/rpmbuild/SOURCES
$ wget http://ftp.gnu.org/gnu/hello/hello-2.10.tar.gz -P ~/rpmbuild/SOURCES

$ cd ~/rpmbuild/SPECS
$ rpmdev-newspec hello
$ ls
hello.spec
// modify the spec file
```


4. build the rpm

```c
// built the rpm from source file
$ cd rpmbuild/SPECS/
$ rpmbuild -ba hello.spec

After the build process, the source RPMs and binary RPMs wills be created in the ../SRPMS/ and ../RPMS/ directories respectively.
use the rpmlint program to check and ensure that the spec file and RPM files created conform to RPM design rules:

// have the source rpm now
$ ls -R rpmbuild/

// install
$ sudo rpm -ivh rpmbuild/RPMS/x86_64/hello-2.10-1.el8.x86_64.rpm

// query it
$ rpm -qi hello
```








.
