---
title: LVM(Logical Volume Manager)-개념, 사용 방법
date: 2023-09-17 15:54:32 +0900
author: kkankkandev
categories: [Linux]
tags: [centos, linux, lvm, storage]     # TAG names should always be lowercase
comments: true
---

## 1. LVM(Logical Volume Manager)의 개념

LVM이란 리눅스와 유닉스 기반 운영체제에서 Logical Volume을 효율적이고 유연하게 관리하기 위한 커널의 한 부분이자 프로그램입니다. LVM은 물리적인 디스크를 논리적인 볼륨으로 추상화하여 유연하게 디스크 공간을 활용할 수 있게 해줍니다.

-> **LV2를 사용 중 디스크 용량이 더 필요하다면?? => 디스크를 추가 장착 후 VG에 추가 후 LV2에 볼륨 사이즈를 추가로 할당가능!**

![그림 1 - LVM 개요](https://github.com/War-Oxi/war-oxi.github.io/assets/72260110/0ec52fc0-1d5d-4d77-b8c2-4e78aa9f80ad)


## 2. LVM의 구성요소

LVM은 PV, PE, VG의 총 5가지로 구성됩니다.

### 2.1. PV(Physical Volume)

LVM에서 블록 장치를 사용하려면 PV로 초기화 후 사용이 가능합니다. PV란 블록 장치 전체 또는 그 장치를 이루고 있는 파티션들을 LVM에서 사용할 수 있게 변환한 것을 말합니다.  

PE - PV는 데이터를 저장하는 각각의 고정된 크기의 논리적 블록으로 나뉘어지는데, 이를 PE(Physical Extent)라고 합니다. PV는 여러개의 PE로 구성됩니다.

![그림 2 - PV의 구조](https://github.com/War-Oxi/war-oxi.github.io/assets/72260110/f55b31f6-f275-431a-a6ff-9005b0394e0a)


### 2.2 VG(Volume Group)

VG는 PV들을 모아놓은 것으로. LV를 할당할 수 있는 공간입니다
즉 PV로 초기화된 장치들은 VG로 통합되게 되고, VG안의 공간을 사용해 LV를 만들 수 있습니다.

![그림 3 - VG의 구조](https://github.com/War-Oxi/war-oxi.github.io/assets/72260110/5dc65b25-1b23-4a90-90c9-24813362e5f7)

### 2.3 LV(Logical Volume)

LV는 VG에서 공간을 할당받아 사용자가 최종적으로 다룰 수 있는 논리적인 스토리지(실제 데이터를 저장)입니다. LV는 PV와 동일하게 LE(Logical Extent)라는 일정한 크기의 블록단위로 나뉘며 각각의 LE들은 PE와 1:1로 매핑되어 사용됩니다.  

**LV는 LE들의 집합이며 LE는 PE와 1:1로 매핑, VG에서 필요한 크기를 할당받아 필요시 크기를 조정가능**

![그림 4 - LV의 구조](https://github.com/War-Oxi/war-oxi.github.io/assets/72260110/8d8def48-5704-444c-8a9a-0c9febfcb1a5)

## 3. LVM 사용법

LVM을 사용하기 위해서는 아래의 절차가 필요합니다.

##### **PV 생성 -> VG 생성 -> LV 생성 -> LV 포맷 -> mount -> /etc/fstab 설정**

아래의 LVM을 CentOS7 을 사용해 구축하면서 사용법을 기술하겠습니다.

![그림 5](https://github.com/War-Oxi/war-oxi.github.io/assets/72260110/20111d13-ec74-4de6-b3f8-636c193c78d2)

------

#### 디스크 확인

현재 sdb, sdc, sdd라는 5G까지 용량을 가진 sdb, sdc, sdd라는 디스크가 있습니다.

```
# 서버에서 인식하고 있는 디스크들 Tree 형태로 확인
lsblk 

# 전체 디스크 목록과 디스크 파티션 목록 확인
fdisk -l 

# 마운트 되어 있는 디스크 확인
df -h
```


![image](https://github.com/War-Oxi/war-oxi.github.io/assets/72260110/bc2805c2-eaa0-42d8-9f7e-41a9e64ecaa0)


### 3.1 PV 만들기

#### 1. lvm2 package 다운로드

```
yum -y install lvm2
```

#### 2. 파티션 생성 (sdb, sdc, sdd 모두에 적용)

```
fdisk /dev/sdb

=> n # 새 파티션 생성
	
Command :

	p
		{Enter} #primary
		{Enter} #파티션 개수
		{Enter} #Frist Sector 지정
		{Enter} #Last Sector 지정
		

=> t # 파일 시스템 타입 설정 8e -> LVM으로 사용될 파티션

=> p # 현재 설정한 정보 확인

=> w # 현재 설정한 정보대로 Write
```

![image](https://github.com/War-Oxi/war-oxi.github.io/assets/72260110/fe14231b-5f3c-442f-bf4f-5e2a1ed0371f)

#### 3. PV 생성

![image](https://github.com/War-Oxi/war-oxi.github.io/assets/72260110/64c02a4f-7ea2-474e-afcc-5d829d8381e4)

```
pvcreate {파티션명}
```

### 3.2 VG 만들기

![image](https://github.com/War-Oxi/war-oxi.github.io/assets/72260110/df15168a-a387-4fdd-b5c4-040a7dd12b09)

```
vgcreate {VG 이름} {VG에 포함될 PV목록}
```

### 3.3 LV 만들기

![image](https://github.com/War-Oxi/war-oxi.github.io/assets/72260110/508eeb87-4b11-4aa7-81a6-b966d990a366)

```
lvcreate --size {lv 사이즈} --name {lv 이름} {VG 이름}

## PV의 용량을 5x3개 15로 했으나 실제로는 15G보다 작은 용량이 할당되게 됩니다. 따라서 lv1을 만들고 남은 VG의 용량을 모두 lv2에 할당하는 명령어를 사용했습니다.
lvcreate --extents 100%FREE --name {lv 이름} {VG 이름}
```

### 3.4 LV 포맷 (ext4 형식으로)

```
mkfs.ext4 {LV 경로}
```

![image](https://github.com/War-Oxi/war-oxi.github.io/assets/72260110/aa276a3e-13a2-4de0-b771-f3e6667d58a0)

### 3.5 Mount 및 /etc/fstab 설정

```
mount -t {파일 타입} {마운트 할 LV} {LV가 마운트 될 디렉토리}
```

![image](https://github.com/War-Oxi/war-oxi.github.io/assets/72260110/23e32bd0-c30b-4c36-a20c-f160c8cc4c6f)

#### /etc/fstab 설정 (vi Editor 사용)

![image](https://github.com/War-Oxi/war-oxi.github.io/assets/72260110/7b2887a1-ac4c-4530-be24-840b648b4eec)

	12행, 13행에 lv1, lv2 추가
	/etc/fstab을 설정해주지 않을 시 재부팅 하면 마운트가 풀립니다.
	/etc/fstab을 설정 후 정상적으로 재부팅이 되고 mount가 잘 되어있는지 확인 후 사용해주세요.

### 3.6 확인

생성된 PV, VG, LV 상태는 아래의 명령어로 확인 가능합니다

```
## PV상태확인
pvdisplay #자세하게 확인
pvscan    #간단하게 확인

## VG상태확인
vgdisplay
vgscan

## LV상태확인
lvdisplay
lvscan
```

#### [결과1] PV 상태

![image](https://github.com/War-Oxi/war-oxi.github.io/assets/72260110/e05f9595-44cb-4c3e-aefa-b1f01a8edb95)

#### [결과2] VG 상태

![image](https://github.com/War-Oxi/war-oxi.github.io/assets/72260110/8241aa78-ac94-44f2-a3e8-be9d32e5b5fb)

#### [결과3] LV 상태

![image](https://github.com/War-Oxi/war-oxi.github.io/assets/72260110/9039d738-4093-4d67-9a95-b23eeba95612)

## 4. 그 외 명령어

```
# VG 명령어

## VG 삭제
vgremove {vg_name}

## VG 확장
vgextend {vg_name} {pv_name}

## VG 안에 있는 PV 삭제
vgreduce {vg_name} {pv_name}


# LV 명령어

## LV 삭제
lvremove {lv_name}

## LV 크기를 {+Size}만큼 확장
lvextend -L {+Size} {lv_path}

## LV 크기를 {-Size}만큼 축소
lvreduce -L {Size} {lv_path}

# 파일 시스템 및 포맷, 사이즈 재 할당 명령어

## ext 파일시스템 타입의 LV 사이즈 재조정
resize2fs {lv_path}

## xfs 파일시스템 타입의 LV 사이즈 재조정
xfs_growfs {lv_path}
```



<br>

<strong>궁금하신점이나 추가해야할 부분은 댓글이나 아래의 링크를 통해 문의해주세요.</strong>   
Written with [KKam.\_\.Ji](https://www.instagram.com/kkam._.ji/)
