
# 10 concurrency 1

## Lock 변천사
1. all interrupt 차단 -> working only single processor
2. 전역 flag로 다른 락 차단 -> no mutual exclusion
3. Test and Set				 : hw atomic func -> no fairness, 항상 업데이트
4. Compare and Swap 			: hw atomic func, 비교 후 업데이트
5. Load-Linked and Store-Conditional 		: 
6. Fetch and add 				: hw atomic func 
7. Ticket lock 				: fairness, 기다린 거 반영

spin lock 비효율 -> os support

8. Yield					: spin -> give up CPU
9. park/unpark 				: queue, sleep 
10. setpark ,, syscall			: atomic 문제로 인한 sleep forever 해결, sleep 무시
11. futex 					: 31번 최상위 비트 
						0이면 lock 가능
						1이면 이미 어디서 lock
futex_wait
futex_wake
-----------------------

## Lock in Linux
pthread_mutex_lock
pthread_mutex_unlock

lock 초기화 : 초기: 매크로로 or dynamic: 함수로

pthread_mutex_trylock
pthread_mutex_timelock 
-----

공유 변수 사이에 lock

Thread 증가 : Time 증가 : Lock 오버헤드
해결 : Sloppy counter
	threshold S : 5 만큼 차면 global에 반영

critical section 최소화


linked list : lock per node
queue : head, tail lock + dummy node
hash table : lock per hash bucket


-------------------------------------

# 11 concurrency 2

## 부모 프로세스가 자식 프로세스 종료 여부 확인 법!? 무한히 spin...? ㄴㄴ
-> condition variable
waiting, signaling

pthread_cond_wait		: sleep
pthread_cond_signal	: unblock one of threads
-> 위아래 mutex_lock 꼭 필요

exit안에 signal
join 안에 wait

join -> exit, exit -> join 어떤 순서일지 모름 -> condition var, lock 둘다 꼭 필요

## producer/consumer 문제 = p1 c2인 경우,
1) p1이 c1한테 날렸는데 c2가 채감, c1은 버퍼 없어서 놀람 -> recheck 후 다시 wait 필요
2) c1이 p1한테 signal날렸는데 c2가 받아버림, 버퍼 없네 놀람 -> condition val 여러개로 empty, fill
						producer wait : empty, signal : fill
						consumer wait : fill,      signal : empty
3) buffer 여러개(배열 : int buffer[ ]) -> 더 효율

allocate 문제 : 
누구를 깨워야할지 모를때 -> 다 깨우고 되는 애 해, 나머지 다시 자(recheck) : pthread_cond_broadcast


## Semaphore
대기자 : int value
sem_init(&s, 0, X);  	: semaphore X로 초기화
sem_wait    : semaphore -=1	if negative( < 0 ) -> sleep
semp_post  : semaphore +=1	waiting 중에서 -> wake one

what can X be!?
ex) 0으로 초기화, semaphore -3 : 대기자 3	: 마치 condition var

ex) 1로 초기화 -> mutex lock과 같음	//switch시 다른 스레드 sleep
semaphore 1에서 wait 0보다 크니 사용, switch 다른스레드에서 wait하니 -1 -> sleep  : 마치 Lock
다시 본 스레드로 복귀 다쓰고 post, 0됨, 아까 스레드 wake, 다 쓰고 post 다시 1


## Thread Trace able case
1) 1st parent sem_wait, 2nd child sem_post
2) 1st child sem_post,   2nd parent sem_post

## producer/consumer using semaphore
1) put in producer
sem_wait(&empty)
put(i)
sem_post(&full)

2) get in consumer
sem_wait(&full)
get(i)
sem_post(&empty)

-> But No mutual exclusion
sol 1 :
sem_wait(&mutex)	: like Lock
sem_wait(&empty)			-> sleep 위험 있음
put(i)
sem_post(&fill)
sem_post(&mutex)	: like Unlock

sol 2 :	(final)
sem_wait(&empty)	
sem_wait(&mutex)	: like Lock	-> critical section 최소화
put(i)
sem_post(&mutex)	: like Unlock
sem_post(&fill)


## Reader-Writer Locks
writerlock only 1

Dining Philosophers
원탁에 5명 포크로, 반드시 두 포크 필요, 한명 먹을때 옆사람 못먹음

## Zemaphore
pthread_cond_t
pthread_mutex_t를 둘다 이용해서 마치 세마포어처럼 사용

--------------------------------------

# 12 concurrency 3

## Non-Deadlock Bugs
1) atomicity violation	-> causing no lock		-> use Lock
2) order violation		-> thread's dependency	-> enforce ordering using condition var

## Deadlock Bugs
Thread 1		Thread 2		-> a cycle
lock L1		lock L2
lock L2		lock L1

reason
1) complex dependencies
2) encapsulation

four condition occuring deadlock	-> if No deadlock, break these
1) Mutual exclusion
2) Hold and wait
3) No preemption
4) Circular wait

## Prevention deadlock
1) Circular wait
total ordering 	-> very hard
2) Hold and wait
all locks at once, atomically		-> performance, parallel 감소
3) No preemption
trylock()			-> 완전 lock이 아닌 lock 시도만, 실패하면 풀기
			-> 두 스레드 동시에 같은 코드일때 livelock	-> random delay 부여
4) Mutual exclusion
HW instruction "CompareAndSwap"
Lock-free
링크드 리스트에서 do ~ while

## via Scheduling
	T1 T2 T3 T4
L1	y   y   y   y  
L2	y   y   y   n

cpu1 t3 t4	-> deadlock 발생 가능성 lock 동시 실행 막음
cpu2 t1 t2	-> 순차적으로


##
deadlock detector 주기적
resouce graph
deadlock -> restart


-------------------------------------------

# 13 IO devices
cpu 	memory
		memory bus
		general i/o bus
		peripheral i/o bus

i/o chip -> slow

## Canonical Device
1) HW interface	(reg : status, command, data)
2) internal	(내부 동작 수행 : cpu, memory, etc)

HW interface
1-1) status reg		: current status		ex) while( STATUS == BUSY )
1-2) command reg		: perform certain task
1-3) data reg		: pass, get data

## Polling
waste cpu time	<- spinning
cpu 11111 ppppp 11111
disk 	  11111

## Interrupts
cpu 11111 22222 11111
disk 	  11111

-> 항상 interrupts가 좋은 게 아님. device가 fast -> context switch 비용이 더 듬
device fast -> use polling
device slow -> use interrupt

## DMA
C : copy data from memory : over-burdened (cpu힘들어) 2를 복사해오는 것
cpu 1111 CCC 22222 111
disk	       11111

DMA : Direct Memory Access
cpu 1111 22222222 111
dma 	CCC
disk 	     11111

## Smart NIC
i/o faster	-> Frequent intterupts! 인터럽트 너무 많아
sol : Smart NIC
	기존			Smart NIC
cpu  application			application
       os				os
       os: tcp layer			
NIC   HW packet processing		os: tcp layer   //이동함
				HW packet processing


## Interaction os <-> device
1) i/o instructions		-> in, out specific device reg
2) memory-mapped i/o	-> Like device reg in memory locations
			    기존 os의 read, write 등으로 접근 가능

## IDE device driver
1) control reg 	8bit (1Byte)	-> reset, interrupt enable	0x3F6	E0이 interrupt enable
2) command block reg		-> 여러 주소 있는데 0x1F0 ~ 0x1F7
3) status reg 	8bit		-> 현상태	0x1F7
4) error reg	8bit		-> 에러 이유	0x1F1

io read -> interrupt handler B_VALID
io write -> interrupt handler B_DIRTY		-> wrtie 한번에 하나씩 밖에 안됨

LBA : logical block address
0x1F0 data port 128byte

SCSI, disks, .... 여러 디바이스 드라이버 어떻게 !?!?
sol : Abstraction
추가적인 디바이스 드라이버 없이 일반적으로 동작 가능
but 특이한 프로토콜(기존에 정의 안된) 은 사용 불가



## HDD
Track	
Sector(512 byte)

rotation delay : 판 돌리는 시간
seek time : head가 track 찾는 시간
track skew : 회전지연 고려해서 미리 돌려 놓음	(데이터는 시계방향) 돌아가는건 시계 반대
	track skew : 2 이면 12 ~ ~ 11 이런 느낌

multi-zoned : 안보다 밖이 sector 수 많음
cache (track buffer) : SRAM 8~16mb 
	write back : 캐시만 써놓고 disk 나중에 반영 (후처리 필요)
	write through : 캐시 disk 동시에 반영

### i/o time
T_io = Tseek + Trotation + Ttransfer		//헤드가 트랙 찾고 돌아가고 읽는
         <-------물리------> <-전자->

### I/O rate (속도)
Ri/o = Size_transfer / T_io

## 4kb random write
Tseek = 4ms
Trotation = 15000RPM -> 250RPS --> 1회전 4ms	/2  양면이라..
	-> avg rotation 2ms
Ttransfer = 4KB / 125 (mb/s) 	4kb = write 크기	1mb = 1024 kb
	= 30us	
T_io = 4ms + 2ms +30us = 6ms
R_io = write 크기( 4kb ) / T_io = 0.66mb/s

## Sequential Write	-> max transfer speed
Tseek = 4ms
Trotation = 150000RPM -> 250RPS --> 1회전 4ms 	/2
	-> avg rotation 2ms
Ttransfer =  100Mb / 125 Mb/s = 800ms		//100mb 쓰기, Max transfer 속도
T_io = 4ms + 2ms + 800ms = 800ms
R_io = 100Mb / 800ms = 125mb/s		//아까 사용한 Max transfer과 같음


IO
performance vs capacity
random write vs sequential write


### Disk Scheduling
1) FCFS	: 순서대로 처리
2) Shortest seek time first(SSTF) : 짧은거 먼저			-> default
3) SCAN(elevator algorithm) : 한쪽 끝까지 갔다가, 반대쪽 끝까지
4) C-SCAN : 한쪽 끝까지 갔다가, 반대쪽에서 다시 시작하기
5) C-LOOK : request 끝까지 갔다가, 반대쪽 끝 request까지 가기	-> default

동적으로 스케줄링 알고리즘 변경해야 최적

disk defragmentation 디스크 조각모음

I/O merging
연속된 요청올때까지 기다렸다가 한꺼번에 처리

------------------------------

## 14 RAID and File systems

### RAID
Redundant Arrays of Inexpensive Disks		: 여러 물리 디스크를 결합, 하나의 논리 디스크로 구성
1) Faster
2) Bigger
3) More reliable

RAID 0 ~ 6

capacity, reliablity, performance(read, write)

### RAID 0 : striping
disk0 disk1 disk2 disk3
0      1      2      3	-> striping : 여러 디스크에 분산 저장
			chunk size만큼 읽는듯 1block : 4kb
avg seek time 7ms
avg rotational delay 3ms
transfer rate 50mb/s

1) performance :
	10Mb seq 	S = 10Mb/(7+3+10mb/50mb/s) = 47.62mb/s
	10kb random 	R = 10KB/(7+3+10kb/50mb/s) = 0.981mb/s
2) capacity : N * disk 1개 용량
3) throughput :
	random r/w = N*R
	sequential r/w = N*S
	N*Speed up

Reliability : No support

### RAID 1
Mirroring

ex) RAID 10 (1 + 0) 	/ RAID 0 : disk0,1 <-> disk2,3 | RAID 1 : Mirrored (Mirroring Level = 2)
disk0 disk1 disk2 disk3
0      0      1      1
2      2      3      3

1) capacity : N * B / 2
2) reliability : can recover, 
3) throughput :
	sequential write	: N/2*S
	sequential read	: N/2*S	-> 두 디스크가 같아 하나의 디스크에서 순차적으로 읽음
	random write 	: N/2*R
	random read	: N*R	-> 두 디스크 병렬 읽기

### RAID 4
parity
disk0 disk1 disk2 disk3 disk4
0      1      2      3      p0
4      5      6      7      p1

0      0      1      1      p0 = xor(0,0,1,1) -> 하나 고장나면 복구
0      1      0      0      p1 = xor(0,1,0,0)

xor : 같으면 0, 다르면1    :     1의 개수 짝수로 유지

disk4 -> 데이터 업데이트마다 수정 필요

1) capacity (N-1)*B
2) 
sequential read (N-1)*S
sequential write (N-1)*S for full stripe
random read (N-1)*R
random write -> R/2	: 한번 데이터 삽입에 두 개의 디스크 접근 필요
	1. read all blocks
	2. update the block
	3. compute new parity
	4. write updated block

	additive parity update	-> 기존에 xor연산해서 parity
	subtractive parity update	-> 기존 데이터 제거, 새로 계산한 parity 삽입

### RAID 5
distribute parity blocks
disk0 disk1 disk2 disk3 disk4
0      1      2      3      p0
5      6      7      p1    4
10    11     p2    8      9
15    p3    12     13     14

-> 다른 disk에 parity가 있기에 병렬 가능

1) capacity (N-1)*B
2) reliability 1
3) performance
	sequential read, write 	(N-1)*S
	random read 		N*R
	random write		(N*R)/4	//한번 쓸때 패리티 수정 위해 다른 4개 디스크 같이














