---
layout: post
title: spectate Zain CTF Network-Security (Easy)
---


# pcap File Analysiz 
![](https://website-cybertalents.s3.us-west-2.amazonaws.com/Competitions/Zain+CTF+2022+Cover+(1).jpg)

[Challenge Link](https://github.com/hithmast/hithmast.github.io/raw/master/zAinCTF/spectate.pcap)
# At First

Open spectate.pcap File With Wireshark

[ ![wireshark]({{ site.baseurl }}/zAinCTF/images/sws.png)]({{ site.baseurl }}/zAinCTF/images/sws.png)

i like to start with protocol Heirarchy for better vision of Network
statistics -> Protocol Heirarchy 

[ ![wireshark-protocol-Heirarchy]({{ site.baseurl }}/zAinCTF/images/swe.png)]({{ site.baseurl }}/zAinCTF/images/swe.png)

There is a lot of data in UDP then Right-click And Apply it as Filter 
And Right-click on any packet and Follow UDP Stream 

[ ![wireshark-UDP-Stream]({{ site.baseurl }}/zAinCTF/images/6a3u47.gif)]({{ site.baseurl }}/zAinCTF/images/6a3u47.gif)

### But All is Encrypted Data and Not Usefull ! Don't Know Yet 
![idk-comic](https://sheepforcomics.files.wordpress.com/2013/08/so-you-dont-know.png)

### Go Back Again and Apply Filter For TCP data 
i Followed The TCP Stream 
And Got That Message Sent From Source 10.0.0.21 To 10.0.0.24 Through TCP Ports (60596,6000) 

---
Hello, Agent; 
I will send private data through port 7754 
use password: d0L1x65900Q

[ ![wireshark-data-tcp]({{ site.baseurl }}/zAinCTF/images/swr.png)]({{ site.baseurl }}/zAinCTF/images/swr.png)

---
set filter to "tcp.port == 7754" and Follow The TCP Stream  
in Stream 5 i Recognized The Mnemonic Of Zip Archive "PK"

[ ![wireshark-data-zip]({{ site.baseurl }}/zAinCTF/images/swl.png)]({{ site.baseurl }}/zAinCTF/images/swl.png)

Save As flag.zip File 
The File Has Password i used "d0L1x65900Q" 

#The End

[ ![](https://i.pinimg.com/originals/b1/1d/58/b11d588e49effe612aeb1e614f273843.jpg)](https://i.pinimg.com/originals/b1/1d/58/b11d588e49effe612aeb1e614f273843.jpg)
