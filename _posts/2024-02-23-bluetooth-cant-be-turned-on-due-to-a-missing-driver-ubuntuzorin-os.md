---
published: true
date: 2023-12-15
title: Bluetooth can’t be turned on due to a missing driver (Ubuntu/Zorin OS)
---
Check the result for this:

    sudo dmesg |grep -i bluetooth

If this error appears in the result, your OS is missing a Bluetooth driver:

    [ 3.935429] Bluetooth: Patch file not found ar3k/AthrBT_0x11020000.dfu

Install the missing driver!